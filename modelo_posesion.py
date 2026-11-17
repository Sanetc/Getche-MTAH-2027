"""
More Than a Hack 2027 — Fase 1, Nivel 3: modelo LightGBM + suavizado temporal
=============================================================================

Usa `datos.py` y `baseline_posesion.py` y agrega:
  1. Features ricas por fotograma (distancias, densidad de jugadores, dinámica
     del balón, ventanas temporales pasadas y futuras, salida de la regla).
  2. Clasificador LightGBM de 3 clases (Ninguno / Local / Visitante).
  3. Suavizado con Viterbi (HMM): la posesión "cuesta" cambiar de estado; y
     fusión de las posesiones más cortas que `min_frames`.
  4. Validación leave-one-out entre partidos: se prueba en uno y se entrena en
     el resto. Todo lo que se ajusta (parámetros de la regla, p_quedarse,
     min_frames) se ajusta solo con los partidos de entrenamiento y por
     spellF1, la métrica del ranking oficial (ver evaluacion.py).

Uso:
    python -I modelo_posesion.py --data ../datos/sample-data/data --out salida
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos as D  # noqa: E402
from datos import FPS, NINGUNO, LOCAL, VISITA  # noqa: E402
from baseline_posesion import (  # noqa: E402
    buscar_parametros, calcular_features, cargar_partido, listar_juegos, predecir_posesion,
)
from evaluacion import (  # noqa: E402
    CABECERA, HUECO_GRID, MIN_D_GRID, MIN_FRAMES_GRID, MIN_PD_GRID, evaluar, fila, fusionar_rachas_cortas,
    mantener_en_huecos, quitar_paradas_cortas, quitar_posesiones_entre_paradas, spell_f1,
)
from balon_estimado import (  # noqa: E402
    enmascarar_balon, entrenar_estimador, estimar_balon, features_colectivas, mostrar_balon_en_paradas,
    preparar_entrenamiento, ruido_balon, velocidades,
)
from tracking_sin_id import quitar_identidades  # noqa: E402
from smart_tagging import anclar_saques, features_smart_tagging, simular_set_pieces  # noqa: E402

Q_GRID = [0.97, 0.99, 0.995, 0.999, 0.9995]   # candidatos de p_quedarse para Viterbi


# ----------------------------------------------------------------------------
# 1. Features
# ----------------------------------------------------------------------------
def distancias_equipo(df, prefijo):
    """Matriz (frames x jugadores) de distancias al balón."""
    xs = D.columnas_jugadores(df, prefijo)
    X = df[xs].to_numpy()
    Y = df[[c[:-2] + "_y" for c in xs]].to_numpy()
    bx, by = df["ball_x"].to_numpy()[:, None], df["ball_y"].to_numpy()[:, None]
    d = np.sqrt((X - bx) ** 2 + (Y - by) ** 2)
    return np.where(np.isnan(d), np.inf, d)


def ventana(s: pd.Series, periodos: pd.Series, w: int, fn: str, futuro=False) -> pd.Series:
    """Rolling hacia el pasado (o el futuro) sin cruzar el cambio de periodo."""
    def aplicar(x):
        if futuro:
            x = x[::-1]
        r = getattr(x.rolling(w, min_periods=1), fn)()
        return r[::-1] if futuro else r
    return s.groupby(periodos.to_numpy()).transform(aplicar)


def features_colectivas_posesion(df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """
    Colocación y movimiento de cada equipo (sin balón) y posición del balón
    relativa a cada equipo. Pensadas para cuando el balón no se ve: quién
    avanza, quién retrocede, dónde está el bloque de cada equipo.
    """
    C = features_colectivas(df)
    quedarse = [c for c in C.columns if not c.endswith(("_n",))]       # el nº de jugadores vistos no aporta
    f = pd.concat([f, C[quedarse]], axis=1)
    for eq in ("l", "v"):
        f[f"ball_dx_{eq}"] = f["ball_x"] - C[f"{eq}_cx"]              # balón respecto al centroide
        f[f"ball_dy_{eq}"] = f["ball_y"] - C[f"{eq}_cy"]
        f[f"ball_d_{eq}"] = np.hypot(f[f"ball_dx_{eq}"], f[f"ball_dy_{eq}"])
        f[f"ball_dw_{eq}"] = np.hypot(f["ball_x"] - C[f"{eq}_wx"], f["ball_y"] - C[f"{eq}_wy"])
    # el local ataca hacia +x: su última línea es l_xmin, la del visitante v_xmax
    f["ball_tras_linea_l"] = f["ball_x"] - C["l_xmin"]
    f["ball_tras_linea_v"] = C["v_xmax"] - f["ball_x"]
    f["ball_d_h"] = np.hypot(f["ball_x"] - C["hx"], f["ball_y"] - C["hy"])   # punto caliente
    return f


def _duracion_rachas(m: np.ndarray, periodos: np.ndarray):
    """Para cada fotograma con m verdadero: fotogramas de la racha antes y después de él (sin cruzar periodos)."""
    n = len(m)
    atras, adelante = np.zeros(n), np.zeros(n)
    corte = np.r_[True, (m[1:] != m[:-1]) | (periodos[1:] != periodos[:-1])]
    ini = np.flatnonzero(corte)
    fin = np.r_[ini[1:], n]
    for a, b in zip(ini, fin):
        if m[a]:
            atras[a:b] = np.arange(b - a)
            adelante[a:b] = np.arange(b - a)[::-1]
    return atras, adelante


def features_parada(f: pd.DataFrame, df: pd.DataFrame, v_jug_matriz: np.ndarray, visto: np.ndarray) -> pd.DataFrame:
    """
    Pistas de balón parado CON el balón visible (prefijo par_): en los 5 min
    públicos (30 oct) el 36% de los fotogramas D se predecían como posesión y
    4 de las 6 posesiones falsas caían dentro de paradas con balón visible.
    Una parada así se ve como: balón quieto un buen rato, en un punto de
    reanudación (banda, fondo, córner, punto penal, centro, área chica) o fuera
    del campo, con casi todos los jugadores caminando. Son reglas del juego,
    valen en cualquier partido aunque los sample data casi no tengan estas
    paradas (las simula `mostrar_balon_en_paradas`).
    """
    L, A = D.LARGO / 2, D.ANCHO / 2
    x, y = f["ball_x"].to_numpy(), f["ball_y"].to_numpy()
    ax, ay = np.abs(x), np.abs(y)
    per = df["Period"].to_numpy()
    f["par_d_banda"] = A - ay                                  # negativo: fuera del campo
    f["par_d_fondo"] = L - ax
    f["par_d_corner"] = np.hypot(L - ax, A - ay)
    f["par_d_penal"] = np.hypot(ax - (L - 11), y)
    f["par_d_centro"] = np.hypot(x, y)
    f["par_d_area_chica"] = np.hypot(np.maximum(0, (L - 5.5) - ax), np.maximum(0, ay - 9.16))
    v = pd.Series(f["v_balon"].to_numpy()).rolling(5, center=True, min_periods=1).mean().to_numpy()
    quieto = (v < 1.0) & visto                                 # visto de verdad, no interpolado
    atras, adelante = _duracion_rachas(quieto, per)
    f["par_quieto_atras"] = np.clip(atras, 0, 1500)
    f["par_quieto_adelante"] = np.clip(adelante, 0, 1500)
    f["par_quieto_total"] = np.clip(atras + adelante, 0, 3000)
    bx, by = pd.Series(x, index=df.index), pd.Series(y, index=df.index)
    g = pd.Series(per, index=df.index)
    for w in (25, 75):                                         # cuánto se movió el balón en 1 y 3 s
        f[f"par_despl_pas{w}"] = np.hypot(bx - bx.groupby(g).shift(w), by - by.groupby(g).shift(w)).to_numpy()
        f[f"par_despl_fut{w}"] = np.hypot(bx.groupby(g).shift(-w) - bx, by.groupby(g).shift(-w) - by).to_numpy()
    with np.errstate(all="ignore"):
        lentos = pd.Series(np.nanmean(v_jug_matriz < 1.5, axis=1), index=df.index)
    f["par_frac_lentos"] = ventana(lentos, df["Period"], 25, "mean")
    f["par_frac_lentos_fut"] = ventana(lentos, df["Period"], 50, "mean", futuro=True)
    return f


def features_base(df: pd.DataFrame, colectivas=True, parada=False):
    """Features que no dependen de parámetros ajustables. Devuelve (F, base)."""
    base = calcular_features(df)
    DL, DV = np.sort(distancias_equipo(df, "H_"), 1), np.sort(distancias_equipo(df, "A_"), 1)
    periodos = df["Period"]

    f = pd.DataFrame(index=df.index)
    for k in range(3):                                   # 3 más cercanos por equipo
        f[f"dl_{k}"] = np.clip(DL[:, k], 0, 60)
        f[f"dv_{k}"] = np.clip(DV[:, k], 0, 60)
    f["dif_cercano"] = f["dl_0"] - f["dv_0"]
    for r in (3, 5, 10):                                 # densidad alrededor del balón
        f[f"n_local_{r}m"] = (DL <= r).sum(1)
        f[f"n_visita_{r}m"] = (DV <= r).sum(1)

    # Posición y dinámica del balón. Como el local ataca siempre hacia +x,
    # ball_x y vx_balon significan lo mismo en todos los periodos y partidos.
    bx = df["ball_x"].interpolate(limit=10)
    by = df["ball_y"].interpolate(limit=10)
    f["ball_x"], f["ball_y"] = bx, by
    f["v_balon"] = base["v_balon"]
    f["vx_balon"] = np.gradient(bx.to_numpy()) * FPS
    f["acc_balon"] = np.abs(np.gradient(f["v_balon"].to_numpy()) * FPS)
    f["balon_fuera"] = base["balon_fuera"].astype(int)
    # Si el balón viene relleno (estimado), `balon_visto` dice qué fotogramas
    # son de verdad y cuáles estimados; si no, lo visto es lo que no es NaN.
    visto = df["balon_visto"] if "balon_visto" in df else df["ball_x"].notna()
    f["balon_nan"] = (~visto).astype(int)

    # Pistas de balón parado que no dependen del balón: cuánto lleva sin verse
    # (hacia atrás y hacia adelante) y a qué velocidad van los 22 jugadores.
    idx = pd.Series(np.arange(len(df)), index=df.index)
    ultimo_visto = idx.where(visto).ffill().fillna(-1e4)
    proximo_visto = idx.where(visto).bfill().fillna(1e4)
    f["frames_sin_balon_atras"] = np.clip(idx - ultimo_visto, 0, 5000)
    f["frames_sin_balon_adelante"] = np.clip(proximo_visto - idx, 0, 5000)
    cols_x = D.columnas_jugadores(df, "H_") + D.columnas_jugadores(df, "A_")
    VX, VY = velocidades(df[cols_x].to_numpy(), df[[c[:-2] + "_y" for c in cols_x]].to_numpy())
    with np.errstate(all="ignore"):
        v_jug = pd.Series(np.nanmean(np.hypot(VX, VY), axis=1), index=df.index)
    f["v_jugadores"] = ventana(v_jug, periodos, 25, "mean")
    f["v_jugadores_50"] = ventana(v_jug, periodos, 50, "mean", futuro=True)

    # Contexto temporal: quién estuvo cerca antes y quién lo estará después.
    # (Mirar al futuro vale porque la tarea es offline sobre el partido completo.)
    for w in (5, 15, 50):
        for col in ("dl_0", "dv_0", "dif_cercano"):
            fn = "mean" if col == "dif_cercano" else "min"
            f[f"{col}_pas{w}"] = ventana(f[col], periodos, w, fn)
            f[f"{col}_fut{w}"] = ventana(f[col], periodos, w, fn, futuro=True)

    if colectivas:
        f = features_colectivas_posesion(df, f)
    if parada:
        with np.errstate(all="ignore"):
            f = features_parada(f, df, np.where(np.isnan(VX), np.nan, np.hypot(VX, VY)), visto.to_numpy())
    f["periodo"] = periodos.astype(int)
    return f, base


def agregar_regla(f: pd.DataFrame, base: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Salida de la regla del Nivel 2 como feature (y su contexto)."""
    f = f.copy()
    regla = predecir_posesion(base, **params)
    f["regla"] = regla
    f["regla_es_local"] = (regla == LOCAL).astype(int)
    for w in (25, 75):
        f[f"regla_local_frac{w}"] = f["regla_es_local"].rolling(w, center=True, min_periods=1).mean()
    return f.fillna(-1)


# ----------------------------------------------------------------------------
# 2. Suavizado temporal (Viterbi)
# ----------------------------------------------------------------------------
def viterbi(probs: np.ndarray, p_quedarse: float, periodos: np.ndarray) -> np.ndarray:
    """probs: (T, 3) probabilidades del modelo. Devuelve la secuencia más probable."""
    T, K = probs.shape
    p_cambio = (1 - p_quedarse) / (K - 1)
    logA = np.log(np.full((K, K), p_cambio) + np.eye(K) * (p_quedarse - p_cambio))
    logB = np.log(np.clip(probs, 1e-9, 1))

    score = np.empty((T, K))
    back = np.zeros((T, K), dtype=np.int8)
    score[0] = logB[0]
    for t in range(1, T):
        if periodos[t] != periodos[t - 1]:               # reinicio en el medio tiempo
            score[t] = logB[t] + score[t - 1].max()
            back[t] = score[t - 1].argmax()
            continue
        cand = score[t - 1][:, None] + logA              # (desde, hacia)
        back[t] = cand.argmax(0)
        score[t] = cand.max(0) + logB[t]

    ruta = np.empty(T, dtype=int)
    ruta[-1] = score[-1].argmax()
    for t in range(T - 1, 0, -1):
        ruta[t - 1] = back[t, ruta[t]]
    return ruta


# ----------------------------------------------------------------------------
# 3. Entrenamiento y evaluación
# ----------------------------------------------------------------------------
def _lgbm(n_estimators):
    # random_state + deterministic + force_col_wise: el mismo modelo en cada
    # ejecución y con distinto número de hilos (la entrega debe regenerarse exacta).
    return lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.05, num_leaves=63,
        min_child_samples=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, verbose=-1,
        random_state=0, deterministic=True, force_col_wise=True,
    )


# Features que NO dependen del balón: colectivas (con sus ventanas), velocidad
# media de los jugadores, cuánto lleva el balón sin verse, Smart Tagging, periodo.
_SIN_BALON = re.compile(
    r"^((l|v)_(cx|cy|vcx|vcy|sx|sy|xmin|xmax|ymin|ymax|v|vmax|wx|wy|r3x|r3y|r3vx|r3vy)"
    r"|dcx|dcy|mx|my|wmx|wmy|hx|hy)(_[pf]\d+)?$"
    r"|^(v_jugadores(_50)?|frames_sin_balon_(atras|adelante)|balon_nan|periodo|st_.*)$")


def columnas_sin_balon(columnas) -> list:
    return [c for c in columnas if _SIN_BALON.match(c)]


class ModeloHibrido:
    """
    Dos especialistas mezclados fotograma a fotograma: `con` (todas las
    features) entrenado y aplicado solo donde el balón se ve, y `sin` (solo
    features que no dependen del balón) aplicado donde no se ve. La columna
    `balon_nan` de las features dice cuál toca. Así el modelo sin balón no
    aprende a fiarse de un balón interpolado que en los huecos largos es falso.
    """

    def __init__(self, n_estimators=400, hueco_min=0):
        self.n_estimators = n_estimators
        self.hueco_min = hueco_min       # el especialista sin balón solo en huecos > hueco_min fotogramas

    def _usa_con(self, X):
        """Dónde manda el modelo con balón: balón visto o hueco corto (interpolación fiable)."""
        hueco = X["frames_sin_balon_atras"].to_numpy() + X["frames_sin_balon_adelante"].to_numpy()
        return (X["balon_nan"].to_numpy() == 0) | (hueco <= self.hueco_min)

    def fit(self, X, y):
        con = self._usa_con(X)
        self.columnas_sin = columnas_sin_balon(X.columns)
        self.con = _lgbm(self.n_estimators).fit(X[con], y[con])
        self.sin = _lgbm(self.n_estimators).fit(X[self.columnas_sin], y)
        self.classes_ = self.con.classes_
        return self

    def predict_proba(self, X):
        con = self._usa_con(X)
        p = self.sin.predict_proba(X[self.columnas_sin])
        if con.any():
            p[con] = self.con.predict_proba(X[con])
        return p

    @property
    def feature_importances_(self):
        return self.con.feature_importances_


def entrenar(X, y, n_estimators=400, hibrido=False, hueco_min=0):
    m = y >= 0
    modelo = ModeloHibrido(n_estimators, hueco_min) if hibrido else _lgbm(n_estimators)
    modelo.fit(X[m], y[m])
    assert list(modelo.classes_) == [NINGUNO, LOCAL, VISITA]
    return modelo


def concatenar(datos, juegos):
    F = pd.concat([datos[j]["F"] for j in juegos], ignore_index=True)
    y = np.concatenate([datos[j]["gt"] for j in juegos])
    return F, y


def elegir_suavizado(datos, entreno, n_estimators, hibrido=False, hueco_min=0, grid_pd=MIN_PD_GRID):
    """
    Elige (p_quedarse del Viterbi, duración mínima de una posesión, duración
    mínima de una parada entre posesiones) maximizando
    spellF1 con predicciones FUERA de muestra: para cada partido de
    entrenamiento, predice con un modelo entrenado en los demás. Con un solo
    partido de entrenamiento no queda más remedio que usar el propio.
    """
    probs, ys, pers = [], [], []
    for j in entreno:
        otros = [o for o in entreno if o != j] or [j]
        F_tr, y_tr = concatenar(datos, otros)
        m = entrenar(F_tr, y_tr, n_estimators, hibrido, hueco_min)
        probs.append(m.predict_proba(datos[j]["F"]))
        ys.append(datos[j]["gt"])
        pers.append(datos[j]["F"]["periodo"].to_numpy() + 100 * len(pers))  # periodos distintos entre partidos
    probs, ys, pers = np.vstack(probs), np.concatenate(ys), np.concatenate(pers)

    mejor, mejor_sf1 = None, -1
    for q in Q_GRID:
        ruta = viterbi(probs, q, pers)
        for d in MIN_D_GRID:
            ruta_d = quitar_paradas_cortas(ruta, d, pers)
            for k in MIN_FRAMES_GRID:
                ruta_k = fusionar_rachas_cortas(ruta_d, k, pers)
                for pd_ in grid_pd:
                    sf1 = spell_f1(quitar_posesiones_entre_paradas(ruta_k, pd_, pers), ys, pers)
                    if sf1 > mejor_sf1:
                        mejor, mejor_sf1 = (q, k, d, pd_), sf1
    return mejor


def suavizar(probs, q, min_frames, periodos, visto=None, hueco=None, min_d=1, min_pd=1):
    """
    Viterbi + paradas cortas fuera + fusión de posesiones cortas + posesiones
    cortas entre dos paradas fuera: la salida final del modelo.
    `hueco=(modo, min_hueco)` con `visto` (balón en los datos, por fotograma)
    mantiene la etiqueta anterior en los huecos largos del balón antes de fundir.
    """
    ruta = viterbi(probs, q, periodos)
    if hueco is not None and visto is not None:
        ruta = mantener_en_huecos(ruta, visto, periodos, hueco[1], hueco[0])
    ruta = quitar_paradas_cortas(ruta, min_d, periodos)
    ruta = fusionar_rachas_cortas(ruta, min_frames, periodos)
    return quitar_posesiones_entre_paradas(ruta, min_pd, periodos)


def simular(df, gt, enmascarar, paradas_visibles, semilla, args, ruido=None) -> dict:
    """
    Aplica al tracking las condiciones simuladas del ATD oficial que tocan el
    balón (oculto en juego, visible en paradas, ruido de detección), lo rellena
    según --balon y calcula las features. Devuelve df, df_real, visto, F0, base.
    """
    df_real = df
    if enmascarar > 0:                               # simular el balón ausente en juego
        df, mascara = enmascarar_balon(df, gt, enmascarar, semilla=semilla)
        print(f"  balón oculto en {mascara.mean():.1%} de los fotogramas "
              f"({(df['ball_x'].isna() & np.isin(gt, (LOCAL, VISITA))).sum() / np.isin(gt, (LOCAL, VISITA)).sum():.1%} de los de juego)")
    if paradas_visibles > 0:                         # balón visible y quieto antes de algunos saques
        df, mascara = mostrar_balon_en_paradas(df, gt, paradas_visibles, semilla=semilla)
        print(f"  balón visible en {df['ball_x'].notna()[gt == NINGUNO].mean():.1%} de las paradas "
              f"({mascara.sum()} fotogramas revelados)")
    if ruido:                                        # ruido de detección del balón (calibrado con el ATD oficial)
        df = ruido_balon(df, *ruido, semilla=semilla)
    visto = df["ball_x"].notna().to_numpy()          # antes de rellenar: qué huecos había
    if args.balon == "interpolar":
        df = estimar_balon(df)
    out = dict(df=df, df_real=df_real, visto=visto)
    if args.balon != "estimar":                      # con "estimar" las features dependen del fold
        out["F0"], out["base"] = features_base(df, args.colectivas, args.features_parada)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("salida"))
    ap.add_argument("--juegos", nargs="*", help="por defecto, todas las subcarpetas reconocidas")
    ap.add_argument("--arboles", type=int, default=400, help="n_estimators de LightGBM")
    ap.add_argument("--enmascarar", type=float, default=0.0,
                    help="fracción de fotogramas en juego en los que se oculta el balón (oficial: 0.41)")
    ap.add_argument("--verdad", choices=["oficial", "control"], default="oficial",
                    help="definición de la verdad aproximada: 'control' exige que el rival controle "
                         "(pase, tiro o --max-toque fotogramas) para cambiar la posesión")
    ap.add_argument("--max-toque", type=int, default=25, help="con --verdad control (fotogramas)")
    ap.add_argument("--prueba-verdad", choices=["oficial", "control"], default=None,
                    help="verdad del partido de prueba (por defecto, la misma que --verdad): "
                         "para comparar modelos entrenados con verdades distintas en el mismo examen")
    ap.add_argument("--paradas-visibles", type=float, default=0.0,
                    help="fracción de fotogramas D con el balón visible y quieto en el punto del saque "
                         "(los sample data casi no tienen; el ATD oficial ~15%%)")
    ap.add_argument("--prueba-enmascarar", type=float, default=None,
                    help="--enmascarar del partido de prueba (por defecto, el mismo que al entrenar): "
                         "para comparar simulaciones de entrenamiento contra el mismo examen")
    ap.add_argument("--prueba-paradas-visibles", type=float, default=None,
                    help="--paradas-visibles del partido de prueba (por defecto, el mismo que al entrenar)")
    ap.add_argument("--ruido-balon", type=float, nargs=3, default=None, metavar=("SIGMA", "TAU", "BLANCO"),
                    help="ruido de detección del balón al entrenar (balon_estimado.ruido_balon); "
                         "calibrado con el ATD oficial: 0.8 30 0.06")
    ap.add_argument("--prueba-ruido-balon", type=float, nargs=3, default=None, metavar=("SIGMA", "TAU", "BLANCO"),
                    help="ruido de balón del partido de prueba (por defecto, el mismo que al entrenar; "
                         "0 0 0 = sin ruido): para comparar modelos contra el mismo examen")
    ap.add_argument("--sin-colectivas", dest="colectivas", action="store_false",
                    help="quita las features colectivas (centroides, líneas, movimiento); por defecto van")
    ap.add_argument("--balon", choices=["ninguno", "interpolar", "estimar"], default="ninguno",
                    help="cómo rellenar el balón ausente: nada, interpolación lineal o estimador colectivo")
    ap.add_argument("--huecos", action="store_true",
                    help="prueba mantener la etiqueta anterior en los huecos largos del balón (rejilla modo × duración)")
    ap.add_argument("--features-parada", action="store_true",
                    help="features de balón parado y visible (par_*: quieto, en un punto de reanudación, jugadores lentos)")
    ap.add_argument("--sin-identidades", action="store_true",
                    help="simula el ATD oficial: slots sin identidad de jugador, con cortes, huecos, duplicados e intercambios")
    ap.add_argument("--hibrido", action="store_true",
                    help="dos especialistas: con balón donde se ve y sin features del balón donde no (ModeloHibrido)")
    ap.add_argument("--hibrido-hueco", type=int, default=0,
                    help="con --hibrido: el especialista sin balón solo en huecos de más de N fotogramas (0 = en todos)")
    ap.add_argument("--smart-tagging", choices=["no", "ancla", "features", "ambos"], default="no",
                    help="simula las fases SET PIECES del Smart Tagging y las usa como ancla de saques (post-procesado), "
                         "como features del modelo, o ambas")
    args = ap.parse_args()
    if args.prueba_enmascarar is None:
        args.prueba_enmascarar = args.enmascarar
    if args.prueba_paradas_visibles is None:
        args.prueba_paradas_visibles = args.paradas_visibles
    if args.ruido_balon is not None and not any(args.ruido_balon):
        args.ruido_balon = None                      # 0 0 0 = sin ruido
    if args.prueba_ruido_balon is None:
        args.prueba_ruido_balon = args.ruido_balon
    elif not any(args.prueba_ruido_balon):
        args.prueba_ruido_balon = None
    examen_aparte = ((args.prueba_enmascarar, args.prueba_paradas_visibles, args.prueba_ruido_balon)
                     != (args.enmascarar, args.paradas_visibles, args.ruido_balon))
    if examen_aparte and args.balon == "estimar":
        ap.error("--prueba-* no está implementado con --balon estimar")
    args.out.mkdir(parents=True, exist_ok=True)

    juegos = args.juegos or listar_juegos(args.data)
    datos = {}
    for k, j in enumerate(juegos):
        print(f"Preparando {j}...")
        df, gt = cargar_partido(args.data, j, args.verdad, args.max_toque)
        gt_examen = gt
        if args.prueba_verdad not in (None, args.verdad):
            gt_examen = D.verdad_desde_eventos(D.cargar_eventos(args.data / j, j), df["Frame"],
                                               args.prueba_verdad, args.max_toque)
        datos_st = dict(fases=None, st=None)
        if args.smart_tagging != "no":               # simular las fases SET PIECES del Smart Tagging
            ev = D.cargar_eventos(args.data / j, j)
            fases = simular_set_pieces(ev, gt, semilla=k, primer_frame=int(df["Frame"].iloc[0]))
            datos_st["fases"] = fases
            if args.smart_tagging in ("features", "ambos"):
                datos_st["st"] = features_smart_tagging(fases, len(df)).set_index(df.index)
            print(f"  smart tagging: {len(fases)} fases SET PIECES simuladas "
                  f"({(fases['equipo'] < 0).sum()} sin equipo, {int(fases['falso'].sum())} falsas)")
        if args.sin_identidades:                     # simular los track slots del ATD oficial
            df, info = quitar_identidades(df, semilla=k)
            print(f"  sin identidades: {info['tracklets']} tracklets, {info['intercambios']} intercambios, "
                  f"{info['visibles_medio']:.1f} jugadores visibles de media")
        datos[j] = dict(gt=gt, gt_examen=gt_examen,
                        **simular(df, gt, args.enmascarar, args.paradas_visibles, k, args, args.ruido_balon), **datos_st)
        if examen_aparte:
            print("  examen:")
            datos[j]["prueba"] = simular(df, gt, args.prueba_enmascarar, args.prueba_paradas_visibles, k, args,
                                         args.prueba_ruido_balon)

    print(f"\n== Validación leave-one-out entre partidos (métricas oficiales; "
          f"entreno: balón oculto {args.enmascarar:.0%}, paradas visibles {args.paradas_visibles:.0%}, "
          f"ruido de balón {args.ruido_balon or 'no'}; "
          f"examen: {args.prueba_enmascarar:.0%} / {args.prueba_paradas_visibles:.0%} / ruido {args.prueba_ruido_balon or 'no'}; "
          f"relleno: {args.balon}) ==")
    resumen = defaultdict(list)
    for prueba in juegos:
        entreno = [j for j in juegos if j != prueba]

        # 0. Estimador de balón entrenado solo con los partidos de entrenamiento.
        #    Cada partido de entrenamiento recibe estimaciones de un estimador que
        #    NO lo vio (cross-fitting): si no, el modelo de posesión aprende con
        #    estimaciones demasiado buenas y en la prueba se fía de más.
        if args.balon == "estimar":
            print("  entrenando los estimadores de balón...")
            frac = max(args.enmascarar, 0.3)

            def estimador_con(juegos_est):
                F_b, x_b, y_b = preparar_entrenamiento(
                    {j: (datos[j]["df_real"], datos[j]["gt"]) for j in juegos_est}, frac)
                return entrenar_estimador(F_b, x_b, y_b)

            est = estimador_con(entreno)
            datos[prueba]["F0"], datos[prueba]["base"] = features_base(estimar_balon(datos[prueba]["df"], est), args.colectivas,
                                                                       args.features_parada)
            for j in entreno:
                otros = [o for o in entreno if o != j] or [j]
                est_j = estimador_con(otros) if otros != [j] else est
                datos[j]["F0"], datos[j]["base"] = features_base(estimar_balon(datos[j]["df"], est_j), args.colectivas,
                                                                 args.features_parada)

        # 1. Parámetros de la regla, ajustados (por spellF1) solo en los partidos de entrenamiento
        params = buscar_parametros([(datos[j]["base"], datos[j]["gt"]) for j in entreno])[0][1]
        for j in juegos:
            datos[j]["F"] = agregar_regla(datos[j]["F0"], datos[j]["base"], params)
            if datos[j]["st"] is not None:           # features del Smart Tagging simulado
                datos[j]["F"] = pd.concat([datos[j]["F"], datos[j]["st"]], axis=1)

        # 2. Suavizado (p_quedarse, duración mínima), con predicciones fuera de muestra
        q, min_frames, min_d, min_pd = elegir_suavizado(datos, entreno, args.arboles, args.hibrido, args.hibrido_hueco)

        # 3. Modelo final: entrena en todos los de entrenamiento, prueba en el restante
        F_tr, y_tr = concatenar(datos, entreno)
        modelo = entrenar(F_tr, y_tr, args.arboles, args.hibrido, args.hibrido_hueco)
        examen = datos[prueba].get("prueba", datos[prueba])   # otras condiciones simuladas para el examen
        F_te, y_te, df_te = datos[prueba]["F"], datos[prueba]["gt_examen"], examen["df"]
        if examen is not datos[prueba]:
            F_te = agregar_regla(examen["F0"], examen["base"], params)
            if datos[prueba]["st"] is not None:
                F_te = pd.concat([F_te, datos[prueba]["st"]], axis=1)
        per = F_te["periodo"].to_numpy()
        probs = modelo.predict_proba(F_te)
        salidas = dict(regla=F_te["regla"].to_numpy(), lgbm=probs.argmax(1),
                       viterbi=viterbi(probs, q, per),
                       final_sin_pd=suavizar(probs, q, min_frames, per, min_d=min_d),
                       final=suavizar(probs, q, min_frames, per, min_d=min_d, min_pd=min_pd))
        if args.smart_tagging in ("ancla", "ambos"):  # el equipo del saque, desde el Smart Tagging
            salidas["final_ancla"] = anclar_saques(salidas["final"], datos[prueba]["fases"], per)
        if args.huecos:                              # mantener la etiqueta en huecos largos del balón
            visto = examen["visto"]
            for modo in ("equipo", "todo", "d"):
                for L in HUECO_GRID:
                    salidas[f"hueco_{modo}_{L}"] = suavizar(probs, q, min_frames, per, visto, (modo, L), min_d, min_pd)

        print(f"\n{prueba}  (regla={params}, p_quedarse={q}, min_frames={min_frames}, min_d={min_d}, min_pd={min_pd})")
        print(CABECERA)
        for nombre, p in salidas.items():
            r = evaluar(p, y_te, per)
            resumen[nombre].append(r)
            print(fila(nombre, r))
        print(D.matriz_confusion(salidas["final"], y_te).round(3).to_string())

        D.exportar(args.out / f"{prueba}_posesion_modelo.csv",
                   df_te["Frame"], df_te["Period"], salidas["final"], probs=probs, verdad=y_te)

    print("\n== Promedio de los tres partidos ==")
    print(CABECERA)
    for nombre, rs in resumen.items():
        prom = {k: float(np.mean([r[k] for r in rs])) for k in ("spell_f1", "macro_f1", "exactitud", "n_pos_pred", "n_pos_gt", "dur_pred", "dur_gt")}
        prom["n_pos_pred"], prom["n_pos_gt"] = round(prom["n_pos_pred"]), round(prom["n_pos_gt"])
        print(fila(nombre, prom))

    imp = pd.Series(modelo.feature_importances_, index=F_tr.columns).sort_values(ascending=False)
    print("\nTop 10 features (último modelo):")
    print(imp.head(10).to_string())


if __name__ == "__main__":
    main()
