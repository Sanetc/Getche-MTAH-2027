"""
More Than a Hack 2027 — ¿dónde está el balón cuando el tracking no lo ve?
=========================================================================

En los datos oficiales el balón falta en el 41% de los fotogramas evaluados,
y en juego (áreas llenas, duelos, entradas). En los sample data casi solo
falta en paradas, así que para prepararnos **simulamos** esa ausencia:
`enmascarar_balon()` oculta el balón en ventanas aleatorias dentro del juego
hasta llegar a la fracción pedida.

Con el balón oculto, lo estimamos desde lo que sí se ve:
  - la colocación y el movimiento de los dos equipos (centroides, líneas,
    dispersión, centroide ponderado por velocidad, jugadores más rápidos),
    también en ventanas hacia atrás y hacia adelante;
  - la última y la siguiente posición conocidas del balón, su velocidad y
    la extrapolación / interpolación entre ambas.

Dos LightGBM de regresión (x, y) aprenden a corregir la interpolación con el
movimiento colectivo. Se valida leave-one-out entre partidos, midiendo el
error en metros por duración del hueco, contra "último visto" e
"interpolación lineal".

Uso:
    python -I balon_estimado.py --data ../datos/sample-data/data [--fraccion 0.4]
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos as D  # noqa: E402
from datos import FPS, LARGO, ANCHO, NINGUNO, LOCAL, VISITA  # noqa: E402
from baseline_posesion import cargar_partido, listar_juegos  # noqa: E402


# ----------------------------------------------------------------------------
# 1. Simular la ausencia del balón
# ----------------------------------------------------------------------------
def enmascarar_balon(df: pd.DataFrame, gt: np.ndarray, fraccion=0.4, semilla=0,
                     mediana_frames=25, sigma=1.0):
    """
    Oculta el balón (NaN) en ventanas aleatorias de duración log-normal
    (mediana `mediana_frames`, 3–500 fotogramas) sobre fotogramas en juego con
    balón visible, hasta ocultar `fraccion` de ellos. Devuelve (df, mascara).
    Con fraccion <= 0 devuelve el df con la columna `balon_visto` y máscara vacía.
    """
    visible = df["ball_x"].notna().to_numpy()
    candidatos = visible & np.isin(gt, (LOCAL, VISITA))
    mascara = np.zeros(len(df), dtype=bool)
    if fraccion > 0:
        rng = np.random.default_rng(semilla)
        objetivo = fraccion * candidatos.sum()
        n = len(df)
        while (mascara & candidatos).sum() < objetivo:
            L = int(np.clip(rng.lognormal(np.log(mediana_frames), sigma), 3, 500))
            i = int(rng.integers(0, n - L))
            mascara[i:i + L] = True
        mascara &= candidatos
    out = df.copy()
    out.loc[mascara, ["ball_x", "ball_y"]] = np.nan
    out["balon_visto"] = out["ball_x"].notna()
    return out, mascara


def mostrar_balon_en_paradas(df: pd.DataFrame, gt: np.ndarray, fraccion=0.15, semilla=0,
                             segundos=(2, 15), ruido=0.1):
    """
    Lo contrario de `enmascarar_balon`, para las paradas: en los sample data el
    balón casi nunca se ve con el juego parado (falta en el 92–99% de los D),
    pero en el ATD oficial ~13% de los fotogramas con balón visible serían
    paradas (estimado el 22 oct por la velocidad de los jugadores). Aquí, en
    paradas al azar, el balón aparece quieto en el punto del saque (la primera
    posición vista después de la parada, con `ruido` m de temblor de tracking)
    durante sus últimos `segundos` (uniforme), hasta que se vea en `fraccion`
    de los fotogramas D. Devuelve (df, mascara de fotogramas revelados).
    """
    bx, by = df["ball_x"].to_numpy(dtype=float, copy=True), df["ball_y"].to_numpy(dtype=float, copy=True)
    d = gt == NINGUNO
    r = np.flatnonzero(np.diff(np.r_[0, d.astype(int), 0]))
    rachas = list(zip(r[::2], r[1::2]))
    visto = ~np.isnan(bx)
    mascara = np.zeros(len(df), dtype=bool)
    rng = np.random.default_rng(semilla)
    objetivo = fraccion * d.sum() - (d & visto).sum()
    for k in rng.permutation(len(rachas)):
        if mascara.sum() >= objetivo:
            break
        a, b = rachas[k]
        despues = np.flatnonzero(visto[b:b + 5 * FPS])         # el saque: primer balón visto tras la parada
        if not len(despues):
            continue
        sx, sy = bx[b + despues[0]], by[b + despues[0]]
        L = int(rng.uniform(*segundos) * FPS)
        ini = max(a, b - L)
        tramo = np.arange(ini, b)[~visto[ini:b]]
        mascara[tramo] = True
        bx[tramo] = sx + rng.normal(0, ruido, len(tramo))
        by[tramo] = sy + rng.normal(0, ruido, len(tramo))
    out = df.copy()
    out["ball_x"], out["ball_y"] = bx, by
    out["balon_visto"] = out["ball_x"].notna()
    return out, mascara


def _desvio_lento(n: int, sigma: float, tau: float, rng, suave=5.0) -> np.ndarray:
    """Proceso lento de media 0 y desviación `sigma`: AR(1) con tiempo de correlación `tau` fotogramas, alisado y reescalado."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import lfilter

    rho = float(np.exp(-1.0 / max(tau, 1e-6)))
    e = rng.normal(0.0, np.sqrt(1 - rho ** 2), n)
    e[0] = rng.normal(0.0, 1.0)
    x = lfilter([1.0], [1.0, -rho], e)
    if suave > 0:
        x = gaussian_filter1d(x, suave, mode="nearest")
    return x * (sigma / x.std())


def ruido_balon(df: pd.DataFrame, sigma=0.8, tau=30.0, blanco=0.06, semilla=0, suave=5.0) -> pd.DataFrame:
    """
    Ruido de detección del balón, para acercar los sample data al ATD oficial.
    En los sample data (tracking corregido) el balón va pegado al jugador que
    lo lleva: con el balón visto y en juego, el jugador más cercano está a
    0,3–1,5 m de mediana (p25 0,1–0,2 m). En el ATD oficial (detección
    automática) está a 1,85 m de mediana (p25 0,88 m) y la velocidad del balón
    tiembla (aceleración mediana 4,9 m/s² frente a ~0,1). Aquí se suma al
    balón visto un desvío lento (desviación `sigma` m por eje, tiempo de
    correlación `tau` fotogramas, alisado) y un temblor blanco de `blanco` m
    por eje. Los tres parámetros se calibran solo con agregados sin etiquetas
    del partido oficial (`calibrar_ruido.py`): cuantiles de la distancia
    jugador–balón, de su mínimo en 2 s y de la aceleración y la velocidad del
    balón. Los NaN (balón no visto) siguen siendo NaN.
    """
    if sigma <= 0 and blanco <= 0:
        return df
    rng = np.random.default_rng(1000 + semilla)
    out = df.copy()
    n = len(out)
    for c in ("ball_x", "ball_y"):
        r = _desvio_lento(n, sigma, tau, rng, suave) if sigma > 0 else np.zeros(n)
        if blanco > 0:
            r = r + rng.normal(0.0, blanco, n)
        out[c] = out[c].to_numpy(dtype=float) + r
    return out


# ----------------------------------------------------------------------------
# 2. Features colectivas (no dependen del balón)
# ----------------------------------------------------------------------------
def _xy(df, prefijo):
    xs = D.columnas_jugadores(df, prefijo)
    return df[xs].to_numpy(), df[[c[:-2] + "_y" for c in xs]].to_numpy()


def _suave(a, w=5):
    return pd.DataFrame(a).rolling(w, center=True, min_periods=1).mean().to_numpy()


def velocidades(X, Y, salto_max=1.0, w=5):
    """
    Velocidad (VX, VY en m/s) de cada columna de jugadores, suavizada con una
    ventana de `w` fotogramas. Un salto de más de `salto_max` m entre dos
    fotogramas no es un movimiento: es un slot que cambia de jugador (tracking
    sin identidades) o un error, así que se corta la pista ahí (NaN en los
    fotogramas que la ventana mezclaría) y la velocidad no se dispara.
    """
    Xv, Yv = X.astype(float).copy(), Y.astype(float).copy()
    salto = np.hypot(np.diff(Xv, axis=0), np.diff(Yv, axis=0)) > salto_max   # (T-1, n): salto entre t-1 y t
    t, c = np.nonzero(salto)
    h = w // 2
    for k in range(-h, h):                                # fotogramas t-h .. t+h-1 alrededor del salto
        tk = np.clip(t + 1 + k, 0, len(Xv) - 1)
        Xv[tk, c] = np.nan
        Yv[tk, c] = np.nan
    return np.gradient(_suave(Xv, w), axis=0) * FPS, np.gradient(_suave(Yv, w), axis=0) * FPS


def _ventanas(F: pd.DataFrame, cols, periodos, ws=(10, 25)):
    """Media de cada columna en ventanas hacia atrás y hacia adelante, por periodo."""
    g = periodos.to_numpy()
    for w in ws:
        for c in cols:
            s = F[c]
            F[f"{c}_p{w}"] = s.groupby(g).transform(lambda x: x.rolling(w, min_periods=1).mean())
            F[f"{c}_f{w}"] = s.groupby(g).transform(
                lambda x: x[::-1].rolling(w, min_periods=1).mean()[::-1])
    return F


def features_colectivas(df: pd.DataFrame) -> pd.DataFrame:
    """Colocación y movimiento de cada equipo, fotograma a fotograma."""
    F = pd.DataFrame(index=df.index)
    por_equipo = {}
    with np.errstate(all="ignore"):
        for nombre, prefijo in (("l", "H_"), ("v", "A_")):
            X, Y = _xy(df, prefijo)
            VX, VY = velocidades(X, Y)
            S = np.hypot(VX, VY)
            cx, cy = np.nanmean(X, 1), np.nanmean(Y, 1)
            F[f"{nombre}_cx"], F[f"{nombre}_cy"] = cx, cy
            F[f"{nombre}_vcx"], F[f"{nombre}_vcy"] = np.nanmean(VX, 1), np.nanmean(VY, 1)
            F[f"{nombre}_sx"], F[f"{nombre}_sy"] = np.nanstd(X, 1), np.nanstd(Y, 1)
            F[f"{nombre}_xmin"], F[f"{nombre}_xmax"] = np.nanmin(X, 1), np.nanmax(X, 1)
            F[f"{nombre}_ymin"], F[f"{nombre}_ymax"] = np.nanmin(Y, 1), np.nanmax(Y, 1)
            F[f"{nombre}_n"] = np.isfinite(X).sum(1)
            F[f"{nombre}_v"] = np.nanmean(S, 1)
            F[f"{nombre}_vmax"] = np.nanmax(S, 1)
            # centroide ponderado por velocidad: hacia donde corre el equipo
            w = np.where(np.isfinite(S), S, 0) + 1e-3
            F[f"{nombre}_wx"] = np.nansum(X * w, 1) / w.sum(1)
            F[f"{nombre}_wy"] = np.nansum(Y * w, 1) / w.sum(1)
            # los 3 jugadores más rápidos: dónde están y hacia dónde van
            orden = np.argsort(-np.where(np.isfinite(S), S, -1), axis=1)[:, :3]
            fila = np.arange(len(df))[:, None]
            F[f"{nombre}_r3x"] = np.nanmean(X[fila, orden], 1)
            F[f"{nombre}_r3y"] = np.nanmean(Y[fila, orden], 1)
            F[f"{nombre}_r3vx"] = np.nanmean(VX[fila, orden], 1)
            F[f"{nombre}_r3vy"] = np.nanmean(VY[fila, orden], 1)
            por_equipo[nombre] = (X, Y)

    # Relación entre equipos
    F["dcx"], F["dcy"] = F["l_cx"] - F["v_cx"], F["l_cy"] - F["v_cy"]
    F["mx"], F["my"] = (F["l_cx"] + F["v_cx"]) / 2, (F["l_cy"] + F["v_cy"]) / 2
    F["wmx"], F["wmy"] = (F["l_wx"] + F["v_wx"]) / 2, (F["l_wy"] + F["v_wy"]) / 2
    # "punto caliente": media de los 4 jugadores (de los 22) más cercanos al centroide ponderado
    X = np.hstack([por_equipo["l"][0], por_equipo["v"][0]])
    Y = np.hstack([por_equipo["l"][1], por_equipo["v"][1]])
    d = np.hypot(X - F["wmx"].to_numpy()[:, None], Y - F["wmy"].to_numpy()[:, None])
    orden = np.argsort(np.where(np.isfinite(d), d, np.inf), axis=1)[:, :4]
    fila = np.arange(len(df))[:, None]
    with np.errstate(all="ignore"):
        F["hx"], F["hy"] = np.nanmean(X[fila, orden], 1), np.nanmean(Y[fila, orden], 1)

    F = _ventanas(F, ["l_cx", "l_cy", "v_cx", "v_cy", "wmx", "wmy", "hx", "hy", "l_vcx", "v_vcx"],
                  df["Period"])
    return F


# ----------------------------------------------------------------------------
# 3. Features de la historia del balón (última / siguiente vez visto)
# ----------------------------------------------------------------------------
def features_historia_balon(df: pd.DataFrame) -> pd.DataFrame:
    """Última y siguiente posición conocidas, velocidad, extrapolación e interpolación."""
    F = pd.DataFrame(index=df.index)
    bx, by = df["ball_x"], df["ball_y"]
    visto = bx.notna()
    idx = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    g = df["Period"].to_numpy()

    # velocidad del balón cuando se ve (media de 5 fotogramas)
    vx = pd.Series(np.gradient(bx.interpolate(limit=3).to_numpy()) * FPS, index=df.index).rolling(5, min_periods=1).mean()
    vy = pd.Series(np.gradient(by.interpolate(limit=3).to_numpy()) * FPS, index=df.index).rolling(5, min_periods=1).mean()

    def ffill(s):
        return s.where(visto).groupby(g).ffill()

    def bfill(s):
        return s.where(visto).groupby(g).bfill()

    F["ux"], F["uy"] = ffill(bx), ffill(by)                       # último visto
    F["uvx"], F["uvy"] = ffill(vx), ffill(vy)
    F["dt_u"] = (idx - ffill(idx)).fillna(1e4).clip(0, 5000)      # fotogramas desde entonces
    F["sx"], F["sy"] = bfill(bx), bfill(by)                       # siguiente visto
    F["svx"], F["svy"] = bfill(vx), bfill(vy)
    F["dt_s"] = (bfill(idx) - idx).fillna(1e4).clip(0, 5000)

    # extrapolación desde el último visto, frenando el balón (rozamiento 0,5 m/s²)
    t = F["dt_u"] / FPS
    F["ex"] = (F["ux"] + F["uvx"] * t).clip(-LARGO / 2, LARGO / 2)
    F["ey"] = (F["uy"] + F["uvy"] * t).clip(-ANCHO / 2, ANCHO / 2)
    # interpolación lineal entre último y siguiente
    peso = F["dt_u"] / (F["dt_u"] + F["dt_s"]).replace(0, 1)
    F["ix"] = F["ux"] + (F["sx"] - F["ux"]) * peso
    F["iy"] = F["uy"] + (F["sy"] - F["uy"]) * peso
    F["hueco"] = (F["dt_u"] + F["dt_s"]).clip(0, 10000)
    return F


def features_estimador(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([features_colectivas(df), features_historia_balon(df)], axis=1)


# ----------------------------------------------------------------------------
# 4. Entrenar y aplicar
# ----------------------------------------------------------------------------
def entrenar_estimador(F: pd.DataFrame, bx, by, n_estimators=300):
    """
    Dos regresores que aprenden la CORRECCIÓN sobre la interpolación lineal
    (real − interpolado). En huecos cortos la interpolación ya es casi exacta y
    el modelo aprende a no tocarla; en los largos aprende del movimiento colectivo.
    """
    modelos = {}
    for nombre, y, base in (("x", bx, F["ix"]), ("y", by, F["iy"])):
        m = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=0.05, num_leaves=63,
                              min_child_samples=100, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1,
                              random_state=0, deterministic=True, force_col_wise=True)
        ok = np.isfinite(base.to_numpy())
        m.fit(F[ok], (y - base.to_numpy())[ok])
        modelos[nombre] = m
    return modelos


def predecir_balon(modelos, F: pd.DataFrame):
    """Posición estimada (x, y) = interpolación + corrección aprendida."""
    ex = F["ix"].to_numpy() + modelos["x"].predict(F)
    ey = F["iy"].to_numpy() + modelos["y"].predict(F)
    return np.clip(ex, -LARGO / 2, LARGO / 2), np.clip(ey, -ANCHO / 2, ANCHO / 2)


def estimar_balon(df: pd.DataFrame, modelos=None, suavizar=15) -> pd.DataFrame:
    """
    Rellena el balón donde falta: con los modelos si se dan, si no por
    interpolación lineal. Deja `balon_visto` para que el modelo de posesión
    sepa qué fotogramas son estimados. `suavizar`: ventana (fotogramas) con la
    que se alisa la corrección del modelo dentro de los huecos.
    """
    out = df.copy()
    if "balon_visto" not in out:
        out["balon_visto"] = out["ball_x"].notna()
    falta = ~out["balon_visto"].to_numpy()
    if not falta.any():
        return out
    F = features_estimador(out)
    ix, iy = F["ix"].to_numpy(), F["iy"].to_numpy()
    if modelos is None:
        ex, ey = ix[falta], iy[falta]
    else:
        ex, ey = predecir_balon(modelos, F[falta])
        if suavizar > 1:
            # La corrección aprendida salta de un fotograma a otro; suavizarla
            # dentro de cada hueco evita velocidades y distancias ruidosas.
            # En huecos cortos (<= 1 s) la interpolación ya es casi exacta.
            pos = np.flatnonzero(falta)
            hueco_id = np.cumsum(np.r_[1, np.diff(pos) != 1])    # cada hueco, su propia ventana
            alisar = lambda c: pd.Series(c).groupby(hueco_id).transform(
                lambda s: s.rolling(suavizar, center=True, min_periods=1).mean()).to_numpy()
            cx, cy = alisar(ex - ix[falta]), alisar(ey - iy[falta])
            corto = F["hueco"].to_numpy()[falta] <= FPS
            ex = np.where(corto, ix[falta], ix[falta] + cx)
            ey = np.where(corto, iy[falta], iy[falta] + cy)
    # donde ni siquiera hay interpolación (antes de verlo por primera vez en la parte) queda NaN
    out.loc[falta, "ball_x"] = np.where(np.isfinite(ex), ex, np.nan)
    out.loc[falta, "ball_y"] = np.where(np.isfinite(ey), ey, np.nan)
    return out


def preparar_entrenamiento(partidos, fraccion, semilla=0):
    """
    partidos: dict juego -> (df, gt). Enmascara cada uno y devuelve (F, bx, by)
    apilados solo con los fotogramas ocultos (el caso real de uso).
    """
    Fs, xs, ys = [], [], []
    for k, (j, (df, gt)) in enumerate(partidos.items()):
        dfm, mascara = enmascarar_balon(df, gt, fraccion, semilla + k)
        F = features_estimador(dfm)
        Fs.append(F[mascara])
        xs.append(df["ball_x"].to_numpy()[mascara])
        ys.append(df["ball_y"].to_numpy()[mascara])
    return pd.concat(Fs), np.concatenate(xs), np.concatenate(ys)


# ----------------------------------------------------------------------------
# 5. Validación
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--fraccion", type=float, default=0.4, help="fracción de fotogramas en juego sin balón")
    ap.add_argument("--juegos", nargs="*")
    args = ap.parse_args()

    juegos = args.juegos or listar_juegos(args.data)
    partidos = {}
    for j in juegos:
        print(f"Cargando {j}...")
        partidos[j] = cargar_partido(args.data, j)

    bordes = [0, 25, 75, 250, 10 ** 6]
    nombres = ["<=1 s", "1-3 s", "3-10 s", ">10 s"]
    print(f"\n== Error (m) al estimar el balón oculto, leave-one-out, {args.fraccion:.0%} oculto en juego ==")
    filas = []
    for prueba in juegos:
        entreno = {j: partidos[j] for j in juegos if j != prueba}
        F_tr, x_tr, y_tr = preparar_entrenamiento(entreno, args.fraccion)
        modelos = entrenar_estimador(F_tr, x_tr, y_tr)

        df, gt = partidos[prueba]
        dfm, mascara = enmascarar_balon(df, gt, args.fraccion, semilla=99)
        F = features_estimador(dfm)
        real = np.c_[df["ball_x"].to_numpy(), df["ball_y"].to_numpy()][mascara]
        est = {
            "último visto": np.c_[F["ux"], F["uy"]][mascara],
            "interpolación": np.c_[F["ix"], F["iy"]][mascara],
            "modelo": np.c_[predecir_balon(modelos, F[mascara])],
        }
        hueco = F["hueco"].to_numpy()[mascara]
        for nombre, e in est.items():
            err = np.hypot(*(e - real).T)
            fila = {"prueba": prueba, "método": nombre, "todos": np.nanmean(err),
                    "<5 m": np.nanmean(err < 5)}
            for a, b, n in zip(bordes, bordes[1:], nombres):
                m = (hueco > a) & (hueco <= b)
                fila[n] = np.nanmean(err[m]) if m.any() else np.nan
            filas.append(fila)
    tabla = pd.DataFrame(filas).set_index(["prueba", "método"])
    print(tabla.round(2).to_string())
    print("\nPromedio por método:")
    print(tabla.groupby("método").mean().round(2).to_string())

    imp = pd.Series(modelos["x"].feature_importances_, index=F_tr.columns).sort_values(ascending=False)
    print("\nTop 12 features (x, último modelo):")
    print(imp.head(12).to_string())


if __name__ == "__main__":
    main()
