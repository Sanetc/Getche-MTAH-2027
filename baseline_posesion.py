"""
More Than a Hack 2027 — Fase 1: línea base de posesión por fotograma
====================================================================

Pipeline:
  1. Carga tracking y eventos con `datos.py` (CSV Raw o EPTS, da igual).
  2. Calcula, por fotograma, la distancia del balón al jugador más cercano
     de cada equipo y la velocidad del balón.
  3. Asigna posesión con reglas + histéresis (Nivel 1 y 2).
  4. Mide contra la "verdad aproximada" de los eventos, busca los mejores
     parámetros en el primer partido y los valida en los demás.
  5. Exporta las predicciones con `datos.exportar()`.

Uso:
    python -I baseline_posesion.py --data ../datos/sample-data/data --out salida
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos as D  # noqa: E402
from datos import FPS, LARGO, ANCHO, NINGUNO, LOCAL, VISITA  # noqa: E402


# ----------------------------------------------------------------------------
# 1. Features por fotograma
# ----------------------------------------------------------------------------
def calcular_features(df: pd.DataFrame) -> pd.DataFrame:
    bx, by = df["ball_x"].to_numpy(), df["ball_y"].to_numpy()

    def dist_min(prefijo):
        xs = D.columnas_jugadores(df, prefijo)
        X = df[xs].to_numpy()
        Y = df[[c[:-2] + "_y" for c in xs]].to_numpy()
        d = np.sqrt((X - bx[:, None]) ** 2 + (Y - by[:, None]) ** 2)
        return np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)

    f = pd.DataFrame({"Frame": df["Frame"], "Period": df["Period"]})
    f["d_local"] = dist_min("H_")
    f["d_visita"] = dist_min("A_")

    # Velocidad del balón (m/s), suavizada
    vx = np.gradient(df["ball_x"].interpolate(limit=5).to_numpy()) * FPS
    vy = np.gradient(df["ball_y"].interpolate(limit=5).to_numpy()) * FPS
    f["v_balon"] = pd.Series(np.hypot(vx, vy)).rolling(5, center=True, min_periods=1).median()

    f["balon_fuera"] = (
        df["ball_x"].isna()
        | (df["ball_x"].abs() > LARGO / 2 + 0.5)
        | (df["ball_y"].abs() > ANCHO / 2 + 0.5)
    )
    return f


# ----------------------------------------------------------------------------
# 2. Asignación de posesión
# ----------------------------------------------------------------------------
def predecir_posesion(f: pd.DataFrame, radio=1.5, frames_cambio=3,
                      v_max=15.0, marcar_fuera=True) -> np.ndarray:
    """
    radio:          distancia máx. (m) para considerar que un jugador controla el balón
    frames_cambio:  fotogramas de control rival necesarios para cambiar posesión.
                    Los fotogramas en los que nadie controla el balón no reinician
                    la cuenta: solo la reinicia un toque del equipo que lo tenía.
    v_max:          si el balón va más rápido (m/s) nadie lo controla (pase / tiro)
    marcar_fuera:   balón fuera de la cancha -> 'Ninguno'
    """
    dl, dv = f["d_local"].to_numpy(), f["d_visita"].to_numpy()
    v, fuera = f["v_balon"].to_numpy(), f["balon_fuera"].to_numpy()
    periodo = f["Period"].to_numpy()

    # Control "instantáneo" en cada frame
    control = np.full(len(f), NINGUNO)
    cerca_l = (dl <= radio) & (dl <= dv) & (v <= v_max)
    cerca_v = (dv <= radio) & (dv < dl) & (v <= v_max)
    control[cerca_l] = LOCAL
    control[cerca_v] = VISITA

    # Histéresis: la posesión solo cambia tras `frames_cambio` frames de control rival
    pred = np.empty(len(f), dtype=int)
    actual, candidato, racha = NINGUNO, NINGUNO, 0
    for i in range(len(f)):
        if i > 0 and periodo[i] != periodo[i - 1]:
            actual, candidato, racha = NINGUNO, NINGUNO, 0
        c = control[i]
        if c != NINGUNO and c != actual:
            racha = racha + 1 if c == candidato else 1
            candidato = c
            if racha >= frames_cambio or actual == NINGUNO:
                actual, racha = c, 0
        elif c == actual:
            racha = 0
        if marcar_fuera and fuera[i]:
            pred[i] = NINGUNO
            actual = NINGUNO
        else:
            pred[i] = actual
    return pred


GRID = dict(radio=[1.0, 1.5, 2.0, 3.0], frames_cambio=[1, 3, 5, 10],
            v_max=[10.0, 15.0, 25.0], marcar_fuera=[True, False])


def buscar_parametros(partidos, grid=GRID, metrica="spell_f1"):
    """
    partidos: lista de (features, verdad). Devuelve [(métrica media, params), ...]
    ordenada de mejor a peor. metrica: "spell_f1" (la del ranking oficial) o
    "exactitud".
    """
    from evaluacion import spell_f1
    resultados = []
    for valores in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), valores))
        puntos = []
        for f, gt in partidos:
            p = predecir_posesion(f, **params)
            puntos.append(spell_f1(p, gt, f["Period"].to_numpy()) if metrica == "spell_f1"
                          else D.exactitud(p, gt))
        resultados.append((float(np.mean(puntos)), params))
    resultados.sort(key=lambda r: -r[0])
    return resultados


# ----------------------------------------------------------------------------
# 3. Carga de todos los partidos de una carpeta
# ----------------------------------------------------------------------------
def listar_juegos(carpeta: Path) -> list:
    """Subcarpetas con un formato reconocido, en orden alfabético."""
    juegos = []
    for sub in sorted(carpeta.iterdir()):
        if sub.is_dir():
            try:
                D.detectar_formato(sub, sub.name)
                juegos.append(sub.name)
            except FileNotFoundError:
                pass
    return juegos


def cargar_partido(carpeta: Path, juego: str, definicion="oficial", max_toque=25):
    """Devuelve (tracking, verdad aproximada con esa definición, ver datos.verdad_desde_eventos)."""
    df = D.cargar_tracking(carpeta / juego, juego)
    ev = D.cargar_eventos(carpeta / juego, juego)
    gt = D.verdad_desde_eventos(ev, df["Frame"], definicion, max_toque)
    return df, gt


# ----------------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("salida"))
    ap.add_argument("--juegos", nargs="*", help="por defecto, todas las subcarpetas reconocidas")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    juegos = args.juegos or listar_juegos(args.data)
    cache = {}
    for j in juegos:
        print(f"Cargando {j}...")
        df, gt = cargar_partido(args.data, j)
        cache[j] = (calcular_features(df), gt)

    from evaluacion import CABECERA, evaluar, fila

    # Nivel 1: regla simple sin histéresis ni filtros
    print("\n== Nivel 1 (jugador más cercano, sin histéresis) ==")
    print(CABECERA)
    for j, (f, gt) in cache.items():
        p = predecir_posesion(f, radio=1.5, frames_cambio=1, v_max=1e9, marcar_fuera=False)
        print(fila(j, evaluar(p, gt, f["Period"].to_numpy())))

    # Nivel 2: búsqueda de parámetros por spellF1 (se ajusta en el primer juego, se valida en los demás)
    ajuste = juegos[0]
    print(f"\n== Nivel 2: búsqueda de parámetros por spellF1 (ajuste en {ajuste}) ==")
    resultados = buscar_parametros([cache[ajuste]])
    for puntos, params in resultados[:5]:
        print(f"  {puntos:.4f}  {params}")
    mejor = resultados[0][1]

    print("\n== Mejor configuración en los demás partidos (validación) ==")
    print(CABECERA)
    for j in juegos[1:]:
        f, gt = cache[j]
        p = predecir_posesion(f, **mejor)
        print(fila(j, evaluar(p, gt, f["Period"].to_numpy())))
        print(D.matriz_confusion(p, gt).round(3).to_string(), "\n")

    for j, (f, gt) in cache.items():
        p = predecir_posesion(f, **mejor)
        ruta = D.exportar(args.out / f"{j}_posesion.csv", f["Frame"], f["Period"], p, verdad=gt)
        print(f"  Exportado: {ruta}")


if __name__ == "__main__":
    main()
