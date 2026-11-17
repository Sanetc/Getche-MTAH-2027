"""
More Than a Hack 2027 — calibrar el ruido de balón con el ATD oficial
======================================================================

Los sample data son tracking corregido: el balón va pegado al jugador que lo
lleva. El ATD oficial es detección automática: el balón queda a ~1 m del jugador
más cercano y su velocidad tiembla. `balon_estimado.ruido_balon` añade ese ruido
a los partidos de entrenamiento; este guion elige sus tres parámetros (desvío
lento `sigma`, tiempo de correlación `tau`, temblor blanco) comparando, entre
los sample data con ruido y el partido oficial, los cuantiles de cuatro
estadísticos SIN etiquetas, con el balón visto:

  - distancia del balón al jugador más cercano (de cualquier equipo),
  - el mínimo de esa distancia en los 2 s anteriores,
  - aceleración y velocidad del balón (como las calcula el modelo).

Del partido oficial solo se imprimen agregados (cuantiles); no se usa ninguna
etiqueta suya. Con `--entrega` (un CSV de predicción ya generado, p. ej. el del
modelo sin ruido) la comparación se restringe a los fotogramas que esa
predicción da como balón en juego, y en los sample data a los que la verdad de
eventos da en juego; sin ella se usan todos los fotogramas con balón visto.

    python -I calibrar_ruido.py --data ../datos/sample-data/data --oficial ../datos/oficiales/data [--entrega entregas/Getche.csv]
"""

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos as D  # noqa: E402
from datos import FPS, LOCAL, VISITA  # noqa: E402
from balon_estimado import mostrar_balon_en_paradas, ruido_balon  # noqa: E402
from baseline_posesion import cargar_partido, listar_juegos  # noqa: E402
from tracking_sin_id import quitar_identidades  # noqa: E402

QS = (10, 25, 50, 75, 90)
CLAVES = ("dmin", "d50", "acc", "v")


def estadisticos(df: pd.DataFrame, mascara: np.ndarray) -> dict:
    """Cuantiles, en `mascara`, de la distancia mínima jugador–balón, su mínimo en 2 s, y aceleración y velocidad del balón."""
    per = df["Period"].to_numpy()
    bx = df["ball_x"].groupby(per).transform(lambda s: s.interpolate(limit_area="inside")).to_numpy()
    by = df["ball_y"].groupby(per).transform(lambda s: s.interpolate(limit_area="inside")).to_numpy()
    cols = D.columnas_jugadores(df, "H_") + D.columnas_jugadores(df, "A_")
    d = np.hypot(df[cols].to_numpy() - bx[:, None], df[[c[:-2] + "_y" for c in cols]].to_numpy() - by[:, None])
    dmin = np.clip(np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1), 0, 60)
    d50 = pd.Series(dmin).groupby(per).transform(lambda s: s.rolling(50, min_periods=1).min()).to_numpy()
    vx = np.gradient(pd.Series(bx).interpolate(limit=5).to_numpy()) * FPS      # como baseline_posesion.calcular_features
    vy = np.gradient(pd.Series(by).interpolate(limit=5).to_numpy()) * FPS
    v = pd.Series(np.hypot(vx, vy)).rolling(5, center=True, min_periods=1).median().to_numpy()
    acc = np.abs(np.gradient(v) * FPS)                                         # como modelo_posesion.features_base
    m = mascara & np.isfinite(dmin) & np.isfinite(acc)
    return {"dmin": np.percentile(dmin[m], QS), "d50": np.percentile(d50[m], QS),
            "acc": np.percentile(acc[m], QS), "v": np.percentile(v[m], QS)}


def perdida(sim: dict, obj: dict, claves=CLAVES) -> float:
    return float(np.mean([np.abs(np.log((sim[k] + 0.05) / (obj[k] + 0.05))).mean() for k in claves]))


def fmt(e: dict) -> str:
    return "  ".join(f"{k}: " + "/".join(f"{x:.2f}" for x in e[k]) for k in CLAVES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="sample data")
    ap.add_argument("--oficial", type=Path, required=True, help="carpeta data del bundle oficial")
    ap.add_argument("--entrega", type=Path, help="CSV de predicción para quedarse con los fotogramas en juego")
    args = ap.parse_args()

    of = D.cargar_tracking(args.oficial / "tactical", "tactical")
    m_of = of["ball_x"].notna().to_numpy() & (of["Period"].to_numpy() > 0)
    if args.entrega:
        lab = pd.read_csv(args.entrega)["label"].to_numpy()[of["Frame"].to_numpy()]
        m_of &= np.isin(lab, ("H", "A"))
    obj = estadisticos(of, m_of)
    print(f"cuantiles p{QS}  (dmin y d50 en m, acc en m/s², v en m/s)")
    print("OFICIAL: " + fmt(obj))

    partidos = []
    for k, j in enumerate(listar_juegos(args.data)):
        df, gt = cargar_partido(args.data, j)
        df, _ = quitar_identidades(df, semilla=k)
        df, _ = mostrar_balon_en_paradas(df, gt, 0.15, semilla=k)
        m = df["ball_x"].notna().to_numpy() & (gt >= 0)
        if args.entrega:
            m &= np.isin(gt, (LOCAL, VISITA))
        partidos.append((k, df, m))
        e = estadisticos(df, m)
        print(f"{j} sin ruido: " + fmt(e) + f"   pérdida {perdida(e, obj):.3f}")

    filas = []
    for sigma, tau, blanco in itertools.product((0.6, 0.8, 1.0, 1.2), (10, 15, 20, 30, 60, 120), (0.02, 0.04, 0.06, 0.09)):
        es = [estadisticos(ruido_balon(df, sigma, tau, blanco, k), m) for k, df, m in partidos]
        e = {c: np.mean([x[c] for x in es], axis=0) for c in CLAVES}
        filas.append((perdida(e, obj), sigma, tau, blanco, e))
    filas.sort(key=lambda f: f[0])
    print("\nMejores 8 (pérdida = media de |log cociente| de los cuantiles):")
    for p, sigma, tau, blanco, e in filas[:8]:
        print(f"  sigma={sigma:.1f} tau={tau:3d} blanco={blanco:.2f}  pérdida {p:.3f}  " + fmt(e))


if __name__ == "__main__":
    main()
