"""
More Than a Hack 2027 — ¿dónde se equivoca el modelo?
=====================================================

Lee `salida/<juego>_posesion_modelo.csv` (generado por modelo_posesion.py) y
clasifica cada fotograma mal predicho en una categoría, en este orden:

  cambio ±N     a N fotogramas o menos de un cambio de posesión en la verdad
                (ambigüedad de cuándo empieza exactamente la posesión)
  balón en vuelo  el balón va a más de `v_vuelo` m/s (pase largo, despeje, tiro)
  disputa       los dos equipos tienen un jugador a <= 2 m del balón
  balón lejos   nadie a menos de 3 m del balón
  sin balón     el balón no está trackeado
  otro          todo lo demás

Uso:
    python -I analiza_errores.py --data ../datos/sample-data/data --salida salida
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos as D  # noqa: E402
from baseline_posesion import calcular_features, listar_juegos  # noqa: E402

CATEGORIAS = ["cambio ±N", "balón en vuelo", "disputa", "balón lejos", "sin balón", "otro"]


def clasificar(f: pd.DataFrame, gt: np.ndarray, margen: int, v_vuelo: float) -> np.ndarray:
    """Categoría de cada fotograma (se usa solo en los errores)."""
    cambios = np.flatnonzero(np.diff(gt) != 0) + 1
    dist_cambio = np.full(len(gt), np.inf)
    if len(cambios):
        idx = np.arange(len(gt))
        pos = np.searchsorted(cambios, idx)
        for vecino in (np.clip(pos - 1, 0, len(cambios) - 1), np.clip(pos, 0, len(cambios) - 1)):
            dist_cambio = np.minimum(dist_cambio, np.abs(idx - cambios[vecino]))

    dl, dv, v = f["d_local"].to_numpy(), f["d_visita"].to_numpy(), f["v_balon"].to_numpy()
    nan = ~np.isfinite(np.minimum(dl, dv))
    cat = np.full(len(gt), "otro", dtype=object)
    cat[np.minimum(dl, dv) > 3] = "balón lejos"
    cat[(dl <= 2) & (dv <= 2)] = "disputa"
    cat[v > v_vuelo] = "balón en vuelo"
    cat[dist_cambio <= margen] = "cambio ±N"
    cat[nan] = "sin balón"
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--salida", type=Path, default=Path("salida"))
    ap.add_argument("--margen", type=int, default=10, help="fotogramas alrededor de un cambio")
    ap.add_argument("--v-vuelo", type=float, default=10.0)
    args = ap.parse_args()

    etq = D.FORMATO_ENTREGA["etiquetas"]
    inv = {v: k for k, v in etq.items()}
    tablas, duraciones = [], []
    for j in listar_juegos(args.data):
        ruta = args.salida / f"{j}_posesion_modelo.csv"
        if not ruta.exists():
            continue
        s = pd.read_csv(ruta)
        pred = s[D.FORMATO_ENTREGA["columnas"]["posesion"]].map(inv).to_numpy()
        gt = s["verdad_aprox"].map(inv).fillna(-1).astype(int).to_numpy()
        f = calcular_features(D.cargar_tracking(args.data / j, j))
        cat = clasificar(f, gt, args.margen, args.v_vuelo)

        err = (gt >= 0) & (pred != gt)
        n_lab = (gt >= 0).sum()
        fila = pd.Series(cat[err]).value_counts().reindex(CATEGORIAS, fill_value=0)
        fila["TOTAL errores"] = err.sum()
        fila["% de fotogramas"] = round(err.sum() / n_lab * 100, 2)
        tablas.append(fila.rename(j))

        # ¿Los errores vienen en rachas largas o en pellizcos?
        rachas, k = [], 0
        while k < len(err):
            if err[k]:
                ini = k
                while k < len(err) and err[k]:
                    k += 1
                rachas.append(k - ini)
            else:
                k += 1
        duraciones.append(pd.Series(rachas, name=j))

        print(f"\n{j}: {err.sum()} errores en {n_lab} fotogramas etiquetados")
        print("  verdad -> predicción en los errores:")
        print(pd.crosstab(pd.Series(gt[err]).map(D.NOMBRES), pd.Series(pred[err]).map(D.NOMBRES),
                          rownames=["verdad"], colnames=["pred"]).to_string().replace("\n", "\n    "))

    print("\n== Errores por categoría (fotogramas) ==")
    print(pd.concat(tablas, axis=1).to_string())

    print("\n== Duración de las rachas de error (fotogramas) ==")
    res = pd.DataFrame({d.name: d.describe(percentiles=[.5, .9]) for d in duraciones}).round(1)
    print(res.loc[["count", "50%", "90%", "max"]].to_string())
    for d in duraciones:
        print(f"  {d.name}: {(d[d > 25].sum() / d.sum()):.0%} de los fotogramas erróneos están en rachas > 1 s")


if __name__ == "__main__":
    main()
