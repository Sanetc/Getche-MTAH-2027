"""
Simular el tracking oficial SIN identidades de jugador ("track slots")
======================================================================

El ATD oficial (DATA.md) tiene 46 "track slots" para 22 jugadores, con equipo
pero sin jugador: los slots aparecen, desaparecen y se intercambian cuando el
tracker pierde y recupera gente; en juego hay de media 20,5 objetos
trackeados y a veces más de 22 (duplicados). Los sample data traen jugadores
con identidad y sin huecos en juego, así que aquí se destruye la identidad
para entrenar y validar en las condiciones oficiales:

  1. Intercambios: cuando dos jugadores del mismo equipo se cruzan a menos de
     `dist_intercambio` m, con probabilidad `p_intercambio` sus pistas se
     intercambian desde ese momento (el tracker se confunde de persona).
  2. Cortes: cada pista se parte en "tracklets" de duración exponencial
     (media `tracklet_s`); entre uno y el siguiente el jugador desaparece un
     hueco log-normal (mediana `hueco_mediana_s`) o, con probabilidad
     `p_duplicado`, el slot viejo sigue un rato con una copia (duplicado)
     mientras el nuevo ya lo trackea.
  3. Slots: cada tracklet ocupa el slot libre de índice más bajo de su equipo
     (`n_slots` por equipo, 23 + 23 = 46). Las columnas pasan a llamarse
     `H_s01_x`… / `A_s01_x`…, y el resto del código las trata como jugadores.

Uso dentro de los scripts: `quitar_identidades(df, semilla)`; `--sin-identidades`
en modelo_posesion.py y entregar.py. Prueba rápida de las estadísticas:
    python -I tracking_sin_id.py --data ../datos/sample-data/data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos as D  # noqa: E402
from datos import FPS  # noqa: E402


def _rachas(mascara: np.ndarray):
    """Inicios y finales (exclusivos) de las rachas de True."""
    m = np.r_[False, mascara, False].astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _intercambiar(X, Y, rng, p_intercambio, dist_intercambio):
    """Intercambia pistas de jugadores del mismo equipo que se cruzan."""
    n = X.shape[1]
    eventos = []
    for i in range(n):
        for j in range(i + 1, n):
            cerca = np.hypot(X[:, i] - X[:, j], Y[:, i] - Y[:, j]) < dist_intercambio
            ini, fin = _rachas(cerca)
            for a, b in zip(ini, fin):                   # cada cruce, un posible intercambio
                if rng.random() < p_intercambio:
                    eventos.append((int(rng.integers(a, b)), i, j))
    aplicados = 0
    for t, i, j in sorted(eventos):                      # en orden: las pistas ya pueden venir intercambiadas
        if np.hypot(X[t, i] - X[t, j], Y[t, i] - Y[t, j]) < dist_intercambio:
            X[t:, [i, j]] = X[t:, [j, i]]
            Y[t:, [i, j]] = Y[t:, [j, i]]
            aplicados += 1
    return aplicados


def _tracklets(X, rng, tracklet_s, hueco_mediana_s, hueco_sigma, p_duplicado, dup_s):
    """
    Parte cada pista en tracklets. Devuelve una lista de
    (inicio, fin, columna, duplicado) con fin exclusivo.
    """
    T, n = X.shape
    out = []
    for c in range(n):
        ini, fin = _rachas(np.isfinite(X[:, c]))
        for a, b in zip(ini, fin):
            t = a
            while t < b:
                largo = max(int(rng.exponential(tracklet_s * FPS)), 2)
                e = min(t + largo, b)
                out.append((t, e, c, False))
                if e >= b:
                    break
                if rng.random() < p_duplicado:           # el slot viejo sigue con un fantasma
                    d = min(e + max(int(rng.exponential(dup_s * FPS)), 2), b)
                    out.append((e, d, c, True))
                    t = e                                # el nuevo tracklet empieza ya
                else:                                    # hueco: el jugador desaparece
                    hueco = int(np.clip(rng.lognormal(np.log(hueco_mediana_s * FPS), hueco_sigma), 1, 15 * FPS))
                    t = min(e + hueco, b)
    return out


def _a_slots(X, Y, tracklets, n_slots, rng, jitter):
    """Coloca cada tracklet en el slot libre más bajo. Devuelve (Xs, Ys)."""
    T = X.shape[0]
    libre_desde = np.zeros(n_slots, dtype=int)           # primer fotograma en que el slot está libre
    Xs = np.full((T, n_slots), np.nan)
    Ys = np.full((T, n_slots), np.nan)
    for a, b, c, dup in sorted(tracklets):
        candidatos = np.flatnonzero(libre_desde <= a)
        if len(candidatos) == 0:                         # no debería pasar con 23 slots por equipo
            libre_desde = np.r_[libre_desde, 0]
            Xs, Ys = np.c_[Xs, np.full(T, np.nan)], np.c_[Ys, np.full(T, np.nan)]
            candidatos = np.array([len(libre_desde) - 1])
        s = candidatos[0]
        Xs[a:b, s], Ys[a:b, s] = X[a:b, c], Y[a:b, c]
        if dup:
            Xs[a:b, s] += rng.normal(0, jitter, b - a)
            Ys[a:b, s] += rng.normal(0, jitter, b - a)
        libre_desde[s] = b + 1
    return Xs, Ys


def quitar_identidades(df: pd.DataFrame, semilla=0, tracklet_s=20.0, hueco_mediana_s=1.0,
                       hueco_sigma=1.0, p_intercambio=0.3, dist_intercambio=1.5,
                       p_duplicado=0.1, dup_s=2.0, n_slots=23, jitter=0.5):
    """
    Sustituye las columnas de jugadores por slots sin identidad (ver cabecera).
    Devuelve (df, info) con info = dict(intercambios, tracklets, visibles_medio).
    """
    rng = np.random.default_rng(semilla)
    otras = [c for c in df.columns
             if not (c.startswith(("H_", "A_")) and c.endswith(("_x", "_y")))]
    out = df[otras].copy()
    info = dict(intercambios=0, tracklets=0, duplicados=0)
    visibles = np.zeros(len(df))
    for prefijo in ("H_", "A_"):
        xs = D.columnas_jugadores(df, prefijo)
        X = df[xs].to_numpy(dtype=float).copy()
        Y = df[[c[:-2] + "_y" for c in xs]].to_numpy(dtype=float).copy()
        info["intercambios"] += _intercambiar(X, Y, rng, p_intercambio, dist_intercambio)
        tl = _tracklets(X, rng, tracklet_s, hueco_mediana_s, hueco_sigma, p_duplicado, dup_s)
        info["tracklets"] += len(tl)
        info["duplicados"] += sum(d for *_, d in tl)
        Xs, Ys = _a_slots(X, Y, tl, n_slots, rng, jitter)
        for s in range(Xs.shape[1]):
            out[f"{prefijo}s{s + 1:02d}_x"] = Xs[:, s]
            out[f"{prefijo}s{s + 1:02d}_y"] = Ys[:, s]
        visibles += np.isfinite(Xs).sum(1)
    info["visibles_medio"] = float(visibles.mean())
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--juegos", nargs="*")
    args = ap.parse_args()
    from baseline_posesion import cargar_partido, listar_juegos
    from datos import LOCAL, VISITA
    for k, j in enumerate(args.juegos or listar_juegos(args.data)):
        df, gt = cargar_partido(args.data, j)
        en_juego = np.isin(gt, (LOCAL, VISITA))
        xs = D.columnas_jugadores(df, "H_") + D.columnas_jugadores(df, "A_")
        antes = np.isfinite(df[xs].to_numpy()).sum(1)[en_juego].mean()
        sin, info = quitar_identidades(df, semilla=k)
        xs2 = D.columnas_jugadores(sin, "H_") + D.columnas_jugadores(sin, "A_")
        vis = np.isfinite(sin[xs2].to_numpy()).sum(1)[en_juego]
        print(f"{j}: {len(xs2)} slots, {info['tracklets']} tracklets ({info['duplicados']} duplicados), "
              f"{info['intercambios']} intercambios; visibles en juego {antes:.1f} -> {vis.mean():.1f} "
              f"(> 22 en {np.mean(vis > 22):.1%} de los fotogramas)")


if __name__ == "__main__":
    main()
