"""
Guion del día de los datos: comprobar el bundle oficial y la entrega
====================================================================

Imprime solo estadísticas agregadas (nada del contenido del partido), para
verificar que el cargador entiende el bundle real y que la entrega está bien
alineada, sin tener que mirar los datos:

  1. Bundle: fotogramas, periodos (primer/último fotograma frente a los
     oficiales), slots por equipo, jugadores visibles por fotograma, balón
     ausente (total y dentro de los tramos evaluados), fases del Smart Tagging
     por tipo y equipo, homografía válida.
  2. Entrega (opcional, --entrega): validate_submission.py oficial, reparto de
     etiquetas, y spellF1 / macroF1 en los 5 minutos públicos con
     run_evaluation.py junto a los cuatro baselines (solo para comprobar la
     alineación: NO ajustar nada contra ese número).

    python -I comprobar_oficial.py --data ../datos/oficiales/data
    python -I comprobar_oficial.py --data ../datos/oficiales/data --entrega entregas/<equipo>.csv
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # -I ignora PYTHONIOENCODING
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos as D  # noqa: E402
from baseline_posesion import listar_juegos  # noqa: E402

# Tramos oficiales (DATA.md): fotogramas evaluados de cada parte y saques.
PARTES, SAQUES = D.PARTES_OFICIALES, D.SAQUES_OFICIALES
REPO_OFICIAL = Path("../datos/mtah-2027-oficial")


def pct(x) -> str:
    return f"{100 * float(x):.1f}%"


def comprobar_bundle(carpeta: Path, juego: str):
    print(f"== Bundle {juego} en {carpeta} ({D.detectar_formato(carpeta, juego)})")
    rutas = D.archivos_oficiales(carpeta, juego) if D.detectar_formato(carpeta, juego) == "oficial" else {}
    for k, v in rutas.items():
        print(f"   {k:14s} {v.name}  ({v.stat().st_size / 1e6:.1f} MB)")

    df = D.cargar_tracking(carpeta, juego)
    n, f0, f1 = len(df), int(df["Frame"].iloc[0]), int(df["Frame"].iloc[-1])
    esperados = D.FORMATO_ENTREGA["n_frames"]
    print(f"\n-- Tracking: {n} fotogramas, Frame {f0}…{f1}"
          f"  (entrega: {esperados} filas → {'OK' if n == esperados and f0 == 0 else 'REVISAR'})")
    saltos = int((np.diff(df["Frame"].to_numpy()) != 1).sum())
    print(f"   fotogramas no consecutivos: {saltos}")
    for p, g in df.groupby("Period"):
        a, b = int(g["Frame"].iloc[0]), int(g["Frame"].iloc[-1])
        extra = ""
        if p in SAQUES:
            extra = (f"   oficial: saque {SAQUES[p]}, evaluado {PARTES[p][0]}–{PARTES[p][1]}"
                     f" → {'OK' if (a, b) == PARTES[p] else 'REVISAR'}")
        print(f"   periodo {p}: {len(g):7d} fotogramas, Frame {a}…{b}{extra}")

    for eq, nombre in (("H_", "local"), ("A_", "visitante")):
        cols = D.columnas_jugadores(df, eq)                   # ya son las columnas _x
        vis = df[cols].notna().sum(axis=1)
        print(f"   {nombre:9s}: {len(cols):2d} slots, visibles por fotograma media {vis.mean():.1f},"
              f" máx {int(vis.max())}, >11 en {pct((vis > 11).mean())} de los fotogramas")
    vis_tot = sum(df[D.columnas_jugadores(df, eq)].notna().sum(axis=1) for eq in ("H_", "A_"))
    print(f"   total     : visibles media {vis_tot.mean():.1f}, >22 en {pct((vis_tot > 22).mean())},"
          f" 0 en {pct((vis_tot == 0).mean())}")

    sin_balon = df["ball_x"].isna().to_numpy()
    fr = df["Frame"].to_numpy()
    evaluado = np.zeros(n, bool)
    for a, b in PARTES.values():
        evaluado |= (fr >= a) & (fr <= b)
    print(f"   balón ausente: {pct(sin_balon.mean())} del total,"
          f" {pct(sin_balon[evaluado].mean())} en los tramos evaluados ({int(evaluado.sum())} fotogramas)")
    if sin_balon[evaluado].any():
        r = np.flatnonzero(np.diff(np.r_[0, sin_balon[evaluado].astype(int), 0]))
        huecos = (r[1::2] - r[::2]) / D.FPS
        print(f"   huecos de balón en los tramos evaluados (paradas incluidas): {len(huecos)}, mediana {np.median(huecos):.1f} s,"
              f" p90 {np.percentile(huecos, 90):.1f} s, máximo {huecos.max():.1f} s")
    vb = np.hypot(*np.gradient(df[["ball_x", "ball_y"]].to_numpy(), axis=0).T) * D.FPS
    print(f"   velocidad del balón: mediana {np.nanmedian(vb):.1f} m/s, p99 {np.nanpercentile(vb, 99):.1f} m/s"
          f" (más de 40 m/s en {pct(np.nanmean(vb > 40))}: saltos de pista)")

    if D.tiene_smart_tagging(carpeta, juego):
        st = D.cargar_smart_tagging(carpeta, juego)
        print(f"\n-- Smart Tagging: {len(st)} fases, {st['codigo'].nunique()} códigos distintos")
        sp = st[st["tipo"].notna()]
        print(f"   SET PIECES: {len(sp)}")
        for (tipo, eq), g in sp.groupby(["tipo", "equipo"]):
            nombre = {D.LOCAL: "H", D.VISITA: "A", D.SIN_EQUIPO: "N/A"}.get(eq, eq)
            print(f"     {tipo:12s} {nombre:3s} {len(g):3d}   duración media {((g['fin'] - g['inicio']) / D.FPS).mean():.1f} s")
        for p, g in sp.groupby("half"):
            num = 1 if str(p).startswith("1") else 2 if str(p).startswith("2") else None
            print(f"   {p}: primera SET PIECE en el fotograma {int(g['inicio'].min())}"
                  f" (saque oficial {SAQUES.get(num, '?')})")
    if "homografia" in rutas:
        h = D.cargar_homografia(carpeta, juego)
        print(f"\n-- Homografía: {len(h)} filas, válida en {pct(h['valid'].mean())}")
    return df


def comprobar_entrega(entrega: Path, repo: Path):
    print(f"\n== Entrega {entrega}")
    entrega, repo = entrega.resolve(), repo.resolve()
    sub = pd.read_csv(entrega)
    print(f"   {len(sub)} filas, columnas {list(sub.columns)}, reparto: "
          + ", ".join(f"{k} {pct(v)}" for k, v in sub["label"].value_counts(normalize=True).items()))
    fr = sub["frame"].to_numpy()
    for p, (a, b) in PARTES.items():
        g = sub[(fr >= a) & (fr <= b)]["label"]
        print(f"   parte {p} evaluada: " + ", ".join(f"{k} {pct(v)}" for k, v in g.value_counts(normalize=True).items()))
    validador = repo / "evaluation" / "validate_submission.py"
    evaluador = repo / "evaluation" / "run_evaluation.py"
    if not validador.exists():
        print(f"   (sin repo oficial en {repo}: no se valida ni se puntúa)")
        return
    print("\n-- validate_submission.py:")
    r = subprocess.run([sys.executable, str(validador), str(entrega)], capture_output=True, text=True)
    print("   " + (r.stdout + r.stderr).strip().replace("\n", "\n   "))
    print("\n-- run_evaluation.py en los 5 minutos públicos (junto a los baselines):")
    with tempfile.TemporaryDirectory() as tmp:
        res, out = Path(tmp) / "results", Path(tmp) / "out"
        res.mkdir()
        for b in (repo / "evaluation" / "results").glob("baseline_*.csv"):
            shutil.copy(b, res / b.name)
        shutil.copy(entrega, res / entrega.name)
        r = subprocess.run([sys.executable, str(evaluador), "--results", str(res), "--out", str(out),
                            "--no-detail", "--bootstrap", "0"],
                           capture_output=True, text=True, cwd=str(repo / "evaluation"))
        salida = (r.stdout + r.stderr).strip().split("\nmacroF1 would order")[0]   # la tabla, sin el comentario
        print("   " + salida.replace("\n", "\n   "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="carpeta con el bundle (p. ej. ../datos/oficiales/data)")
    ap.add_argument("--juegos", nargs="*")
    ap.add_argument("--entrega", type=Path, help="CSV de entrega a validar y puntuar")
    ap.add_argument("--repo", type=Path, default=REPO_OFICIAL, help="clon del repo oficial (evaluador)")
    args = ap.parse_args()
    for j in args.juegos or listar_juegos(args.data):
        comprobar_bundle(args.data / j, j)
    if args.entrega:
        comprobar_entrega(args.entrega, args.repo)


if __name__ == "__main__":
    main()
