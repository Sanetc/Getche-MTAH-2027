"""
Autoentrenamiento (self-training) con el partido que se predice
================================================================

El modelo se entrena con los sample data, que son otros partidos con otro
tracking: las mejoras que se veían bien ahí (posesiones cortas entre paradas,
features de balón parado, verdad "control") no se notaron en el partido
oficial (30 oct – 9 nov). Aquí el modelo aprende también del propio partido:

  1. Modelo base con los sample data (igual que `entregar.py entrenar`).
  2. Predice el partido objetivo y se queda con las pseudo-etiquetas en las que
     está muy seguro: probabilidad de la clase >= `umbral`, dentro de una racha
     suavizada de al menos `min_racha` fotogramas y a más de `margen` de sus
     bordes (los cambios son lo dudoso).
  3. Reentrena con los sample data + esas pseudo-etiquetas (peso `peso`) y
     vuelve a predecir. Los parámetros de la regla y del suavizado no cambian:
     se eligieron fuera de muestra con los sample data.

Ni etiquetas a mano ni verdad privada: todo sale del modelo y del tracking.

Validación en los sample data (leave-one-out; el partido de prueba hace de
"partido oficial": se autoentrena con él sin ver su verdad):
    python -I autoentreno.py validar --data ../datos/sample-data/data

Partido oficial (los 5 min públicos se excluyen de las pseudo-etiquetas: de esa
ventana no entra nada al entrenamiento), escribe el CSV de entrega:
    python -I autoentreno.py oficial --data ../datos/sample-data/data --oficial ../datos/oficiales/data --out salida_auto
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos as D  # noqa: E402
from baseline_posesion import buscar_parametros, listar_juegos  # noqa: E402
from evaluacion import CABECERA, evaluar, fila  # noqa: E402
from entregar import preparar  # noqa: E402
from modelo_posesion import _lgbm, agregar_regla, concatenar, elegir_suavizado, entrenar, suavizar  # noqa: E402

NINGUNO, LOCAL, VISITA = D.NINGUNO, D.LOCAL, D.VISITA
PUBLICOS = (11185, 18684)          # los 5 min de verdad pública: nunca pseudo-etiquetados


def pseudo_etiquetas(probs, ruta, periodos, umbral=0.9, margen=25, min_racha=75, excluir=None):
    """Etiqueta (0/1/2) donde el modelo está seguro, -1 en el resto."""
    y = np.full(len(ruta), -1)
    corte = np.r_[True, (ruta[1:] != ruta[:-1]) | (periodos[1:] != periodos[:-1])]
    ini = np.flatnonzero(corte)
    fin = np.r_[ini[1:], len(ruta)]
    for a, b in zip(ini, fin):
        if b - a < min_racha:
            continue
        a2, b2 = a + margen, b - margen
        k = ruta[a]
        seguro = probs[a2:b2, k] >= umbral
        y[a2:b2][seguro] = k
    if excluir is not None:
        y[excluir] = -1
    return y


def reentrenar(F_tr, y_tr, F_ps, y_ps, n_estimators, peso):
    m = y_ps >= 0
    X = pd.concat([F_tr, F_ps[m]], ignore_index=True)
    y = np.r_[y_tr, y_ps[m]]
    w = np.r_[np.ones(len(y_tr)), np.full(int(m.sum()), peso)]
    ok = y >= 0
    modelo = _lgbm(n_estimators).fit(X[ok], y[ok], sample_weight=w[ok])
    assert list(modelo.classes_) == [NINGUNO, LOCAL, VISITA]
    return modelo


def resumen_pseudo(y_ps, gt=None):
    m = y_ps >= 0
    txt = (f"  pseudo-etiquetas: {m.mean():.1%} de los fotogramas "
           f"(D {np.mean(y_ps[m] == NINGUNO):.0%}, H {np.mean(y_ps[m] == LOCAL):.0%}, A {np.mean(y_ps[m] == VISITA):.0%})")
    if gt is not None:
        txt += f"; aciertan la verdad en {np.mean(y_ps[m] == gt[m]):.1%}"
    print(txt)


def preparar_muestra(args, juegos):
    datos = {}
    for k, j in enumerate(juegos):
        print(f"Preparando {j}...")
        datos[j] = preparar(args.data, j, None, 0.0, semilla=k, balon="interpolar",
                            sin_identidades=True, paradas_visibles=args.paradas_visibles)
    return datos


def modo_validar(args):
    juegos = listar_juegos(args.data)
    datos = preparar_muestra(args, juegos)
    resumen = {"base": [], "auto": []}
    for prueba in juegos:
        entreno = [j for j in juegos if j != prueba]
        params = buscar_parametros([(datos[j]["base"], datos[j]["gt"]) for j in entreno])[0][1]
        for j in juegos:
            datos[j]["F"] = agregar_regla(datos[j]["F0"], datos[j]["base"], params)
        q, min_frames, min_d, min_pd = elegir_suavizado(datos, entreno, args.arboles, grid_pd=[1])
        F_tr, y_tr = concatenar(datos, entreno)
        F_te, gt = datos[prueba]["F"][F_tr.columns], datos[prueba]["gt"]
        per = F_te["periodo"].to_numpy()

        modelo = entrenar(F_tr, y_tr, args.arboles)
        probs = modelo.predict_proba(F_te)
        ruta = suavizar(probs, q, min_frames, per, min_d=min_d)
        salidas = {"base": ruta}
        for r in range(args.rondas):
            y_ps = pseudo_etiquetas(probs, ruta, per, args.umbral, args.margen, args.min_racha)
            resumen_pseudo(y_ps, gt)
            modelo = reentrenar(F_tr, y_tr, F_te, y_ps, args.arboles, args.peso)
            probs = modelo.predict_proba(F_te)
            ruta = suavizar(probs, q, min_frames, per, min_d=min_d)
        salidas["auto"] = ruta

        print(f"\n{prueba}  (p_quedarse={q}, min_frames={min_frames}, min_d={min_d})")
        print(CABECERA)
        for nombre, p in salidas.items():
            r = evaluar(p, gt, per)
            resumen[nombre].append(r)
            print(fila(nombre, r))

    print("\n== Promedio de los tres partidos ==")
    print(CABECERA)
    for nombre, rs in resumen.items():
        prom = {k: float(np.mean([r[k] for r in rs])) for k in
                ("spell_f1", "macro_f1", "exactitud", "n_pos_pred", "n_pos_gt", "dur_pred", "dur_gt")}
        prom["n_pos_pred"], prom["n_pos_gt"] = round(prom["n_pos_pred"]), round(prom["n_pos_gt"])
        print(fila(nombre, prom))


def modo_oficial(args):
    juegos = listar_juegos(args.data)
    datos = preparar_muestra(args, juegos)
    params = buscar_parametros([(d["base"], d["gt"]) for d in datos.values()])[0][1]
    for d in datos.values():
        d["F"] = agregar_regla(d["F0"], d["base"], params)
    q, min_frames, min_d, min_pd = elegir_suavizado(datos, juegos, args.arboles, grid_pd=[1])
    F_tr, y_tr = concatenar(datos, juegos)

    print("Preparando el partido oficial...")
    o = preparar(args.oficial, "tactical", None, balon="interpolar")
    F_o = agregar_regla(o["F0"], o["base"], params)[F_tr.columns]
    per = F_o["periodo"].to_numpy()
    frames = o["df"]["Frame"].to_numpy()
    excluir = (frames >= PUBLICOS[0]) & (frames <= PUBLICOS[1])
    excluir |= D.fuera_de_periodo(o["df"])

    modelo = entrenar(F_tr, y_tr, args.arboles)
    probs = modelo.predict_proba(F_o)
    ruta = suavizar(probs, q, min_frames, per, min_d=min_d)
    args.out.mkdir(parents=True, exist_ok=True)
    escribir(args.out / "base.csv", o, ruta)
    for r in range(args.rondas):
        y_ps = pseudo_etiquetas(probs, ruta, per, args.umbral, args.margen, args.min_racha, excluir)
        resumen_pseudo(y_ps)
        modelo = reentrenar(F_tr, y_tr, F_o, y_ps, args.arboles, args.peso)
        probs = modelo.predict_proba(F_o)
        ruta = suavizar(probs, q, min_frames, per, min_d=min_d)
    escribir(args.out / "auto.csv", o, ruta)


def escribir(ruta_csv, o, pred):
    pred = pred.copy()
    pred[D.fuera_de_periodo(o["df"])] = D.NINGUNO
    D.exportar(ruta_csv, o["df"]["Frame"], o["df"]["Period"], pred, n_frames=D.FORMATO_ENTREGA.get("n_frames"))
    print(f"  {ruta_csv}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="modo", required=True)
    for nombre in ("validar", "oficial"):
        s = sub.add_parser(nombre)
        s.add_argument("--data", type=Path, required=True, help="sample data (entrenamiento)")
        s.add_argument("--arboles", type=int, default=400)
        s.add_argument("--paradas-visibles", type=float, default=0.15)
        s.add_argument("--umbral", type=float, default=0.9, help="probabilidad mínima de la pseudo-etiqueta")
        s.add_argument("--margen", type=int, default=25, help="fotogramas descartados en cada borde de racha")
        s.add_argument("--min-racha", type=int, default=75, help="racha suavizada mínima para pseudo-etiquetar")
        s.add_argument("--peso", type=float, default=1.0, help="peso de las pseudo-etiquetas frente a los sample data")
        s.add_argument("--rondas", type=int, default=1)
    sub.choices["oficial"].add_argument("--oficial", type=Path, required=True, help="carpeta data del bundle oficial")
    sub.choices["oficial"].add_argument("--out", type=Path, default=Path("salida_auto"))
    args = ap.parse_args()
    {"validar": modo_validar, "oficial": modo_oficial}[args.modo](args)


if __name__ == "__main__":
    main()
