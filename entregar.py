"""
More Than a Hack 2027 — Fase 1: entrenar el modelo final y generar la entrega
=============================================================================

Dos modos:

  Entrenar con todos los partidos de una carpeta (necesitan eventos o etiquetas)
  y guardar el modelo; con --enmascarar 0.41 --balon estimar se entrena en las
  condiciones de los datos oficiales (balón ausente en juego) y el estimador
  de balón viaja dentro del modelo:
      python -I entregar.py entrenar --data ../datos/sample-data/data --modelo modelos/posesion.pkl --enmascarar 0.41 --balon estimar

  Predecir los partidos de una carpeta (con o sin eventos) y escribir los CSV
  de entrega según `datos.FORMATO_ENTREGA`:
      python -I entregar.py predecir --data ../datos/oficiales --modelo modelos/posesion.pkl --out entregas

El modelo guardado incluye los parámetros de la regla, p_quedarse de Viterbi y
el orden de las columnas, así que predecir no reajusta nada.
"""

import argparse
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos as D  # noqa: E402
from baseline_posesion import buscar_parametros, listar_juegos  # noqa: E402
from evaluacion import MIN_PD_GRID, evaluar  # noqa: E402
from modelo_posesion import (  # noqa: E402
    agregar_regla, concatenar, elegir_suavizado, entrenar, features_base, suavizar,
)
from balon_estimado import (  # noqa: E402
    enmascarar_balon, entrenar_estimador, estimar_balon, mostrar_balon_en_paradas, preparar_entrenamiento,
    ruido_balon,
)
from tracking_sin_id import quitar_identidades  # noqa: E402
from smart_tagging import anclar_saques  # noqa: E402


def preparar(carpeta: Path, juego: str, estimador=None, enmascarar=0.0, semilla=0,
             balon="ninguno", sin_identidades=False, paradas_visibles=0.0, verdad="oficial", max_toque=25,
             parada=False, ruido=None) -> dict:
    """
    Tracking + features; la verdad solo si hay eventos. `balon`: cómo rellenar
    el balón ausente ("ninguno", "interpolar" o "estimar" con `estimador`); deja
    `balon_visto` para el modelo. `enmascarar` > 0 oculta antes esa fracción del
    balón en juego, `paradas_visibles` > 0 deja el balón visible y quieto en esa
    fracción de las paradas y `sin_identidades` destruye la identidad de los
    jugadores (solo para entrenar con las condiciones de los datos oficiales).
    `ruido` = (sigma, tau, blanco) suma al balón visto de los partidos de
    entrenamiento el ruido de detección calibrado con el ATD oficial
    (`balon_estimado.ruido_balon`); nunca se aplica al predecir.
    """
    df = D.cargar_tracking(carpeta / juego, juego)
    gt = None
    if D.tiene_eventos(carpeta / juego, juego):
        gt = D.verdad_desde_eventos(D.cargar_eventos(carpeta / juego, juego), df["Frame"], verdad, max_toque)
    if sin_identidades:
        df, _ = quitar_identidades(df, semilla)
    df_real = df
    if enmascarar > 0 and gt is not None:
        df, _ = enmascarar_balon(df, gt, enmascarar, semilla)
    if paradas_visibles > 0 and gt is not None:
        df, _ = mostrar_balon_en_paradas(df, gt, paradas_visibles, semilla)
    if ruido and gt is not None:
        df = ruido_balon(df, *ruido, semilla=semilla)
    if balon == "estimar" and estimador is not None:
        df = estimar_balon(df, estimador)
    elif balon == "interpolar":
        df = estimar_balon(df)
    F0, base = features_base(df, parada=parada)
    return dict(df=df, df_real=df_real, F0=F0, base=base, gt=gt)


def modo_entrenar(args):
    juegos = args.juegos or listar_juegos(args.data)
    estimador = None
    if args.balon == "estimar":
        print("Entrenando el estimador de balón...")
        partidos = {}
        for j in juegos:
            df = D.cargar_tracking(args.data / j, j)
            partidos[j] = (df, D.verdad_desde_eventos(D.cargar_eventos(args.data / j, j), df["Frame"], args.verdad, args.max_toque))
        F_b, x_b, y_b = preparar_entrenamiento(partidos, max(args.enmascarar, 0.3))
        estimador = entrenar_estimador(F_b, x_b, y_b)

    datos = {}
    for k, j in enumerate(juegos):
        print(f"Preparando {j}...")
        datos[j] = preparar(args.data, j, estimador, args.enmascarar, semilla=k, balon=args.balon,
                            sin_identidades=args.sin_identidades, paradas_visibles=args.paradas_visibles,
                            verdad=args.verdad, max_toque=args.max_toque, parada=args.features_parada,
                            ruido=args.ruido_balon)
        if datos[j]["gt"] is None:
            sys.exit(f"{j} no tiene eventos ni etiquetas; no sirve para entrenar.")

    print("Ajustando la regla...")
    params = buscar_parametros([(d["base"], d["gt"]) for d in datos.values()])[0][1]
    for d in datos.values():
        d["F"] = agregar_regla(d["F0"], d["base"], params)

    print("Eligiendo el suavizado por spellF1 (fuera de muestra)...")
    grid_pd = MIN_PD_GRID if args.entre_paradas else [1]
    q, min_frames, min_d, min_pd = (elegir_suavizado(datos, juegos, args.arboles, args.hibrido, grid_pd=grid_pd)
                                    if len(juegos) > 1 else (0.999, 12, 1, 1))

    print(f"Entrenando con {len(juegos)} partidos...")
    F, y = concatenar(datos, juegos)
    modelo = entrenar(F, y, args.arboles, args.hibrido)

    paquete = dict(modelo=modelo, params_regla=params, p_quedarse=q, min_frames=min_frames, min_d=min_d, min_pd=min_pd,
                   columnas=list(F.columns), juegos=juegos, fecha=str(date.today()),
                   estimador=estimador, enmascarar=args.enmascarar, balon=args.balon,
                   paradas_visibles=args.paradas_visibles, verdad=args.verdad, max_toque=args.max_toque,
                   features_parada=args.features_parada, ruido_balon=args.ruido_balon)
    args.modelo.parent.mkdir(parents=True, exist_ok=True)
    with open(args.modelo, "wb") as fh:
        pickle.dump(paquete, fh)
    print(f"Guardado {args.modelo}  (regla={params}, p_quedarse={q}, min_frames={min_frames}, min_d={min_d}, min_pd={min_pd})")


def modo_predecir(args):
    with open(args.modelo, "rb") as fh:
        paq = pickle.load(fh)
    print(f"Modelo del {paq['fecha']} entrenado con {paq['juegos']}"
          f" (balón oculto al entrenar: {paq.get('enmascarar', 0):.0%}, "
          f"paradas con balón visible: {paq.get('paradas_visibles', 0):.0%}, "
          f"relleno del balón: {paq.get('balon', 'ninguno')}, "
          f"ruido de balón al entrenar: {paq.get('ruido_balon') or 'no'})")

    juegos = args.juegos or listar_juegos(args.data)
    for j in juegos:
        print(f"Prediciendo {j}...")
        d = preparar(args.data, j, paq.get("estimador"), balon=paq.get("balon", "ninguno"),
                     parada=paq.get("features_parada", False))
        F = agregar_regla(d["F0"], d["base"], paq["params_regla"])[paq["columnas"]]
        probs = paq["modelo"].predict_proba(F)
        per = F["periodo"].to_numpy()
        pred = suavizar(probs, paq["p_quedarse"], paq.get("min_frames", 1), per, min_d=paq.get("min_d", 1),
                        min_pd=paq.get("min_pd", 1))
        oficial = D.detectar_formato(args.data / j, j) == "oficial"
        if args.ancla_saques and D.tiene_smart_tagging(args.data / j, j):
            fases = D.cargar_smart_tagging(args.data / j, j)
            fases = fases[fases["tipo"].notna()]                 # solo SET PIECES
            primer = int(d["df"]["Frame"].iloc[0])
            fases = fases.assign(inicio=fases["inicio"] - primer, fin=fases["fin"] - primer)
            antes = pred.copy()
            pred = anclar_saques(pred, fases, per)
            print(f"  ancla de saques: {len(fases)} fases, {(pred != antes).sum()} fotogramas cambiados")
        pred[D.fuera_de_periodo(d["df"])] = D.NINGUNO           # sin periodo = balón muerto seguro
        if oficial and not args.sin_saque and not D.tiene_smart_tagging(args.data / j, j):
            sys.exit(f"{j}: falta *_smart_tagging.xml en {args.data / j} (regla del saque inicial); "
                     f"--sin-saque la omite, pero el archivo no sería el entregado")
        if oficial and not args.sin_saque:
            # antes de la fase 'Kick Off (Start)' de cada parte el balón está muerto (regla general, del Smart Tagging)
            antes = D.antes_del_saque(d["df"], D.cargar_smart_tagging(args.data / j, j))
            print(f"  antes del saque inicial de cada parte: {int(antes.sum())} fotogramas a D "
                  f"({int((pred[antes] != D.NINGUNO).sum())} cambian)")
            pred[antes] = D.NINGUNO
        ruta = D.exportar(args.out / f"{j}.csv", d["df"]["Frame"], d["df"]["Period"], pred,
                          probs=probs if args.con_verdad else None,
                          verdad=d["gt"] if args.con_verdad else None,
                          n_frames=D.FORMATO_ENTREGA.get("n_frames") if oficial else None)
        resumen = ", ".join(f"{D.NOMBRES[k]} {np.mean(pred == k):.1%}" for k in D.NOMBRES)
        extra = ""
        if d["gt"] is not None:
            r = evaluar(pred, d["gt"], per)
            extra = f"  spellF1 {r['spell_f1']:.4f}  macroF1 {r['macro_f1']:.4f} vs. eventos"
        print(f"  {ruta}  ({len(pred)} fotogramas: {resumen}){extra}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="modo", required=True)
    for nombre in ("entrenar", "predecir"):
        p = sub.add_parser(nombre)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--modelo", type=Path, default=Path("modelos/posesion.pkl"))
        p.add_argument("--juegos", nargs="*")
    sub.choices["entrenar"].add_argument("--arboles", type=int, default=400)
    sub.choices["entrenar"].add_argument("--enmascarar", type=float, default=0.0,
                                         help="fracción del balón en juego que se oculta al entrenar (oficial: 0.41)")
    sub.choices["entrenar"].add_argument("--balon", choices=["ninguno", "interpolar", "estimar"], default="interpolar",
                                         help="relleno del balón ausente: nada, interpolación lineal (por defecto) o estimador colectivo")
    sub.choices["entrenar"].add_argument("--paradas-visibles", type=float, default=0.0,
                                         help="fracción de fotogramas D con el balón visible y quieto (oficial ~0.15)")
    sub.choices["entrenar"].add_argument("--verdad", choices=["oficial", "control"], default="oficial",
                                         help="definición de la verdad de entrenamiento (ver datos.verdad_desde_eventos)")
    sub.choices["entrenar"].add_argument("--max-toque", type=int, default=25,
                                         help="con --verdad control: fotogramas por debajo de los que un toque sin pase ni tiro no cambia la posesión")
    sub.choices["entrenar"].add_argument("--ruido-balon", type=float, nargs=3, default=None,
                                         metavar=("SIGMA", "TAU", "BLANCO"),
                                         help="ruido de detección del balón en los partidos de entrenamiento: desvío lento "
                                              "(m por eje), su tiempo de correlación (fotogramas) y temblor blanco (m); "
                                              "calibrado con el ATD oficial: 0.8 30 0.06 (calibrar_ruido.py)")
    sub.choices["entrenar"].add_argument("--features-parada", action="store_true",
                                         help="features de balón parado y visible (modelo_posesion.features_parada)")
    sub.choices["entrenar"].add_argument("--entre-paradas", action="store_true",
                                         help="elige también min_pd: posesiones cortas entre dos paradas pasan a D "
                                              "(evaluacion.quitar_posesiones_entre_paradas)")
    sub.choices["entrenar"].add_argument("--hibrido", action="store_true",
                                         help="dos especialistas: con balón donde se ve, sin balón donde no")
    sub.choices["entrenar"].add_argument("--sin-identidades", action="store_true",
                                         help="entrena con slots sin identidad de jugador (como el ATD oficial)")
    sub.choices["predecir"].add_argument("--out", type=Path, default=Path("entregas"))
    sub.choices["predecir"].add_argument("--con-verdad", action="store_true",
                                         help="añade probabilidades y verdad_aprox (NO para la entrega)")
    sub.choices["predecir"].add_argument("--sin-saque", action="store_true",
                                         help="NO fuerza D antes de la fase 'Kick Off (Start)' de cada parte "
                                              "(comportamiento anterior al 17 nov)")
    sub.choices["predecir"].add_argument("--ancla-saques", action="store_true",
                                         help="usa las fases SET PIECES del Smart Tagging oficial como ancla del equipo del saque")
    args = ap.parse_args()
    if args.modo == "entrenar" and args.ruido_balon is not None and not any(args.ruido_balon):
        args.ruido_balon = None                              # 0 0 0 = sin ruido
    (modo_entrenar if args.modo == "entrenar" else modo_predecir)(args)


if __name__ == "__main__":
    main()
