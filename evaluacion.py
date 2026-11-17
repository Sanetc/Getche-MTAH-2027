"""
More Than a Hack 2027 — evaluación con las métricas oficiales
=============================================================

Envuelve `oficial/possession_metrics.py` (copia del evaluador de la organización)
para puntuar nuestras predicciones internas (NINGUNO / LOCAL / VISITA / SIN_ETIQUETA)
como lo hará el ranking:

  - `spellF1`: cada posesión (racha máxima de un equipo con el balón vivo) se
    empareja 1 a 1 con las reales por IoU temporal; F1 promediado en umbrales
    0,3–0,7. Es la métrica del ranking.
  - `macroF1`: media de F1 por clase (H, A, D) fotograma a fotograma. Desempate.

Correspondencia de etiquetas: LOCAL → H, VISITA → A, NINGUNO → D, SIN_ETIQUETA → X.

Los sample data no tienen hueco de fotogramas entre partes, así que insertamos
un fotograma X artificial en cada cambio de periodo: el evaluador oficial corta
ahí las rachas, igual que hace con el descanso del partido real.

También vive aquí el post-procesado orientado a spellF1 (`fusionar_rachas_cortas`).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datos import NINGUNO, LOCAL, VISITA, SIN_ETIQUETA, FPS  # noqa: E402
from oficial import possession_metrics as pm  # noqa: E402

A_OFICIAL = {LOCAL: pm.H, VISITA: pm.A, NINGUNO: pm.D, SIN_ETIQUETA: pm.X}
ETIQUETA_OFICIAL = {LOCAL: "H", VISITA: "A", NINGUNO: "D", SIN_ETIQUETA: "X"}


def a_codigos(etiquetas) -> np.ndarray:
    """Etiquetas internas → códigos del evaluador oficial (H=0, A=1, D=2, X=3)."""
    etiquetas = np.asarray(etiquetas)
    out = np.full(len(etiquetas), pm.X, dtype=np.int8)
    for k, v in A_OFICIAL.items():
        out[etiquetas == k] = v
    return out


def _cortes(periodos) -> np.ndarray:
    """Índices donde empieza un periodo nuevo."""
    periodos = np.asarray(periodos)
    return np.flatnonzero(periodos[1:] != periodos[:-1]) + 1


def _con_cortes(codigos: np.ndarray, periodos) -> np.ndarray:
    """Inserta un X en cada cambio de periodo para que ninguna racha lo cruce."""
    return np.insert(codigos, _cortes(periodos), pm.X)


def evaluar(pred, gt, periodos) -> dict:
    """Todas las métricas oficiales, con las claves que usamos en las tablas."""
    r = pm.evaluate(_con_cortes(a_codigos(gt), periodos), _con_cortes(a_codigos(pred), periodos))
    return dict(
        spell_f1=r["spell_f1_mean"], macro_f1=r["macro_f1"], exactitud=r["accuracy"],
        n_pos_pred=r["n_pred_spells"], n_pos_gt=r["n_gt_spells"],
        dur_pred=r["mean_spell_duration_s"], dur_gt=r["gt_mean_spell_duration_s"],
        f1_vivo_muerto=r["alive_dead_f1"], abstenciones=r["n_abstain_on_graded"],
        completo=r,
    )


def spell_f1(pred, gt, periodos) -> float:
    return evaluar(pred, gt, periodos)["spell_f1"]


def fila(nombre: str, r: dict) -> str:
    return (f"{nombre:<22}{r['spell_f1']:>9.4f}{r['macro_f1']:>9.4f}{r['exactitud']:>9.2%}"
            f"{r['n_pos_pred']:>7}/{r['n_pos_gt']:<5}{r['dur_pred']:>6.1f}/{r['dur_gt']:<5.1f}")


CABECERA = f"{'':<22}{'spellF1':>9}{'macroF1':>9}{'exact.':>9}{'posesiones':>13}{'dur. (s)':>12}"


# ----------------------------------------------------------------------------
# Post-procesado orientado a spellF1
# ----------------------------------------------------------------------------
def fusionar_rachas_cortas(pred, min_frames: int, periodos) -> np.ndarray:
    """
    Elimina las posesiones (rachas de LOCAL o VISITA) de menos de `min_frames`
    fotogramas: cada una se funde con la racha anterior si esa era una posesión
    (la posesión real sigue, el parpadeo era ruido) y si no, con la siguiente
    (la posesión real empieza un poco más tarde). Nunca cruza un cambio de periodo.
    Con min_frames <= 1 devuelve la entrada tal cual.
    """
    pred = np.asarray(pred).copy()
    if min_frames <= 1:
        return pred
    limites = [0] + _cortes(periodos).tolist() + [len(pred)]
    for a, b in zip(limites, limites[1:]):
        seg = pred[a:b]
        for _ in range(3):                                  # varias pasadas por si se encadenan
            cambios = np.flatnonzero(seg[1:] != seg[:-1]) + 1
            ini = np.concatenate(([0], cambios))
            fin = np.concatenate((cambios, [len(seg)]))
            tocado = False
            for k, (i, j) in enumerate(zip(ini, fin)):
                if seg[i] in (LOCAL, VISITA) and j - i < min_frames:
                    anterior = seg[ini[k - 1]] if k > 0 else None
                    siguiente = seg[fin[k]] if fin[k] < len(seg) else None
                    nuevo = anterior if anterior in (LOCAL, VISITA) else siguiente
                    if nuevo is None or nuevo == seg[i]:
                        continue
                    seg[i:j] = nuevo
                    tocado = True
            if not tocado:
                break
        pred[a:b] = seg
    return pred


MIN_FRAMES_GRID = [1, 5, 12, 25, 50]


def quitar_paradas_cortas(pred, min_d: int, periodos) -> np.ndarray:
    """
    Elimina las paradas (rachas de NINGUNO) de menos de `min_d` fotogramas que
    están entre dos posesiones: una parada real dura segundos (saque, falta),
    y un D de un instante en pleno juego parte una posesión en dos, lo que
    spellF1 castiga mucho. La racha pasa al equipo de la posesión anterior.
    Las paradas al principio o al final de un periodo no se tocan. Con
    min_d <= 1 devuelve la entrada tal cual.
    """
    pred = np.asarray(pred).copy()
    if min_d <= 1:
        return pred
    limites = [0] + _cortes(periodos).tolist() + [len(pred)]
    for a, b in zip(limites, limites[1:]):
        seg = pred[a:b]
        cambios = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        ini = np.concatenate(([0], cambios))
        fin = np.concatenate((cambios, [len(seg)]))
        for k, (i, j) in enumerate(zip(ini, fin)):
            if seg[i] == NINGUNO and j - i < min_d and 0 < k < len(ini) - 1:
                anterior = seg[ini[k - 1]]
                if anterior in (LOCAL, VISITA):
                    seg[i:j] = anterior
        pred[a:b] = seg
    return pred


MIN_D_GRID = [1, 25, 50, 75, 125]   # 0–5 s


def quitar_posesiones_entre_paradas(pred, min_pd: int, periodos) -> np.ndarray:
    """
    Lo simétrico de `quitar_paradas_cortas`: una posesión (LOCAL o VISITA) de
    menos de `min_pd` fotogramas con una parada justo antes y justo después
    pasa a ser parada. En los 5 min públicos (30 oct) 4 de las 6 posesiones
    predichas sin pareja eran rachas de 1–3 s dentro de una parada real con el
    balón visible. Los bordes del periodo no cuentan como parada. Con
    min_pd <= 1 devuelve la entrada tal cual.
    """
    pred = np.asarray(pred).copy()
    if min_pd <= 1:
        return pred
    limites = [0] + _cortes(periodos).tolist() + [len(pred)]
    for a, b in zip(limites, limites[1:]):
        seg = pred[a:b]
        cambios = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        ini = np.concatenate(([0], cambios))
        fin = np.concatenate((cambios, [len(seg)]))
        for k, (i, j) in enumerate(zip(ini, fin)):
            if (seg[i] in (LOCAL, VISITA) and j - i < min_pd and 0 < k < len(ini) - 1
                    and seg[ini[k - 1]] == NINGUNO and seg[ini[k + 1]] == NINGUNO):
                seg[i:j] = NINGUNO
        pred[a:b] = seg
    return pred


MIN_PD_GRID = [1, 50, 75, 100, 150]   # 0–6 s


def mantener_en_huecos(pred, visto, periodos, min_hueco: int, modo="equipo") -> np.ndarray:
    """
    En los huecos largos del balón (>= `min_hueco` fotogramas seguidos sin
    verlo, sin cruzar el cambio de periodo) la etiqueta anterior al hueco se
    mantiene dentro de él. Solo actúa si esa etiqueta es una posesión (LOCAL o
    VISITA) y el hueco no abre el periodo.
      "todo":   todo el hueco toma la etiqueta anterior. Es exactamente lo que
                hace el evaluador oficial con una X (la rellena hacia delante
                en spellF1), pero sin perder los fotogramas en macroF1.
      "equipo": solo los fotogramas del hueco predichos como el OTRO equipo;
                las paradas (NINGUNO) predichas dentro del hueco se respetan.
      "d":      solo los fotogramas del hueco predichos como NINGUNO (apuesta a
                que un hueco en juego no es una parada; pierde las paradas reales).
    `visto`: booleano por fotograma, True si el balón estaba en los datos.
    """
    pred = np.asarray(pred).copy()
    visto = np.asarray(visto, dtype=bool)
    per = np.asarray(periodos)
    n, i = len(pred), 0
    while i < n:
        if visto[i]:
            i += 1
            continue
        j = i
        while j < n and not visto[j] and per[j] == per[i]:
            j += 1
        if j - i >= min_hueco and i > 0 and per[i - 1] == per[i] and pred[i - 1] in (LOCAL, VISITA):
            ant, seg = pred[i - 1], pred[i:j]
            if modo == "todo":
                seg[:] = ant
            elif modo == "equipo":
                seg[np.isin(seg, (LOCAL, VISITA)) & (seg != ant)] = ant
            elif modo == "d":
                seg[seg == NINGUNO] = ant
            else:
                raise ValueError(f"modo desconocido: {modo}")
        i = j
    return pred


HUECO_GRID = [25, 50, 125, 250]


def resumen_rachas(pred, periodos) -> str:
    """Cuántas posesiones y de qué duración, para leer de un vistazo."""
    r = pm.possession_stats(_con_cortes(a_codigos(pred), periodos),
                            pm.valid_runs(_con_cortes(np.zeros(len(pred), dtype=np.int8), periodos)))
    return f"{r['n_spells']} posesiones, media {r['mean_duration_s']:.1f} s, cuota local {r['home_share']:.1%}"
