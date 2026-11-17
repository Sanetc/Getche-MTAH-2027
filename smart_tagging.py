"""
Smart Tagging como ancla de saques
==================================

El dato oficial trae las fases automáticas de Metrica (SportsCode XML). Las
`SET PIECES` (59 en el partido: 26 saques de banda, 12 córners, 10 saques de
puerta, 5 faltas, 6 saques de centro) dicen que ahí hubo un reinicio y, salvo
en los saques de banda (`Team` = N/A), de qué equipo. Es evidencia automática:
hay falsos positivos, falta más de la mitad de los reinicios reales (un partido
tiene ~100), y los tiempos son fases, no instantes.

Los sample data no tienen Smart Tagging, así que se simula desde los eventos
anotados (`simular_set_pieces`): se conserva cada SET PIECE con un recall por
tipo calibrado con los recuentos oficiales, se le quita el equipo a los saques
de banda, se le pone una ventana [inicio − 1..5 s, inicio + 5..15 s] y se
añaden fases falsas. La duración real de las fases oficiales no se conoce:
por eso el ancla no depende de ella.

Dos usos:
  - `features_smart_tagging`: columnas por fotograma para el modelo (dentro de
    una fase de qué equipo, segundos desde su inicio, tipo). Riesgo: el modelo
    aprende los tiempos de la simulación, no los oficiales.
  - `anclar_saques`: post-procesado. Dentro de la ventana de una fase con
    equipo T, el primer reinicio que predice el modelo (D → equipo) pasa a ser
    de T si el modelo decía el otro. No usa la duración de la fase ni cambia
    dónde está el reinicio: solo de quién es.
"""

import numpy as np
import pandas as pd

from datos import EQUIPO, FPS, LOCAL, NINGUNO, SIN_EQUIPO, VISITA

# Recall por tipo, aproximado desde los 59 SET PIECES oficiales frente a los
# ~100 reinicios de un partido (saques de banda ~40, faltas ~30, ...).
RECALL = {"THROW IN": 0.7, "CORNER KICK": 0.9, "GOAL KICK": 0.6, "FREE KICK": 0.15,
          "KICK OFF": 1.0, "PENALTY": 0.9}
TIPOS = list(RECALL)


def tipo_de(subtipo) -> str | None:
    """'FREE KICK,INDIRECT' → 'FREE KICK'; 'KICK OFF-RETAKEN' → 'KICK OFF'."""
    s = str(subtipo).upper()
    return next((t for t in TIPOS if s.startswith(t)), None)


def simular_set_pieces(ev: pd.DataFrame, gt: np.ndarray, semilla=0, recall=RECALL,
                       antes_s=(1.0, 5.0), despues_s=(5.0, 15.0), falsos=5,
                       p_equipo_mal=0.05, primer_frame=1) -> pd.DataFrame:
    """
    Fases SET PIECES simuladas: DataFrame con tipo, equipo (LOCAL / VISITA /
    SIN_EQUIPO en los saques de banda), inicio y fin (índices de fila del
    tracking, fin exclusivo; `primer_frame` es el Frame de la fila 0).
    """
    rng = np.random.default_rng(semilla)
    filas = []
    sp = ev[ev["Type"] == "SET PIECE"]
    for subtipo, team, frame in sp[["Subtype", "Team", "Start Frame"]].itertuples(index=False):
        tipo = tipo_de(subtipo)
        if tipo is None or rng.random() > recall.get(tipo, 0.5):
            continue
        equipo = EQUIPO.get(team, SIN_EQUIPO)
        if tipo == "THROW IN":
            equipo = SIN_EQUIPO
        elif equipo != SIN_EQUIPO and rng.random() < p_equipo_mal:
            equipo = VISITA if equipo == LOCAL else LOCAL
        f = int(frame) - primer_frame
        filas.append(dict(tipo=tipo, equipo=equipo,
                          inicio=f - int(rng.uniform(*antes_s) * FPS),
                          fin=f + int(rng.uniform(*despues_s) * FPS), falso=False))
    en_juego = np.flatnonzero(np.isin(gt, (LOCAL, VISITA)))
    for _ in range(falsos):                             # fases que no corresponden a ningún reinicio
        f = int(rng.choice(en_juego))
        tipo = str(rng.choice(TIPOS))
        equipo = SIN_EQUIPO if tipo == "THROW IN" else int(rng.choice((LOCAL, VISITA)))
        filas.append(dict(tipo=tipo, equipo=equipo,
                          inicio=f - int(rng.uniform(*antes_s) * FPS),
                          fin=f + int(rng.uniform(*despues_s) * FPS), falso=True))
    out = pd.DataFrame(filas).sort_values("inicio").reset_index(drop=True)
    out["inicio"] = out["inicio"].clip(lower=0)
    return out


def features_smart_tagging(fases: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Por fotograma: st_local / st_visita / st_na (1 dentro de una fase de ese
    equipo), st_desde (segundos desde el inicio de la fase, 30 fuera),
    st_hasta (segundos hasta el fin, 30 fuera) y st_<tipo> (1 dentro de una
    fase de ese tipo).
    """
    F = pd.DataFrame(0.0, index=range(n), columns=["st_local", "st_visita", "st_na"]
                     + [f"st_{t.lower().replace(' ', '_')}" for t in TIPOS])
    F["st_desde"], F["st_hasta"] = 30.0, 30.0
    idx = np.arange(n)
    for fa in fases.itertuples(index=False):
        a, b = max(int(fa.inicio), 0), min(int(fa.fin), n)
        if b <= a:
            continue
        col = {LOCAL: "st_local", VISITA: "st_visita", SIN_EQUIPO: "st_na"}[int(fa.equipo)]
        F.loc[a:b - 1, col] = 1.0
        F.loc[a:b - 1, f"st_{fa.tipo.lower().replace(' ', '_')}"] = 1.0
        F.loc[a:b - 1, "st_desde"] = np.minimum((idx[a:b] - a) / FPS, 30.0)
        F.loc[a:b - 1, "st_hasta"] = np.minimum((b - idx[a:b]) / FPS, 30.0)
    return F


def anclar_saques(pred, fases: pd.DataFrame, periodos, margen=FPS) -> np.ndarray:
    """
    Para cada fase con equipo conocido, el primer reinicio (NINGUNO → equipo)
    que el modelo predice dentro de [inicio − margen, fin + margen] se asigna a
    ese equipo: si el modelo decía el otro, se cambia la posesión entera que
    empieza ahí (hasta el siguiente cambio de etiqueta, sin cruzar el periodo).
    Las fases sin equipo (saques de banda) no se tocan.
    """
    pred = np.asarray(pred).copy()
    per = np.asarray(periodos)
    n = len(pred)
    cambios = np.flatnonzero(pred[1:] != pred[:-1]) + 1        # inicio de cada racha (salvo la primera)
    es_reinicio = np.isin(pred[cambios], (LOCAL, VISITA)) & (pred[cambios - 1] == NINGUNO)
    reinicios = cambios[es_reinicio]
    for fa in fases.itertuples(index=False):
        equipo = int(fa.equipo)
        if equipo == SIN_EQUIPO:
            continue
        a, b = max(int(fa.inicio) - margen, 1), min(int(fa.fin) + margen, n)
        dentro = reinicios[(reinicios >= a) & (reinicios < b)]
        if len(dentro) == 0:
            continue
        t = int(dentro[0])
        if pred[t] == equipo:
            continue
        fin = t + 1
        while fin < n and pred[fin] == pred[t] and per[fin] == per[t]:
            fin += 1
        pred[t:fin] = equipo
    return pred
