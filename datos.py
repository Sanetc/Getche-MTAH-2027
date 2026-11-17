"""
More Than a Hack 2027 — carga de datos, verdad aproximada y exportación
=======================================================================

Todo lo que depende del FORMATO de los datos vive aquí. Cuando lleguen los
datos oficiales (20 oct) solo debería hacer falta tocar este archivo:

  - `cargar_tracking()`  → DataFrame con una fila por fotograma y columnas
                           Frame, Period, ball_x, ball_y, H_<id>_x/_y, A_<id>_x/_y
                           en metros (105 x 68, origen al centro) y con el
                           equipo local atacando SIEMPRE hacia +x.
  - `cargar_eventos()`   → DataFrame con columnas Team ('Home'/'Away'), Type,
                           Subtype, Period, Start Frame, End Frame.
  - `exportar()`         → escribe el CSV de entrega (ver FORMATO_ENTREGA).

Formatos soportados hoy (los dos de los sample data de Metrica):
  - CSV "Raw" (Sample_Game_1 y 2): <juego>_RawTrackingData_{Home,Away}_Team.csv
    y <juego>_RawEventsData.csv.
  - EPTS (Sample_Game_3): <juego>_metadata.xml + <juego>_tracking.txt +
    <juego>_events.json. Se lee con kloppy.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

LARGO, ANCHO = 105.0, 68.0   # metros de la cancha
FPS = 25

NINGUNO, LOCAL, VISITA = 0, 1, 2
SIN_ETIQUETA = -1            # fotograma sin verdad conocida (se excluye al evaluar)
NOMBRES = {NINGUNO: "Ninguno", LOCAL: "Local", VISITA: "Visitante"}
EQUIPO = {"Home": LOCAL, "Away": VISITA}

# Formato del CSV de entrega, según SUBMISSION.md del repo oficial: cabecera
# exacta `frame,label`, una fila por fotograma, etiquetas H / A / D / X.
# ("periodo": None → no se escribe esa columna.)
FORMATO_ENTREGA = {
    "columnas": {"frame": "frame", "periodo": None, "posesion": "label"},
    "etiquetas": {LOCAL: "H", VISITA: "A", NINGUNO: "D", SIN_ETIQUETA: "X"},
    "n_frames": 199319,      # filas de la entrega oficial (frame 0…199318); None = las del tracking
}

# Partes evaluadas del BAR-ATH (fotogramas del video táctico, ambos incluidos;
# DATA.md / games.half_video_blocks). El ATD trae todo el partido como un solo
# periodo, así que las partes se asignan con esto. La 2ª arranca en 110.808
# pero el saque es en 113.742: esos fotogramas son balón muerto.
PARTES_OFICIALES = {1: (11185, 90651), 2: (110808, 190197)}
SAQUES_OFICIALES = {1: 11185, 2: 113742}


# ----------------------------------------------------------------------------
# 1. Detección de formato
# ----------------------------------------------------------------------------
def stem_oficial(carpeta: Path, juego: str):
    """
    Prefijo de los archivos oficiales en `carpeta` ("BAR-ATH_tactical" en
    data/tactical/BAR-ATH_tactical_atd.xml). `juego` puede ser ese prefijo o
    el nombre de la carpeta ("tactical"); None si no hay ATD oficial.
    """
    if (carpeta / f"{juego}_atd.xml").exists():
        return juego
    atds = sorted(carpeta.glob("*_atd.xml"))
    return atds[0].name[:-len("_atd.xml")] if atds else None


def detectar_formato(carpeta: Path, juego: str) -> str:
    """'csv' (sample Raw), 'epts' (sample Game 3) u 'oficial' (bundle MTAH 2027)."""
    if stem_oficial(carpeta, juego):
        return "oficial"
    if (carpeta / f"{juego}_metadata.xml").exists():
        return "epts"
    if (carpeta / f"{juego}_RawTrackingData_Home_Team.csv").exists():
        return "csv"
    raise FileNotFoundError(f"No reconozco el formato de {juego} en {carpeta}")


def archivos_oficiales(carpeta: Path, juego: str) -> dict:
    """Rutas del bundle oficial (las que existan): atd_meta, atd_raw, smart_tagging, homografia."""
    stem = stem_oficial(carpeta, juego)
    if stem is None:
        raise FileNotFoundError(f"No hay *_atd.xml en {carpeta}")
    rutas = dict(atd_meta=carpeta / f"{stem}_atd.xml", atd_raw=carpeta / f"{stem}_atd.txt",
                 smart_tagging=carpeta / f"{stem}_smart_tagging.xml",
                 homografia=carpeta / f"{stem}_homography.csv")
    return {k: v for k, v in rutas.items() if v.exists()}


# ----------------------------------------------------------------------------
# 2. Tracking
# ----------------------------------------------------------------------------
def _leer_csv_equipo(path: Path, prefijo: str) -> pd.DataFrame:
    """Lee un CSV Raw de Metrica (3 filas de encabezado) a columnas planas."""
    raw = pd.read_csv(path, header=None, skiprows=2, low_memory=False)
    encabezado = raw.iloc[0].tolist()
    datos = raw.iloc[1:].reset_index(drop=True)

    columnas = []
    for nombre in encabezado:
        if isinstance(nombre, str) and nombre.startswith("Player"):
            columnas.append(f"{prefijo}_{nombre}_x")
        elif nombre == "Ball":
            columnas.append("ball_x")
        elif isinstance(nombre, str):
            columnas.append(nombre)
        else:  # columna 'y' sin nombre que sigue a cada 'x'
            columnas.append(columnas[-1][:-2] + "_y")
    datos.columns = columnas
    return datos.apply(pd.to_numeric, errors="coerce")


def _tracking_csv(carpeta: Path, juego: str) -> pd.DataFrame:
    casa = _leer_csv_equipo(carpeta / f"{juego}_RawTrackingData_Home_Team.csv", "H")
    visita = _leer_csv_equipo(carpeta / f"{juego}_RawTrackingData_Away_Team.csv", "A")
    visita = visita.drop(columns=["Period", "Time [s]", "ball_x", "ball_y"])
    return casa.merge(visita, on="Frame")


def _tracking_epts(meta: Path, raw: Path, desfase_frame=0) -> pd.DataFrame:
    """
    EPTS de Metrica vía kloppy (sample Game 3 y el ATD oficial). `Frame` =
    frame_id + desfase_frame (oficial: −1, para que coincida con el fotograma
    del video y la fila de la entrega). Los fotogramas fuera de los periodos
    (antes del saque, descanso, después del final) llevan Period 0.
    """
    from kloppy import metrica

    ds = metrica.load_tracking_epts(
        meta_data=meta, raw_data=raw,
        coordinates="metrica",          # normalizado 0-1, origen arriba-izquierda, como el CSV
    )
    t = ds.to_df()
    prefijo = {}
    for equipo in ds.metadata.teams:
        p = "H" if str(equipo.ground) == "home" else "A"
        for jugador in equipo.players:
            prefijo[jugador.player_id] = p

    periodo = pd.to_numeric(t["period_id"], errors="coerce").fillna(0).astype(int)
    tiempo = t["timestamp"]
    if hasattr(tiempo, "dt"):
        tiempo = tiempo.dt.total_seconds()
    df = pd.DataFrame({
        "Frame": t["frame_id"].astype(int) + desfase_frame,
        "Period": periodo,
        "Time [s]": pd.to_numeric(tiempo, errors="coerce"),
        "ball_x": pd.to_numeric(t["ball_x"], errors="coerce"),
        "ball_y": pd.to_numeric(t["ball_y"], errors="coerce"),
    })
    for c in t.columns:
        if c.endswith("_x") and c[:-2] in prefijo:
            pid = c[:-2]
            df[f"{prefijo[pid]}_{pid}_x"] = pd.to_numeric(t[c], errors="coerce")
            df[f"{prefijo[pid]}_{pid}_y"] = pd.to_numeric(t[f"{pid}_y"], errors="coerce")
    return df


def columnas_jugadores(df: pd.DataFrame, prefijo: str) -> list:
    """Columnas _x de los jugadores de un equipo ('H_' o 'A_')."""
    return [c for c in df.columns if c.startswith(prefijo) and c.endswith("_x")]


def normalizar_direccion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hace que el equipo local ataque hacia +x en TODOS los periodos.

    Heurística sin identidades (el ATD oficial no las tiene): en cada fotograma
    el jugador local más alejado del centro es el portero; si en la mediana del
    periodo está más lejos por la derecha (x > 0), el local defiende la derecha
    y hay que voltear el periodo (x -> -x, y -> -y). Funciona con cualquier
    formato, con o sin identidades, y no depende de metadatos.
    """
    df = df.copy()
    cols_xy = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    for periodo, idx in df.groupby("Period").groups.items():
        if periodo == 0:                 # fuera de los periodos (descanso): no hay dirección
            continue
        X = df.loc[idx, columnas_jugadores(df, "H_")].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            profundidad = np.nanmax(X, 1) + np.nanmin(X, 1)   # > 0: el más lejano está a la derecha
        if np.nanmedian(profundidad) > 0:
            df.loc[idx, cols_xy] *= -1
    return df


def cargar_tracking(carpeta: Path, juego: str, normalizar=True) -> pd.DataFrame:
    """Tracking en metros, origen al centro, local atacando hacia +x."""
    formato = detectar_formato(carpeta, juego)
    if formato == "csv":
        df = _tracking_csv(carpeta, juego)
    elif formato == "epts":
        df = _tracking_epts(carpeta / f"{juego}_metadata.xml", carpeta / f"{juego}_tracking.txt")
    else:                                # oficial: frame_id 1…N → fotograma del video 0…N−1
        rutas = archivos_oficiales(carpeta, juego)
        df = _tracking_epts(rutas["atd_meta"], rutas["atd_raw"], desfase_frame=-1)
        periodo = np.zeros(len(df), dtype=int)
        for p, (ini, fin) in PARTES_OFICIALES.items():
            periodo[df["Frame"].between(ini, fin).to_numpy()] = p
        df["Period"] = periodo

    # Normalizado (0-1, origen arriba-izquierda) -> metros (origen al centro)
    for c in df.columns:
        if c.endswith("_x"):
            df[c] = (df[c] - 0.5) * LARGO
        elif c.endswith("_y"):
            df[c] = -(df[c] - 0.5) * ANCHO
    return normalizar_direccion(df) if normalizar else df


# ----------------------------------------------------------------------------
# 3. Eventos
# ----------------------------------------------------------------------------
COLUMNAS_EVENTOS = ["Team", "Type", "Subtype", "Period", "Start Frame", "End Frame"]


def _eventos_csv(carpeta: Path, juego: str) -> pd.DataFrame:
    ev = pd.read_csv(carpeta / f"{juego}_RawEventsData.csv")
    return ev[COLUMNAS_EVENTOS]


def _eventos_epts(carpeta: Path, juego: str) -> pd.DataFrame:
    raiz = ET.parse(carpeta / f"{juego}_metadata.xml").getroot()
    score = next(e for e in raiz.iter() if e.tag.endswith("Score"))
    id_local = score.attrib["idLocalTeam"]

    datos = json.load(open(carpeta / f"{juego}_events.json", encoding="utf-8"))["data"]

    def subtipo(s):
        if isinstance(s, dict):
            return s["name"]
        if isinstance(s, list):
            return ",".join(x["name"] for x in s)
        return np.nan

    filas = [{
        "Team": "Home" if e["team"]["id"] == id_local else "Away",
        "Type": e["type"]["name"],
        "Subtype": subtipo(e.get("subtypes")),
        "Period": e["period"],
        "Start Frame": e["start"]["frame"],
        "End Frame": e["end"]["frame"],
    } for e in datos]
    return pd.DataFrame(filas, columns=COLUMNAS_EVENTOS)


def tiene_eventos(carpeta: Path, juego: str) -> bool:
    """Los partidos de entrega pueden venir sin eventos."""
    return (carpeta / f"{juego}_RawEventsData.csv").exists() or (carpeta / f"{juego}_events.json").exists()


def cargar_eventos(carpeta: Path, juego: str) -> pd.DataFrame:
    formato = detectar_formato(carpeta, juego)
    if formato == "oficial":
        raise FileNotFoundError("El bundle oficial no trae eventos (solo Smart Tagging).")
    ev = _eventos_csv(carpeta, juego) if formato == "csv" else _eventos_epts(carpeta, juego)
    return ev.sort_values(["Start Frame", "End Frame"], kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3b. Smart Tagging y homografía (solo en el bundle oficial)
# ----------------------------------------------------------------------------
SIN_EQUIPO = -1                      # `Team` = N/A (todos los saques de banda oficiales)
TIPOS_SAQUE = {                      # tag `Type` oficial → tipo de smart_tagging.py
    "throw-in": "THROW IN", "corner kick": "CORNER KICK", "goal kick": "GOAL KICK",
    "free kick": "FREE KICK", "kick off (after goal)": "KICK OFF", "kick off (start)": "KICK OFF",
    "penalty": "PENALTY",
}


def tiene_smart_tagging(carpeta: Path, juego: str) -> bool:
    return detectar_formato(carpeta, juego) == "oficial" and "smart_tagging" in archivos_oficiales(carpeta, juego)


def cargar_smart_tagging(carpeta: Path, juego: str, local="barcelona", fps=FPS) -> pd.DataFrame:
    """
    Fases del XML SportsCode: una fila por instancia con codigo, tipo (solo en
    SET PIECES, en el vocabulario de smart_tagging.py), equipo (LOCAL si el
    tag Team contiene `local`, VISITA si es otro club, SIN_EQUIPO si N/A),
    half, inicio y fin en fotogramas del video (fin exclusivo; <start>/<end>
    son segundos desde el primer fotograma del video, verificado por la
    organización), y falso=False (columna que espera `anclar_saques`).
    """
    ruta = archivos_oficiales(carpeta, juego)["smart_tagging"]
    filas = []
    for inst in ET.parse(ruta).iter("instance"):
        etiquetas = {(lab.findtext("group") or "").strip(): (lab.findtext("text") or "").strip()
                     for lab in inst.findall("label")}
        team = etiquetas.get("Team", "N/A")
        if team.upper() in ("N/A", "", "NA"):
            equipo = SIN_EQUIPO
        else:
            equipo = LOCAL if local.lower() in team.lower() else VISITA
        codigo = (inst.findtext("code") or "").strip().upper()
        tipo = TIPOS_SAQUE.get(etiquetas.get("Type", "").lower()) if codigo == "SET PIECES" else None
        ini, fin = float(inst.findtext("start")), float(inst.findtext("end"))
        filas.append(dict(codigo=codigo, tipo=tipo, equipo=equipo, half=etiquetas.get("Half"),
                          inicio=int(round(ini * fps)), fin=int(round(fin * fps)), falso=False,
                          etiquetas=etiquetas))
    return pd.DataFrame(filas).sort_values("inicio").reset_index(drop=True)


def cargar_homografia(carpeta: Path, juego: str) -> pd.DataFrame:
    """
    CSV oficial: frame, h00…h22, reprojection_error, valid. Devuelve el
    DataFrame indexado por fotograma del video, con `valid` booleano; en los
    fotogramas no válidos la matriz es NaN. No lo usa el modelo: sirve para
    comprobar alineaciones y para la fase 2.
    """
    h = pd.read_csv(archivos_oficiales(carpeta, juego)["homografia"])
    h["valid"] = h["valid"].astype(bool)
    return h.set_index("frame")


def fuera_de_periodo(df: pd.DataFrame) -> np.ndarray:
    """Fotogramas sin periodo (antes del saque inicial, descanso, tras el final): balón muerto seguro."""
    return (df["Period"].to_numpy() == 0)


TIPO_SAQUE_INICIAL = "kick off (start)"     # tag `Type` de la fase SET PIECES del saque inicial de cada parte


def antes_del_saque(df: pd.DataFrame, fases: pd.DataFrame) -> np.ndarray:
    """
    Fotogramas de una parte anteriores al saque inicial de esa parte: balón
    muerto seguro. Regla general, sin números del partido: en cada parte, los
    fotogramas anteriores al INICIO de la fase `SET PIECES` de tipo
    `Kick Off (Start)` de esa parte (Smart Tagging entregado; tag `Half`) son
    D; desde ese inicio decide el modelo. Si una parte no tiene esa fase, no se
    fuerza nada. `fases`: salida de `cargar_smart_tagging`.

    En el BAR-ATH solo actúa en la 2ª parte, que se evalúa desde bastante antes
    del saque (los jugadores vuelven al campo y pelotean); en la 1ª la fase
    empieza antes del primer fotograma evaluado. SAQUES_OFICIALES no interviene:
    queda solo para las comprobaciones de `comprobar_oficial.py`.
    """
    fr, per = df["Frame"].to_numpy(), df["Period"].to_numpy()
    m = np.zeros(len(df), dtype=bool)
    if fases is None or not len(fases):
        return m
    tipo = fases["etiquetas"].apply(lambda e: str(e.get("Type", "")).strip().lower())
    saques = fases[(fases["codigo"] == "SET PIECES") & (tipo == TIPO_SAQUE_INICIAL)]
    for p in np.unique(per[per > 0]):
        de_la_parte = saques[saques["half"].astype(str).str.strip().str.startswith(str(int(p)))]
        if len(de_la_parte):
            m |= (per == p) & (fr < int(de_la_parte["inicio"].min()))
    return m


# ----------------------------------------------------------------------------
# 4. "Verdad aproximada" desde eventos (solo para practicar con sample data)
# ----------------------------------------------------------------------------
# El equipo que ejecuta estos eventos tiene el balón desde el inicio del evento.
TIPOS_CON_POSESION = {"PASS", "RECOVERY", "SET PIECE", "SHOT", "CARRY"}
# Subtipos que seguro paran el juego: desde el fin del evento, balón parado.
SUBTIPOS_PARADA = ("GOAL", "OUT", "WOODWORK", "OFFSIDE", "END HALF")


def _para_el_juego(tipo, subtipo) -> bool:
    s = str(subtipo).upper()
    if tipo == "FAULT RECEIVED" or (tipo == "CHALLENGE" and "FAULT" in s):
        return True                      # falta: el juego se para (la ventaja es rara)
    return tipo in ("SHOT", "BALL LOST") and any(p in s for p in SUBTIPOS_PARADA)


def _ambiguo_despues(tipo, subtipo) -> bool:
    """Tras el evento no se sabe quién tiene el balón hasta el siguiente."""
    return tipo == "SHOT" and not _para_el_juego(tipo, subtipo)   # parada del portero, rechace...


def verdad_desde_eventos(ev: pd.DataFrame, frames: pd.Series, definicion="oficial", max_toque=FPS) -> np.ndarray:
    """
    Etiqueta por fotograma a partir de eventos:
      - Desde el inicio de un PASS / CARRY / RECOVERY / SET PIECE / SHOT, el
        balón es del equipo que lo ejecuta.
      - Balón parado -> NINGUNO hasta el siguiente evento (normalmente el
        SET PIECE): tras el fin de un BALL OUT, de un tiro a gol / fuera / al
        palo, de un fuera de juego, de una falta y del final de la parte.
      - SIN_ETIQUETA (se excluye) antes del primer evento de cada periodo.

    definicion="oficial" (la del concurso, CHALLENGE.md): con el balón en juego
      siempre posee un equipo; en un balón suelto o disputado lo conserva el
      que lo tenía, y cambia cuando el rival lo controla (= su siguiente
      evento). Así, un BALL LOST o un tiro parado NO quitan la etiqueta: el
      equipo la conserva hasta el siguiente evento con equipo.
    definicion="eventos" (la que usábamos hasta el 12 oct): SIN_ETIQUETA
      durante un BALL LOST y tras un tiro parado o rechazado, hasta el
      siguiente evento. Sirve para comparar con los resultados antiguos.
    definicion="control": la oficial, y además un toque del rival sin control
      (racha de menos de `max_toque` fotogramas sin pase, tiro ni saque,
      ver `_exigir_control`) no cambia la posesión.
    """
    n_max = int(frames.max()) + 2
    gt = np.full(n_max, SIN_ETIQUETA)
    oficial = definicion in ("oficial", "control")

    marcas = []
    for e in ev.itertuples(index=False):
        ini, fin = int(e[4]), int(e[5])
        tipo, equipo = e.Type, EQUIPO.get(e.Team)
        if fin <= 0:
            fin = ini
        if tipo in TIPOS_CON_POSESION and equipo is not None:
            marcas.append((ini, equipo))
        elif tipo == "BALL LOST" and not oficial:
            marcas.append((ini, SIN_ETIQUETA))
        elif tipo == "BALL OUT":
            marcas.append((fin, NINGUNO))
        if _para_el_juego(tipo, e.Subtype):
            marcas.append((fin, NINGUNO))
        elif _ambiguo_despues(tipo, e.Subtype) and not oficial:
            marcas.append((fin, SIN_ETIQUETA))
    marcas.sort(key=lambda m: m[0])

    for (ini, etiqueta), (sig, _) in zip(marcas, marcas[1:] + [(n_max, None)]):
        gt[ini:sig] = etiqueta
    if definicion == "control":
        gt = _exigir_control(gt, ev, max_toque)
    return gt[frames.astype(int).to_numpy()]


# Sin CARRY: el EPTS de Metrica (Sample_Game_3) anota un CARRY tras casi cada
# RECOVERY aunque dure un instante, y los CSV (Games 1 y 2) no tienen CARRY; una
# conducción real dura más de `max_toque` y ya cuenta por duración.
ACCIONES_CON_CONTROL = {"PASS", "SHOT", "SET PIECE"}


def _exigir_control(gt: np.ndarray, ev: pd.DataFrame, max_toque: int) -> np.ndarray:
    """
    Regla oficial "la posesión cambia cuando el rival tiene el balón BAJO
    CONTROL": una racha de un equipo más corta que `max_toque` fotogramas, que
    llega justo después de una posesión del rival y en la que ese equipo no
    hace ningún pase, tiro ni saque (típicamente RECOVERY + BALL
    LOST o BALL OUT: un toque o despeje sin controlar), no es posesión: el
    equipo anterior la conserva hasta el siguiente cambio o la parada.
    """
    gt = gt.copy()
    c = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    ini, fin = np.r_[0, c], np.r_[c, len(gt)]
    acciones = {LOCAL: [], VISITA: []}
    for e in ev.itertuples(index=False):
        equipo = EQUIPO.get(e.Team)
        if equipo is not None and e.Type in ACCIONES_CON_CONTROL:
            acciones[equipo].append(int(e[4]))
    acciones = {k: np.sort(v) for k, v in acciones.items()}
    for k in range(1, len(ini)):
        equipo, antes = gt[ini[k]], gt[ini[k] - 1]
        if equipo not in (LOCAL, VISITA) or antes not in (LOCAL, VISITA) or antes == equipo:
            continue
        if fin[k] - ini[k] >= max_toque:
            continue
        a = acciones[equipo]
        if np.any((a >= ini[k]) & (a < fin[k])):
            continue
        gt[ini[k]:fin[k]] = antes
    return gt


# ----------------------------------------------------------------------------
# 5. Evaluación
# ----------------------------------------------------------------------------
def exactitud(pred, gt) -> float:
    """Fracción de fotogramas etiquetados en los que pred == gt."""
    m = gt >= 0
    return float((pred[m] == gt[m]).mean())


def matriz_confusion(pred, gt) -> pd.DataFrame:
    """Filas = verdad, columnas = predicción, en fracción por fila."""
    m = gt >= 0
    tabla = pd.crosstab(
        pd.Series(gt[m]).map(NOMBRES), pd.Series(pred[m]).map(NOMBRES),
        rownames=["verdad"], colnames=["pred"], normalize="index",
    )
    return tabla.reindex(index=NOMBRES.values(), columns=NOMBRES.values(), fill_value=0)


# ----------------------------------------------------------------------------
# 6. Exportación
# ----------------------------------------------------------------------------
def exportar(ruta: Path, frames, periodos, pred, probs=None, verdad=None, n_frames=None) -> Path:
    """
    Escribe el CSV de entrega según FORMATO_ENTREGA.
    probs: opcional, array (T, 3) en el orden NINGUNO, LOCAL, VISITA; se
           escriben como columnas extra (NO para la entrega oficial).
    verdad: opcional, columna extra `verdad_aprox` (NO para la entrega oficial).
    n_frames: si se da, la salida tiene exactamente las filas 0…n_frames−1 en
           orden; los fotogramas sin tracking salen como balón muerto (D).
    """
    col = FORMATO_ENTREGA["columnas"]
    etq = FORMATO_ENTREGA["etiquetas"]
    out = pd.DataFrame({col["frame"]: np.asarray(frames).astype(int)})
    if col.get("periodo"):
        out[col["periodo"]] = np.asarray(periodos).astype(int)
    out[col["posesion"]] = [etq[x] for x in pred]
    if probs is not None:
        for k, nombre in NOMBRES.items():
            out[f"prob_{nombre.lower()}"] = np.round(probs[:, k], 4)
    if verdad is not None:
        out["verdad_aprox"] = [etq.get(x, "X") for x in verdad]
    if n_frames is not None:
        out = (out.drop_duplicates(col["frame"]).set_index(col["frame"])
                  .reindex(range(n_frames)).rename_axis(col["frame"]).reset_index())
        out[col["posesion"]] = out[col["posesion"]].fillna(etq[NINGUNO])
    ruta.parent.mkdir(parents=True, exist_ok=True)
    # Fin de línea fijo (CRLF, el que pandas escribe en Windows, donde se generó la
    # entrega): así el archivo es idéntico byte a byte en Windows, Linux y macOS.
    out.to_csv(ruta, index=False, lineterminator="\r\n")
    return ruta
