# CLAUDE.md — More Than a Hack 2027 (MTAH 2027)

Contexto del proyecto para Claude Code. Léelo completo antes de trabajar.

## 1. El concurso

**More Than a Hack 2027** es un hackathon de analítica de fútbol organizado por
Mobile World Capital Barcelona, Barça Innovation Hub (BIHUB) y Metrica Sports,
con datos reales de partidos del FC Barcelona. Segunda edición (la primera tuvo
más de 3,200 inscritos de 95+ países).

- Landing / registro: https://landing.mobileworldcapital.com/more-than-a-hack-2027
- Blog de Metrica: https://www.metrica-sports.com/es/blog/more-than-a-hack-2027-build-something-real-with-fc-barcelona-match-data
- Contacto: hackathon.talentarena@mobileworldcapital.com

### Calendario
| Fase | Fechas | Modalidad |
|---|---|---|
| Registro | 1 oct – 10 nov 2026 | online |
| **Fase 1 – Initial Challenge** | **20 oct – 17 nov 2026** | online, ranking automático |
| Fase 2 – Innovation Challenge (top 20) | 26 nov 2026 – 14 ene 2027 | online, retos abiertos, 10 partidos |
| Final (top 5) | 2 mar 2027 | presencial, Talent Arena Barcelona |

### Reglas clave
- Equipos de **3 a 5 personas**. Con menos de 3 no se puede pasar a fase 2 ni a la final. (El equipo ya está formado: **Getche** → `Getche.csv` y `Getche.md`/`.pdf`.)
- Premios para finalistas: viaje y hospedaje a Barcelona, camisetas oficiales FCB 26/27, merch BIHUB, experiencia exclusiva con el FCB.

### Repositorio oficial de la fase 1 (leído el 12 oct 2026)
https://github.com/metrica-sports/more-than-a-hack-2027 — clonado en
`../datos/mtah-2027-oficial`. **Es la autoridad**: CHALLENGE.md, DATA.md,
EVALUATION.md, SUBMISSION.md, RULES.md, FAQ.md, OVERVIEW.md. Contiene el
evaluador oficial (`evaluation/run_evaluation.py`, `possession_metrics.py`),
`validate_submission.py`, la verdad pública de 5 min y cuatro baselines.
Los datos del partido llegan el 20 oct tras inscribirse y aceptar los términos.
**La organización actualizó la documentación el 22–23 oct** (commits c073f29 y f8f2629, solo
`.md`; el clon local se quedó en 3de6b60 hasta el 16 nov): antes de citar reglas, leer la
versión remota (`git -C ../datos/mtah-2027-oficial show origin/HEAD:RULES.md`).

### Datos oficiales (un solo partido)
**FC Barcelona (H) vs Athletic Club (A)**, LaLiga 2024-25, jornada 2. Dos grabaciones:
- **Táctica (la referencia):** 199.319 fotogramas a 25 fps. Fila *i* de la entrega
  = fotograma *i* de este video. ATD en EPTS (`BAR-ATH_tactical_atd.xml` + `.txt`,
  `frame_id` 1…199319 → fotograma = frame_id − 1), homografía CSV (válida al 100%),
  Smart Tagging en XML SportsCode (638 fases; `<start>/<end>` en segundos desde el
  inicio del video, fotograma = round(s·25)).
- **Broadcast:** otra línea de tiempo (164.488 fotogramas, cortes y repeticiones),
  sin mapeo a la táctica. Opcional.
- **ATD sin identidades:** 46 "track slots" para 22 jugadores, con equipo pero sin
  jugador; los slots aparecen, desaparecen y se intercambian. Coordenadas 0–1
  (campo 105,3 × 68). **El balón falta en el 41% de los fotogramas evaluados,
  y en juego** (cajas llenas, duelos aéreos, entradas), no solo en paradas.
- **Smart Tagging:** fases (Build Up, Progression, Set Pieces con tipo, Long Ball,
  Goals…) con `Team` (Barcelona / Atletic Club / **N/A en todos los saques de
  banda**) y `Half`. Automático, con falsos positivos; son fases, no posesiones.
- Fotogramas evaluados: 1ª parte 11.185–90.651, 2ª parte 110.808–190.197
  (158.857). La 2ª parte se evalúa desde 110.808 pero el saque es en 113.742:
  esos 2.934 son `D`.

## 2. Fase 1: la tarea (especificación oficial)

**Una etiqueta por fotograma: `H` / `A` / `D` / `X`.** `D` = balón muerto (no en
juego). `X` = sin respuesta; fuera de los fotogramas evaluados no cuesta nada.

**Definición de posesión (la verdad sale del tracking corregido, no de eventos
ni de video):** con el balón en juego siempre posee un equipo (no hay neutro);
en balón suelto / 50-50 la conserva el que la tenía; cada saque tiene dueño
claro; la posesión cambia cuando un rival tiene el balón **bajo control**. No
cambian la posesión: un toque sin control, un pase desviado sin interceptar, un
duelo aéreo ganado sin controlar después. Nadie pide quién "posee" un balón
muerto: es `D`. Referencia: los 5 min de verdad pública (17 posesiones, media
14,2 s; fotogramas 11.185–18.684).

**Métrica de ranking: `spellF1`.** Cada posesión (racha máxima de `H` o de `A`;
`D` corta) se empareja 1 a 1 con las reales por IoU temporal, F1 promediado en
umbrales 0,3–0,7. Un fotograma de retraso es gratis; 1 s de retraso baja a
0,65; un parpadeo que parte una posesión en dos cuesta mucho. `macroF1` por
fotograma (media de F1 de H, A, D) desempata. Floor publicado:
`baseline_smooth50` spellF1 0,2433 (macroF1 0,67) en el partido completo.
Diferencias < 0,04 en spellF1 son ruido. **No ajustar nada contra los 5 min
públicos** (elige mal la ventana de suavizado); sirven para validar el pipeline.

**`X` en spellF1 = "repite mi última etiqueta de esa parte"**: gratis dentro
de una posesión (mejor que un parpadeo), caro en un cambio. En macroF1 cuenta
como fallo.

**Entrega (formulario Google, enlace por correo, antes del 17 nov):**
`<equipo>.csv` (`frame,label`, 199.319 filas, frame 0…199318 en orden, sin
celdas vacías; pasar `validate_submission.py` al archivo exacto), **repositorio
público** que regenere el mismo archivo (sin datos dentro), y documento técnico
corto (`<equipo>.pdf` o `.md`: enfoque, qué datos se usaron y cuáles se
descartaron, reglas de posesión, limitaciones, reproducción). Una respuesta por
equipo, la última cuenta. No hay leaderboard ni feedback antes del cierre.

**Reglas actualizadas el 22–23 oct:** (1) los 5 min públicos "are not training data and not a
tuning set" y no sirven para elegir entre dos versiones ("That is a rule, not just advice"):
solo para comprobar alineación, formato y significado de las etiquetas; (2) el documento técnico
debe traer una sección de **datos de entrenamiento** (cada dataset etiquetado, fuente, quién
etiquetó, cuánto; nada del partido entregado); (3) prohibido cualquier dato producido por
humanos sobre el partido (eventos, tracking corregido) y convertir lo que se vio en "rules and
constants tied to specific frames of the game"; (4) permitido: aprendizaje sin etiquetas y
pseudo-etiquetas de algoritmo sobre el partido, y mirar video y datos para depurar.

**Reglas:** cualquier método inspeccionable y explicado; se puede mejorar o
sustituir el tracking; prohibido etiquetar a mano, usar la verdad privada o
entregar algo que el repo no regenere. **Prohibido subir los datos del partido
(o extractos) a LLMs en la nube** → en este proyecto Claude Code NO debe leer
los archivos oficiales de datos; solo código y estadísticas agregadas.

**Fase 2 (Innovation Challenge, 26 nov – 14 ene, top 20):** retos de diseño
abierto sobre 10 partidos con video, ATD y Smart Tagging, mentoría técnica,
jurado profesional elige 5 finalistas para el pitch del 2 mar 2027. En 2026 se
valoró originalidad, solidez técnica, uso de LLMs, relevancia futbolística y
calidad de presentación. Ganador 2026 (IpurdIArs): tracking + trayectoria del
balón + audio contextual + interfaz conversacional, una herramienta táctica
interactiva. Otros finalistas: asistentes tácticos con IA, detección automática
de eventos desde tracking, visualización del comportamiento colectivo.

## 3. Estado actual del código

Trabajo de preparación hecho con los **datos de muestra públicos de Metrica**
(https://github.com/metrica-sports/sample-data). Son partidos anónimos
("Home"/"Away", "Player1…"), NO del Barça ni del concurso. Se usan los 3:
Games 1 y 2 en CSV "Raw" y Game 3 en formato EPTS (metadata.xml +
tracking.txt + events.json, el formato de exportación actual de Metrica, que
es el más probable para los datos oficiales).

### Archivos
- `datos.py` — **todo lo que depende del formato de datos**: carga de tracking
  (CSV Raw, EPTS de muestra o bundle oficial `*_atd.xml` vía kloppy, detección
  automática), eventos, Smart Tagging XML y homografía CSV oficiales,
  normalización de la dirección de ataque (sin identidades), verdad aproximada,
  métricas y `exportar()` con `FORMATO_ENTREGA` (199.319 filas, `D` fuera de
  los periodos). Al llegar los datos oficiales solo debería hacer falta tocar
  este archivo.
- `baseline_posesion.py` — Niveles 1 y 2: reglas + histéresis, `buscar_parametros()`,
  `listar_juegos()` / `cargar_partido()` que usan los demás scripts.
- `modelo_posesion.py` — Nivel 3: LightGBM (~60 features) + Viterbi + fusión de
  posesiones cortas, ajustado por spellF1; validación leave-one-out entre
  partidos; `--enmascarar 0.41 --balon {ninguno,interpolar,estimar}` simula el
  balón ausente oficial (estimador con cross-fitting en los partidos de entreno).
- `evaluacion.py` — envuelve `oficial/possession_metrics.py` (copia MIT del
  evaluador del concurso, commit b429693): `evaluar()`, `spell_f1()`,
  `fusionar_rachas_cortas()`. Inserta un X entre periodos porque los sample
  data no tienen hueco de fotogramas en el descanso.
- `balon_estimado.py` — `enmascarar_balon()` (oculta el balón en juego con
  huecos log-normales) y estimador del balón ausente: features colectivas +
  historia del balón → LightGBM que corrige la interpolación lineal.
- `tracking_sin_id.py` — `quitar_identidades()`: simula los track slots del
  ATD oficial (cortes, huecos, duplicados, intercambios) sobre los sample data.
- `smart_tagging.py` — simula las fases `SET PIECES` del Smart Tagging desde
  los eventos (recall por tipo, saques de banda sin equipo, falsos positivos)
  y las usa como ancla de saques (`anclar_saques`) o features.
- `entregar.py` — `entrenar` (modelo final con todos los partidos, opcional
  `--enmascarar --balon estimar --sin-identidades`, → `modelos/posesion.pkl`, fuera de git) y
  `predecir` (carpeta de partidos → `<juego>.csv` con `frame,label` H/A/D/X).
- `analiza_errores.py` — clasifica los errores del modelo por causa y mide rachas.
- `comprobar_oficial.py` — guion del día de los datos: imprime solo agregados
  del bundle (periodos frente a los oficiales, slots visibles, balón ausente
  en los tramos evaluados, Smart Tagging por tipo, homografía válida) y con
  `--entrega` pasa `validate_submission.py` y `run_evaluation.py` en los 5 min
  públicos junto a los baselines. Es la forma de verificar el bundle real sin
  que Claude lea los datos.
- `calibrar_ruido.py` — elige los tres parámetros de `balon_estimado.ruido_balon` (ruido de
  detección del balón para los partidos de entrenamiento) comparando cuantiles sin etiquetas
  entre los sample data y el ATD oficial; solo imprime agregados. `reproducir.*` pasa
  `--ruido-balon 0.8 30 0.06` desde el 17 nov (ver la revisión del 16–17 nov más abajo).
- `setup.ps1` (Windows) / `setup.sh` (Linux/macOS) — instalan dependencias y
  descargan los datos de muestra a `../datos`.
- `reproducir.ps1` / `reproducir.sh` — regeneran la entrega completa
  (entrenar → predecir → `validate_submission.py` → SHA-256). Determinista:
  LightGBM con `random_state=0, deterministic=True, force_col_wise=True` y
  semillas fijas en las simulaciones (comprobado el 13 oct: dos entrenamientos
  completos seguidos → árboles idénticos y CSV con el mismo SHA-256; con 1 o
  4 hilos igual). Dos entrenamientos en paralelo agotan los 16 GB de RAM.
- `Getche.md` — documento técnico de la entrega (24 oct), con el SHA-256 de
  `Getche.csv`: si cambia el modelo, actualizar SHA, parámetros y números.
- `requirements.txt`, `README.md`.

### Pipeline
1. `datos.cargar_tracking()` → DataFrame con `Frame, Period, ball_x, ball_y,
   H_<id>_x/_y, A_<id>_x/_y` en metros (105 × 68, origen al centro) **con el
   local atacando siempre hacia +x** (se voltea cada periodo según dónde está
   el portero local). Así `ball_x` y `vx_balon` significan lo mismo en todos
   los periodos y partidos.
2. `datos.cargar_eventos()` → DataFrame unificado `Team, Type, Subtype, Period,
   Start Frame, End Frame`.
3. `datos.verdad_desde_eventos()` → etiqueta por fotograma: equipo del último
   PASS / CARRY / RECOVERY / SET PIECE / SHOT; balón parado (tras BALL OUT,
   tiro a gol/fuera/palo, falta —FAULT RECEIVED o CHALLENGE con FAULT—,
   fuera de juego, final de parte) → Ninguno hasta el siguiente evento. Con
   `definicion="oficial"` (por defecto desde el 12 oct) un BALL LOST o un tiro
   parado NO quitan la etiqueta: el equipo la conserva hasta el siguiente
   evento con equipo (regla oficial del balón suelto). Lecciones: etiquetar las paradas
   como Ninguno fue decisivo (antes llevaban el equipo del último toque y el
   modelo acertaba a cara o cruz en ellas); las faltas en disputa deben
   seguir siendo parada, porque son la mayoría de los ejemplos de "balón
   visible + jugadores quietos = Ninguno" y sin ellos el modelo falla en
   paradas largas (lesiones).
4. `baseline.calcular_features()` → distancia mínima balón–jugador por equipo,
   velocidad del balón, balón fuera.
5. `baseline.predecir_posesion()` → regla: jugador más cercano a ≤ radio;
   histéresis de N frames de control rival para cambiar (los frames sin
   control de nadie NO reinician la cuenta); balón rápido = nadie lo controla.
6. `modelo_posesion.py`: features ricas (3 más cercanos por equipo, densidad a
   3/5/10 m, dinámica del balón, ventanas temporales pasadas y futuras de
   5/15/50 frames que no cruzan el medio tiempo, velocidad media de los 22
   jugadores y frames sin ver el balón como detectores de balón parado,
   salida de la regla) →
   LightGBM 3 clases → Viterbi. En cada fold, los parámetros de la regla se
   ajustan con los partidos de entrenamiento y `p_quedarse` con predicciones
   fuera de muestra entre ellos.

### Resultados (métricas oficiales, verdad con definición oficial, 12 oct 2026)
Ver tablas en `README.md`. Leave-one-out, promedio de 3 partidos, con las
features colectivas (por defecto): regla spellF1 0.705, LightGBM 0.729
(parpadea), +Viterbi 0.870, +fusión **0.873** (0.863 sin colectivas).
Con el balón oculto el 41% en juego: 0.649 sin rellenar, 0.692 interpolando,
0.667 con el estimador de balón (0.662 con cross-fitting, 0.672 suavizado:
todo dentro del ruido y por debajo de interpolar; el balón estimado salta
entre fotogramas y LightGBM parpadea más), **0.761 con interpolación +
features colectivas en el modelo**. Lección: el modelo no necesita el balón
reconstruido, necesita ver el movimiento de los equipos. Estimador de balón
solo: 5.1 m de error medio vs 6.1 interpolación; gana a partir de 3 s.
**Por defecto: colectivas + interpolación.**
**X en huecos largos, probada el 13 oct y descartada** (`mantener_en_huecos()`,
`--huecos`): el evaluador rellena la X con la etiqueta anterior, así que
equivale a mantener la etiqueta en el hueco; empeora en todas las variantes
(balón oculto: 0.761 → 0.737–0.750 manteniendo solo cambios de equipo, 0.69–0.71
pisando también los D) porque el modelo ya sitúa bien los cambios que ocurren
sin ver el balón y mantener los retrasa. X solo fuera de los tramos evaluados.
**Tracking sin identidades, simulado el 13 oct** (`tracking_sin_id.py`,
`--sin-identidades`: 46 slots, tracklets ~20 s, huecos, duplicados,
intercambios; 20,7 visibles de media como el oficial): cuesta 0.873 → 0.857
sin ocultar el balón y 0.760 → 0.750 con el balón oculto (ruido). Para ello
la dirección de ataque ya no usa la identidad del portero (máximo/mínimo por
fotograma, mediana del periodo) y las velocidades por jugador cortan la pista
en saltos > 1 m (`balon_estimado.velocidades`). **Condición oficial completa
(sin identidades + balón oculto 41%): 0.750.** Smart Tagging simulado
(`smart_tagging.py`, `--smart-tagging`): ancla 0.746, features 0.752,
ambos 0.748: no aporta, el modelo ya acierta el equipo de los saques; repetir
con las fases reales el 20 oct. Con el código actual (velocidades robustas
a saltos) la configuración por defecto con identidades da 0.869 (antes
0.873, ruido). Modelo híbrido (`--hibrido`, especialista con balón donde se
ve + especialista sin features del balón donde no): 0.750 → 0.661, peor; el
modelo único ya reparte por `balon_nan` / `frames_sin_balon`. Solo en
huecos > 1 s (`--hibrido-hueco 25`): 0.669; > 3 s: 0.684. Cuanto más
especialista, peor: descartado. Los números de exactitud
anteriores (98,6%) ya no valen: la verdad cambió (balón suelto conserva el
equipo) y la métrica también.

### Advertencias
- La "verdad" es aproximada (sale de eventos), no son etiquetas reales de
  posesión. Usarla para comparar versiones, no como número absoluto.
- En los sample data el balón no está trackeado (NaN) en 28–36% de los
  fotogramas, pero casi todos son paradas del juego (gol, falta, fuera →
  siguiente saque), no balón perdido en juego. Con la verdad corregida, esos
  fotogramas se aciertan al 97–100%. Falta saber cómo los etiqueta la
  organización y si el ATD oficial pierde el balón en juego.
- Las ventanas "futuras" usan información posterior al fotograma: válido solo
  si la tarea es offline sobre el partido completo (lo esperado). Confirmar
  con las reglas oficiales.
- Entorno: `lightgbm` 4.7.0 da "access violation" en Windows; `requirements.txt`
  lo fija a < 4.7. Hace falta `scikit-learn` para la API sklearn de lightgbm.
  Smart App Control de Windows bloquea desde el 12 oct un `.pyd` de pandas
  3.0.6 ("directiva de Control de aplicaciones"); `requirements.txt` excluye
  esa versión, 3.0.5 funciona.
- `python -I` no agrega el directorio del script al path; los scripts lo
  insertan a mano con `sys.path.insert`.

### Bundle oficial (recibido el 22 oct, en `../datos/oficiales/data/{tactical,broadcast}`)
Agregados de `comprobar_oficial.py --data ../datos/oficiales/data --juegos tactical`:
199.319 fotogramas consecutivos; el ATD trae **un solo periodo** → `datos.py`
asigna las partes con `PARTES_OFICIALES`. 46 slots (24 H + 22 A), 19,0 visibles
de media. Balón ausente 40,7% en lo evaluado; huecos en juego: mediana 10,7 s,
p90 33,8 s, máx 221 s. Homografía 100%. Smart Tagging: 638 fases, 59 SET
PIECES, todas de **35 s fijos** (saque de centro 5 s): solo vale el inicio; el
primer KICK OFF de cada parte va 88 fotogramas antes del saque oficial.
`reproducir.ps1/.sh` predicen solo `tactical`. Primera entrega (22 oct,
modelo por defecto): válida, en los 5 min públicos spellF1 0,605 / macroF1
0,759 (mejor baseline 0,424), lat_f −1,2 fotogramas → alineada. En el partido
completo: 412 posesiones de 9,6 s y D ≈ 38% de lo evaluado.
**Diagnóstico del 22 oct (agregados):** el 38% de D es plausible (D es la
etiqueta más común según el starter oficial; en LaLiga el balón está parado
~40%), pero el modelo hace D ≈ "no hay balón": P(D | ausente) 87%,
P(D | visible) 4%, 94% de los D sin balón; huecos ≥ 15 s → D casi siempre,
de 5–15 s → 59%. En los 5 min públicos (balón ausente solo 11,8%, no
representativos) pasa lo contrario: nunca pone D en juego, pero solo acierta
el 34% de los D reales, que en 2/3 tienen el balón **visible**. Los sample
data enseñan "parada = sin balón" porque ahí el balón falta casi solo en
paradas, y la simulación oculta huecos de 1 s de mediana (máx. 20 s). El ATD
no trae ball_state ni ball_owning_team (100% nulos en kloppy).
Pero los huecos de balón de los sample data se parecen a los oficiales
(32–41% ausente, mediana 5–8 s, p90 ~35 s vs 40,7%, 10,7 s, 33,8 s) y la
velocidad de los jugadores dice que ~96% del balón ausente oficial es parada
(jugadores < 1,5 m/s en el 87% de esos fotogramas; en los sample data, 90% en
D y 29–39% en juego) y que ~13% del balón visible también lo es. Conclusión:
D ≈ "sin balón" es casi correcto; lo que falta son las paradas con balón
visible (D real estimado ~47% frente al 38% predicho), y ocultar un 41% extra
en juego al entrenar sobra. Experimento (22 oct, sin identidades, mismo
examen para todos: balón nativo + 15% de paradas con balón visible,
`--prueba-enmascarar 0 --prueba-paradas-visibles 0.15`): A actual
(`--enmascarar 0.41`) 0.816 / macro 0.909; B sin ocultar 0.814 / 0.911;
**C sin ocultar + `--paradas-visibles 0.15` 0.835 / 0.933** (recall de D
85–92% → 97% en los tres partidos); D ambas 0.827 / 0.929.
C partía las posesiones con D de < 1 s en el partido real (263 rachas D):
`evaluacion.quitar_paradas_cortas` (D entre dos posesiones más corto que
`min_d`, elegido fuera de muestra en `MIN_D_GRID`; sale 75 = 3 s) lo arregla.
Robustez con `min_d` (spellF1 / macroF1, A → C) frente a exámenes con 5, 15 y
30% de paradas visibles: 0.847/0.932 → 0.850/0.935, 0.818/0.909 →
0.845/0.933, 0.776/0.870 → 0.839/0.928. **C es la configuración por defecto
desde el 22 oct** (`--enmascarar 0 --paradas-visibles 0.15`). En el partido
real: D 46,6% de lo evaluado, 161 paradas (mediana 10,7 s), ~409 posesiones;
en los 5 min públicos spellF1 0,630 / macroF1 0,842 (A: 0,605 / 0,759).
`entregas/Getche.csv` actual (23 oct, modelo C): válido, SHA-256
9fdb24a6a8e077380b0b76d15bfb8c029850bc01f6260f6d701dc4e5f693d92c (el mismo en
dos predicciones separadas).
**Diagnóstico de fragmentación (24 oct, agregados):** 409 posesiones de 8,3 s
parecían demasiadas frente a la verdad pública (17 de 14,2 s), pero no hay
parpadeo de más: idas y vueltas X-Y-X < 3 s = 18% de las posesiones (verdad de
los sample data 13–17%, pública 18%) y cambios directos H↔A por minuto de juego
iguales a los de la verdad de los sample data (~4,4/min); el 96% ocurre con el
balón visible. Lo distinto es D entre dos posesiones del mismo equipo (23% de
las transiciones frente a 4–8%), pero esas rachas tienen jugadores caminando
(79–95%), balón casi quieto (mediana 1,5–2,5 m/s, < 7% de fotogramas a más de
8 m/s) y en los 5 min públicos son D de verdad en el 95%: parecen paradas
reales (en la definición oficial un despeje sin control no cambia la
posesión, así que el saque suele ser del mismo equipo). Rellenarlas en los
sample data no mejora (0.845 → 0.844 con < 4 s, peor con más). No se cambia
nada. Quedan 161 paradas en lo evaluado, algo más de las ~110–130 típicas de
un partido: posible exceso pequeño, sin forma de verificarlo.

**Mejoras probadas del 30 oct al 9 nov, todas descartadas** (detalle en `Getche.md` §4;
quedan en el código apagadas). Diagnóstico de los 5 min públicos (solo agregados): los
cambios están bien situados (mediana −0,2 s); el error está en las paradas con balón
visible (36% de los D reales como posesión, 4 de 6 posesiones falsas dentro de paradas).
(a) `--entre-paradas` / `quitar_posesiones_entre_paradas`: 0.844 → 0.843, se apaga solo.
(b) `--features-parada` (`par_*`): 0.844 → 0.863 en sample data pero en el partido real
P(D | balón visible) 13,8% → 5,0% y 5 min 0,630 → 0,615: aprende el balón perfectamente
quieto de `mostrar_balon_en_paradas`, que no existe en una parada real. Para
reintentarlo habría que calibrar esa simulación con agregados del partido real.
(c) `--verdad control` (`_exigir_control`; `--prueba-verdad` evalúa con otra verdad):
0.834 → 0.860 en sample data con examen "control", partido real igual (354 posesiones).
(d) `autoentreno.py` (pseudo-etiquetas p ≥ 0,9, rachas ≥ 3 s, ±1 s de margen, sin los
5 min): 0.845 → 0.853, pseudo-etiquetas 93–98% correctas en sample data, pero en el real
solo cubre el 56% y queda igual (0,626 vs 0,630). Lección: los sample data premian cosas
que no se trasladan; cada cambio pasa por leave-one-out + agregados sin etiquetas del partido
real plausibles. Los 5 min públicos no son un filtro (las reglas prohíben usarlos para elegir
entre versiones): se pasan solo sobre el archivo final, para comprobar el pipeline. **Entrega congelada en el modelo C**: el 9 nov se
regeneró en Windows con el código actual y dio exactamente `9fdb24a6…`. El
`entregas/Getche.csv` que había en disco (`678c9a89…`, 24 oct) salía de una versión sin
commit de `--verdad control` y no es la entrega. No se ha subido nada al formulario.

**Revisión del 16–17 nov (detalle en `Getche.md`, secciones 2 y 5). Entrega nueva: modelo
entrenado con ruido de balón + regla del saque inicial**, SHA-256
`b31952fbea4b91ae0de9b860fae524c83b1617f2efe477328ac08b23178563e2` (334 posesiones, D 46,0% de
lo evaluado; p_quedarse 0,99, min_frames 25, min_d 50; regla con radio 1,5 m y 10 fotogramas).
Entregas anteriores guardadas en `entregas/`: modelo C + regla del saque (`2c4b8193…`, 388
posesiones; se regenera quitando `--ruido-balon` de `reproducir.*`) y modelo C del 9 nov
(`9fdb24a6…`, `Getche_9fdb24a6_modeloC_sin_regla_del_saque.csv`). La decisión de adoptar el
ruido se tomó con leave-one-out y agregados sin etiquetas, antes de pasar la comprobación de
los 5 min públicos al archivo final: 0,600 / 0,815, 15 posesiones de 17, latencia mediana
1 fotograma (el modelo C daba 0,630 / 0,842; con 17 posesiones es ruido y, por regla, no es un
criterio). `reproducir.ps1` desde `main` regeneró `b31952fb…` exacto el 17 nov.
**Repo público de la entrega:** https://github.com/Sanetc/Getche-MTAH-2027. Se genera desde la rama `publico` de este repo
(sin historial; quita las notas internas de planificación), que se actualiza con commits normales
encima, nunca forzando. Este repo de trabajo (`Sanetc/Talent-Arena-AE`) sigue privado.
- *Comprobado:* reconstruir el modelo C desde cero da el mismo leave-one-out (0,850 / 0,845 /
  0,839) y el mismo CSV fotograma a fotograma; la dirección de ataque está bien en las dos
  partes del partido real (centroide local a la izquierda del visitante en el 98,5% de la 1ª
  parte y, tras voltear, mediana −12,9 m en la 2ª); `oficial/possession_metrics.py` es idéntico
  al oficial vigente.
- *Trampa del saque de la 2ª parte* (FAQ.md oficial): se evalúa desde 110.808 pero el saque es en
  113.742, y esos 2.934 fotogramas son D. El modelo C ponía posesión en 1.498 (21 posesiones
  falsas; balón visible 81%: los jugadores pelotean). Regla adoptada, sin números del partido:
  `datos.antes_del_saque` pone D antes del inicio de la fase `SET PIECES` `Kick Off (Start)` de
  cada parte (empieza 88 / 82 fotogramas antes del saque en la 1ª / 2ª parte); `entregar.py
  predecir --sin-saque` la apaga. No se usa la constante 113.742: el material oficial la anota
  como "from the ground truth" y las reglas nuevas prohíben constantes atadas a fotogramas.
- *Distancia entre dominios, medida sin etiquetas:* con el balón visto y en juego, el jugador
  más cercano está a 0,3–1,5 m de mediana en los sample data (p25 0,1–0,2 m; balón pegado al
  jugador) y a 1,85 m en el ATD oficial (p25 0,88 m); aceleración mediana del balón ~0,1 frente
  a 4,9 m/s². `dl_0_pas50` y `dv_0_pas50` (36% de la ganancia del modelo C) tienen KS 0,22
  entre dominios. Probabilidad máxima ≥ 0,9: 83% de los fotogramas oficiales frente a 95–96%
  en los de muestra (con balón visible, 77% frente a 95%).
- *Ruido de balón* (`balon_estimado.ruido_balon`, `--ruido-balon 0.8 30 0.06`, calibrado con
  `calibrar_ruido.py`; **adoptado el 17 nov**, `reproducir.*` lo pasa). Validación completa el 17 nov (leave-one-out,
  spellF1 medio): con ese ruido en el examen, modelo C 0,583 / 0,584 / 0,581 (5 / 15 / 30% de
  paradas visibles) frente a 0,700 / 0,682 / 0,657 del modelo entrenado con ruido (gana en 9 de
  9); con el examen sin ruido, 0,845 → 0,715. Robustez (examen al 15%): desvío 0,4 → +0,03;
  0,6 → +0,07; 0,8 → +0,11; 1,0 → +0,13; 1,2 → +0,14; tau 120 → +0,12; temblor 0,02 → +0,11;
  ruido también en jugadores → +0,10; el modelo con ruido se mueve entre 0,64 y 0,71 y el C cae
  de 0,85 a 0,50. Partido real, sin etiquetas (ambos con la regla del saque): 388 → 334
  posesiones (6,97 → 5,84 por minuto de juego; la verdad de los sample data ronda 5,7), D 47,5%
  → 46,0%, P(D | balón visible) 15,3% → 12,4%, cambios de argmax 19,9 → 11,3 por minuto,
  p máx ≥ 0,9 con balón visible 77% → 81%; el 7,5% de los fotogramas evaluados cambia. Avisos:
  la recalibración con `tau` de 10 a 120 pone el mínimo en (0,8; 10; 0,02): el desvío queda bien
  determinado por las distancias (0,8–1,0 m), `tau` y el temblor no, pero el modelo es
  insensible a ellos; el ruido saca el balón del campo cerca de las líneas (`balon_fuera`).
  El modelo final con ruido elige p_quedarse 0,99, min_frames 25, min_d 50.
- *Probado y neutro:* orden fusión → paradas → fusión (leave-one-out idéntico en 9 de 9);
  `p_quedarse` de 0,999 a 0,99999 solo mueve 416 → 388 posesiones en el partido real (la
  fragmentación no viene del Viterbi). Smart Tagging real: acuerdo de equipo del modelo con
  BUILD UP 97% y PROGRESSION 87%, pero las fases tienen pre-roll y duración fija; tras una SET
  PIECES con equipo la primera posesión predicha es de ese equipo solo en el 59–72%: no sirve
  de ancla sin validación.
- *Defectos conocidos sin corregir (exigen reentrenar):* `velocidades()` no corta los saltos de
  slot (el `rolling(min_periods=1)` de `_suave` rellena los NaN del corte); el balón "quieto" de
  `mostrar_balon_en_paradas` lleva ruido blanco de 0,1 m por fotograma (~2 m/s); en
  `verdad_desde_eventos` la comparación por subcadena trata "GOAL KICK" como "GOAL" y las
  paradas sin evento de parada quedan como posesión.
- *Entorno:* `requirements.txt` con versiones exactas (pandas 3.0.5, numpy 2.4.6, lightgbm
  4.6.0, scikit-learn 1.9.1, scipy 1.17.1, kloppy 3.19.1; Python 3.11.15); `exportar()` escribe
  siempre CRLF. Un solo proceso pesado a la vez: con un entrenamiento y otro script en segundo
  plano, Claude Code mató ambos por presión de memoria.

## 4. Pendientes (en orden)

0. **Antes del 20 oct (con la especificación oficial ya conocida):**
   - Inscribir al equipo (3–5 personas, mismo nombre de equipo para todos) y
     aceptar los términos de datos; cuenta de Google para el formulario.
   - Que todo el equipo reproduzca los resultados de los sample data.
   - ~~Cambiar la métrica de desarrollo a `spellF1`~~ Hecho el 12 oct
     (`evaluacion.py`, verdad con definición oficial, regla/modelo/suavizado
     ajustados por spellF1).
   - ~~Features sin balón / estimador de balón~~ Hecho el 12 oct: el estimador
     de balón no mejora la posesión; las features colectivas en el modelo sí
     (0.692 → 0.761 con balón oculto). Van por defecto.
   - ~~Rehacer `datos.py` para: EPTS sin identidades de jugador (46 slots, equipo
     sí), Smart Tagging XML SportsCode, homografía CSV; quitar la heurística
     del portero~~ Hecho el 13 oct: formato `"oficial"` (`stem_oficial`,
     `archivos_oficiales`, `cargar_smart_tagging`, `cargar_homografia`,
     `fuera_de_periodo`, `exportar(n_frames=199319)`), probado con un bundle
     sintético (Game 3 renombrado) y `validate_submission.py` OK. Pendiente
     de comprobar con los datos reales: metadata de 46 slots en kloppy,
     periodos que empiezan en el saque, alineación con los 5 min públicos.
   - Features sin balón para el 41% de fotogramas sin balón **en juego**
     (en los sample data no existían): movimiento colectivo, centroides,
     última posición conocida del balón y extrapolación, Smart Tagging como
     prior de equipo y de saque.
   - Salida `H/A/D/X` con `X` fuera de 11.185–90.651 y 110.808–190.197.
1. **Al llegar los datos (20 oct):** `comprobar_oficial.py --data` (agregados
   del bundle), `reproducir.ps1 -Equipo <equipo>` (primer archivo válido) y
   `comprobar_oficial.py --entrega` (validador + 5 min públicos solo para
   comprobar alineación y etiquetas), todo el primer día. Entrenar contra los 5 min
   es poco (17 posesiones): entrenar con sample data + verdad aproximada con la
   definición oficial, y usar los 5 min solo para validar.
2. Mejoras orientadas a `spellF1`: suavizado/segmentación que no parta
   posesiones (Viterbi con p_quedarse alto, duración mínima), decidir bien los
   cambios (control vs. rebote), Smart Tagging `SET PIECES` como ancla de
   saques y de `D`. (`X` en huecos largos: probada y descartada el 13 oct.)
3. Antes del 17 nov: repo público sin datos (el historial está limpio, revisado
   el 13 oct), ~~comando que regenera el CSV exacto~~ (`reproducir.ps1/.sh`,
   13 oct), documento técnico corto.

## 5. Convenciones
- Idioma de código y comentarios: español.
- Python 3.10+, pandas, numpy, lightgbm, scikit-learn; kloppy para formatos de Metrica.
- Los datos descargados (sample-data u oficiales) van en `../datos`, fuera del
  código, y no se suben a git.
- Ejecutar Python que lee datos descargados con `python -I`.
- Desarrollo en Windows (PowerShell): `.\.venv\Scripts\python.exe -I script.py ...`.
