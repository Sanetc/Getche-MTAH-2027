# Posesión por fotograma — documento técnico (Fase 1, MTAH 2027)

**Equipo:** Getche  
**Archivo entregado:** `Getche.csv` (SHA-256 `b31952fbea4b91ae0de9b860fae524c83b1617f2efe477328ac08b23178563e2`)  
**Repositorio:** https://github.com/Sanetc/Getche-MTAH-2027 (código sin datos del partido)

## 1. Enfoque

Un clasificador de gradiente (LightGBM) da, para cada fotograma, la probabilidad de `H`, `A` y
`D` a partir de unas 140 variables calculadas solo del tracking: distancias y densidades de cada
equipo alrededor del balón, dinámica del balón, movimiento colectivo de los equipos (velocidad
media de los jugadores, centroides) y ventanas temporales hacia atrás y hacia adelante. Un
decodificador de Viterbi convierte esas probabilidades en posesiones enteras penalizando los
cambios de etiqueta. Después, las paradas de menos de 2 s entre dos posesiones y las posesiones
de menos de 1 s se funden con la posesión anterior. Todo se ajusta con la métrica del
ranking, `spellF1`, no con la exactitud por fotograma.

**Por qué así.** El partido entregado no trae verdad para entrenar: los 5 minutos públicos son,
por regla, solo para comprobar el pipeline, no para entrenar ni para ajustar. Por eso entrenamos
con los tres partidos de muestra públicos de Metrica, con una verdad construida desde sus
eventos según la definición oficial de posesión, y simulamos sobre ellos las condiciones del ATD
oficial: sin identidades, con paradas con el balón visible y con el ruido de detección del balón
que medimos en el partido entregado. Un modelo aprendido en lugar de reglas escritas a mano
porque las señales se combinan mal con umbrales: la cercanía al balón decide en juego abierto,
pero cuando el balón no se ve o está quieto lo que manda es cómo se mueven los equipos. Viterbi
en lugar de una ventana de suavizado porque una posesión partida en dos cuesta casi tanto como
una perdida en `spellF1`.

## 2. Datos usados y descartados

| Dato | Uso |
|---|---|
| ATD táctico (`BAR-ATH_tactical_atd.xml/.txt`, leído con kloppy) | **Sí, es la única entrada del modelo.** Los 46 *track slots* se usan solo por equipo: no hay identidades de jugador, así que las variables son colectivas o "del slot más cercano". El ATD trae un solo periodo; las dos partes se asignan con los tramos evaluados oficiales. La dirección de ataque se normaliza por parte con la posición de los porteros (máximo/mínimo por fotograma), sin identidades. Además, de él salen (sin etiquetas) los agregados con que se calibra el ruido de entrenamiento. |
| Balón ausente (40,7 % de los fotogramas evaluados) | Se rellena por interpolación lineal y el modelo recibe además cuántos fotogramas lleva sin verse. En este partido el balón ausente es casi siempre una parada: en el 87 % de esos fotogramas la velocidad mediana de los jugadores está por debajo de 1,5 m/s, como en las paradas de los partidos de muestra (90 %) y no como en el juego (29–39 %). |
| Smart Tagging | **Sí, para una sola regla.** En cada parte, los fotogramas anteriores al inicio de la fase `SET PIECES` de tipo `Kick Off (Start)` de esa parte son `D` (sección 4). Es una regla general, calculada del archivo entregado; en este partido solo actúa en la 2ª parte, que se evalúa desde antes del saque. El resto no se usa: todas las `SET PIECES` duran 35 s fijos, así que solo el inicio es informativo, y simulado sobre los partidos de muestra como ancla de saques o como variables no mejoró `spellF1`. Medido en el partido, sin etiquetas: tras una `SET PIECES` con equipo, la primera posesión que predice el modelo es de ese equipo en el 59–72 % de los casos según cómo se cuente (18–27 fases); sin verdad para saber quién acierta, no lo usamos como ancla. |
| Homografía | **No.** Las coordenadas ya vienen en el campo y no usamos el video. |
| Grabación broadcast | **No.** Otra línea de tiempo, sin mapeo a la táctica. |
| Verdad pública (5 min) | **Solo para comprobar el pipeline:** alineación de fotogramas, formato del archivo y que las etiquetas significan lo que creemos (ahí vimos que hay `D` con el balón visible). No se entrenó con ella y ningún parámetro se ajustó contra ella. También calculamos ahí la puntuación de variantes anteriores que no se adoptaron; se cuenta en la sección 5. |
| Partidos de muestra de Metrica (3, anónimos, públicos) | **Entrenamiento.** El modelo final se entrena solo con ellos (sección 3). |

No se etiquetó nada a mano y no se usó ninguna verdad privada. Usamos un asistente de
programación (Claude Code) para escribir y revisar código; por los términos del concurso nunca
recibió los archivos del partido ni extractos: solo código y estadísticas agregadas impresas por
nuestros guiones.

**Simular el partido oficial sobre los de muestra.** Tres simulaciones, las tres con semillas
fijas:

- *Tracking sin identidades:* los jugadores de muestra se reparten en 46 slots con cortes de
  ~20 s, huecos, duplicados e intercambios (≈ 20 visibles de media, como en el ATD oficial).
- *Paradas con el balón visible:* en los partidos de muestra el balón falta en casi todas las
  paradas, así que el modelo aprendía "parada = no hay balón". Al comprobar en la ventana
  pública qué significa `D` vimos que allí hay paradas con el balón visible. Cuántas hay en el
  partido no sale de esa ventana: la velocidad de los jugadores en todo el partido entregado, sin
  etiquetas, sugiere que ~13 % del balón visible también es parada. Por eso en parte de las
  paradas de entrenamiento mostramos el balón quieto en el punto del saque durante sus últimos
  2–15 s, hasta que el balón se ve en el 15 % de los fotogramas de parada.
- *Ruido de detección del balón:* los partidos de muestra son tracking corregido y el balón va
  pegado al jugador que lo lleva; el ATD entregado es detección automática. Medido sin
  etiquetas, con el balón visto y en juego: el jugador más cercano está a 0,3–1,5 m de mediana
  en los de muestra (p25 0,1–0,2 m) y a 1,85 m en el entregado (p25 0,88 m), y la aceleración
  mediana del balón es ~0,1 m/s² frente a 4,9. Las dos variables que más pesaban en el modelo
  (distancia mínima jugador–balón en los 2 s anteriores, 36 % de la ganancia) dependen de esa
  distancia. Por eso al balón visible de los partidos de entrenamiento le sumamos un desvío
  lento (0,8 m por eje, 30 fotogramas de correlación) y un temblor blanco de 0,06 m
  (`balon_estimado.ruido_balon`). Los valores salen de `calibrar_ruido.py`, que compara entre
  los partidos de muestra con ruido y el ATD entregado los cuantiles p10–p90 de cuatro
  estadísticos sin etiquetas (distancia del balón al jugador más cercano, su mínimo en 2 s,
  aceleración y velocidad del balón) y minimiza la media de |log cociente|. El desvío queda bien
  determinado por las distancias (0,8–1,0 m en todas las rejillas probadas); la correlación y
  el temblor no (el mínimo se mueve entre 10 y 30 fotogramas según la rejilla): usamos 30 y
  0,06, el mínimo de la primera rejilla (correlación de 30 a 250), y comprobamos que el
  resultado no depende de ellos (sección 5). En el partido entregado, los fotogramas "en juego"
  de esa comparación son los que predice nuestro propio modelo sin ruido: ninguna persona
  decidió ninguna etiqueta y no intervino la verdad pública ni la privada. El ruido solo se
  aplica a los partidos de entrenamiento; el tracking entregado se predice tal como llega.

Probamos también ocultar el balón en juego al 41 % (la primera hipótesis, antes de ver el
bundle): con el examen sin ruido de entonces daba 0,818 frente a 0,845, y con un 30 % de paradas
visibles, 0,776 frente a 0,839. Lo descartamos.

## 3. Datos de entrenamiento

- **Fuente:** datos de muestra públicos de Metrica Sports, https://github.com/metrica-sports/sample-data
  (commit `e706dd5`). Es el único conjunto etiquetado del pipeline.
- **Partidos:** tres, anónimos y públicos (`Sample_Game_1`, `Sample_Game_2` y `Sample_Game_3`).
  Ninguno es el partido del concurso.
- **Qué está etiquetado:** cada fotograma, con `H`, `A` o `D`, según las reglas de posesión de la
  sección 4.
- **Quién etiquetó y cómo:** el equipo no hizo ninguna etiqueta. Las etiquetas por fotograma se
  derivan por código (`datos.verdad_desde_eventos`) de los eventos anotados por Metrica que
  acompañan a esos datos (pases, conducciones, recuperaciones, saques, tiros, faltas, balón
  fuera).
- **Cuánto:** 429.513 fotogramas etiquetados en total; 354, 319 y 380 posesiones en los partidos
  1, 2 y 3.
- **Simulaciones:** sobre esos partidos, también por código y con semillas fijas, se quitan las
  identidades de los jugadores, se muestra el balón en parte de las paradas y se suma al balón
  un ruido de detección calibrado sin etiquetas (sección 2). Las etiquetas no cambian.
- **Del partido entregado no se usó ninguna etiqueta, evento ni tracking corregido, de ninguna
  fuente.** El modelo solo lo recibe como entrada al predecir, y de él solo salen los agregados
  sin etiquetas con que se calibra el ruido.
- **Los 5 minutos de verdad pública no se usaron para entrenar ni para ajustar.**

## 4. Reglas de posesión

Las reglas las aprende el modelo de la verdad de entrenamiento, que se construyó así desde los
eventos de los partidos de muestra:

- **Balón en juego:** siempre posee un equipo, el del último pase, conducción, recuperación,
  saque o tiro.
- **Balón suelto, desvíos y duelos:** un pase desviado sin interceptar, un despeje sin control o
  un duelo aéreo sin control posterior **no cambian** la posesión. La conserva el que la tenía
  hasta que el rival controla. Un tiro parado tampoco la cambia por sí solo.
- **Balón muerto (`D`):** desde que el balón sale, hay gol, falta o fuera de juego, hasta el
  siguiente saque. Las faltas siguen siendo `D` aunque haya disputa: son la mayoría de los
  ejemplos de "balón visible y jugadores quietos" y sin ellas el modelo falla en paradas largas
  (lesiones). En el descanso y tras el final de cada parte, `D`.
- **Antes del saque inicial de cada parte:** `D`, por regla y no por el modelo. La 2ª parte se
  evalúa desde bastante antes de su saque (los jugadores vuelven al campo con el balón a la
  vista) y el modelo, entrenado con partes que empiezan en el saque, marca posesión en buena
  parte de ese tramo (1.952 de 2.852 fotogramas antes de aplicar la regla). La regla sale del
  Smart Tagging entregado y vale para cualquier partido: en cada parte, todo fotograma anterior
  al inicio de la fase `SET PIECES` de tipo `Kick Off (Start)` es `D`; desde ese inicio decide
  el modelo. En la 1ª parte no cambia nada (la fase empieza antes del primer fotograma
  evaluado). La organización publica en FAQ.md el fotograma del saque de la 2ª parte (113.742):
  ese número no entra en el pipeline y solo lo usamos para comprobar que la regla cae donde debe
  (el inicio de la fase queda 82 fotogramas antes).
- **Saques:** pertenecen al equipo que saca; la posesión empieza con el saque.
- **`X`:** no se usa dentro de los fotogramas evaluados. La probamos en los huecos largos del
  balón y empeora: el evaluador la rellena con la etiqueta anterior y eso retrasa los cambios de
  posesión que ocurren sin ver el balón.

**Post-procesado** (solo continuidad, todo elegido por `spellF1` fuera de muestra entre los
partidos de entrenamiento): Viterbi con probabilidad de permanencia 0,99; paradas de menos de
50 fotogramas (2 s) entre dos posesiones pasan a la posesión anterior, porque un `D` de un
instante parte una posesión en dos; posesiones de menos de 25 fotogramas (1 s) se funden con la
anterior. Al final se aplica la regla del saque inicial.

## 5. Validación

Sin verdad del partido completo, la validación es *leave-one-out* entre los tres partidos de
muestra, con el evaluador oficial (`possession_metrics.py`) y en condiciones simuladas: sin
identidades, con un porcentaje de paradas con balón visible y con el ruido de detección del
balón calibrado (sección 2). spellF1 / macroF1, media de los tres partidos:

| Paradas con balón visible en el examen | Modelo entregado (entrenado con ruido) | El mismo modelo entrenado sin ruido |
|---|---|---|
| 5 % | 0,700 / 0,905 | 0,583 / 0,877 |
| 15 % | 0,682 / 0,893 | 0,584 / 0,875 |
| 30 % | 0,657 / 0,877 | 0,581 / 0,871 |

El modelo con ruido gana en los nueve exámenes (tres partidos × tres porcentajes). Con el examen
sin ruido (el balón pegado al jugador, como vienen los datos de muestra) pasa lo contrario:
0,845 entrenando sin ruido frente a 0,715 con él. Elegimos el modelo con ruido porque las
distancias medidas en el partido entregado dicen que ese examen limpio no lo representa; fue
una decisión tomada con la validación en otros partidos y con agregados sin etiquetas, no con
los 5 minutos públicos.

Robustez a la calibración (examen al 15 %; ventaja en spellF1 del modelo con ruido sobre el
entrenado sin ruido cuando el ruido del examen no es el del entrenamiento):

| Ruido del examen | Ventaja |
|---|---|
| Desvío de 0,4 / 0,6 / 0,8 / 1,0 / 1,2 m | +0,03 / +0,07 / +0,11 / +0,13 / +0,14 |
| Correlación de 120 fotogramas | +0,12 |
| Correlación de 10 fotogramas y temblor de 0,02 m | +0,11 |
| Desvío repartido entre el balón y los jugadores | +0,10 |

En todos esos exámenes el modelo con ruido queda entre 0,64 y 0,71; el entrenado sin ruido cae
de 0,85 (sin ruido) a 0,50 (desvío de 1,2 m).

Lo que aporta cada etapa (examen con ruido al 15 %): LightGBM solo 0,34 (parpadea), +Viterbi
0,68, +paradas mínimas y fusión 0,68. La regla del saque inicial no afecta a esta validación: en
los partidos de muestra cada parte empieza en el saque.

Son otros partidos, una verdad aproximada y un ruido simulado: indican el orden de magnitud, no
el resultado final. Comprobación del pipeline: en los 5 minutos públicos el archivo entregado da
spellF1 0,600 y macroF1 0,815 (mejor baseline oficial 0,424; 15 posesiones predichas frente a
17, cambios con un fotograma de retraso en la mediana). Son 17 posesiones: el número confirma
que el archivo está alineado y bien formado, no mide el modelo, y no se usó para entrenar, para
ajustar ni para elegir entre versiones. La pasamos una sola vez, sobre este archivo y después de
decidir; el modelo anterior daba ahí 0,630 / 0,842, una diferencia menor que lo que mueve una
sola posesión (unos 6 puntos). En el partido completo predice 334
posesiones y `D` en el 46,0 % de lo evaluado.

Agregados sin etiquetas del partido entregado, modelo entrenado sin ruido → con ruido (los dos
con la regla del saque): 388 → 334 posesiones (7,0 → 5,8 por minuto de juego; en la verdad de
los partidos de muestra son unas 5,7), `D` 47,5 % → 46,0 %, `D` con balón visible 15,3 % →
12,4 % (se esperaba ~13 %), y la clase más probable cambia 19,9 → 11,3 veces por minuto.

**Historia: lo probado antes con la validación sin ruido** (modelo de entonces, "modelo C":
0,850 / 0,845 / 0,839 con 5 / 15 / 30 % de paradas visibles). Descartado: estimar la posición
del balón ausente con un modelo (acierta mejor la posición, pero no la posesión), un modelo
híbrido con un especialista sin balón para los huecos (parpadea más), el Smart Tagging simulado
como ancla o variables (dentro del ruido), la `X` en huecos largos (peor en todas las variantes)
y rellenar las paradas cortas entre dos posesiones del mismo equipo (sin mejora).

Probado entre el 30 oct y el 9 nov y descartado. Cada prueba se midió con la validación
*leave-one-out* en los partidos de muestra y con agregados sin etiquetas del partido entregado
(estadísticas de las predicciones del propio modelo). Para ser exactos sobre la ventana pública:
la idea de las dos primeras pruebas salió de ver allí que el modelo daba como posesión parte de
los `D` con el balón visible, y también miramos allí la puntuación de las variantes (0,615 con
las variables de balón parado y 0,626 con el autoentrenamiento, frente a 0,630 del modelo de
entonces; con 17 posesiones, una sola mueve unos 6 puntos, así que esas diferencias son ruido).
Ningún parámetro se ajustó con esa ventana y ninguna de estas variantes se adoptó. Las reglas no
permiten usar esa ventana para elegir entre versiones y no forma parte de nuestro criterio de
decisión.

| Prueba | Partidos de muestra (leave-one-out, sin ruido) | Partido entregado (agregados sin etiquetas) |
|---|---|---|
| Posesiones de menos de `min_pd` entre dos paradas pasan a `D` (`--entre-paradas`) | 0,844 → 0,843: el ajuste fuera de muestra lo apaga en 2 de 3 partidos | — |
| Variables de balón parado y visible: quieto, en un punto de reanudación, jugadores lentos (`--features-parada`) | 0,844 → 0,863 | Peor: `D` con balón visible baja de 13,8 % a 5,0 % (se esperaba ~13 %); lo aprendido con las paradas simuladas no se traslada a las reales |
| Verdad "control": un toque del rival sin pase, tiro ni saque de menos de 1 s no cambia la posesión (`--verdad control`) | 0,834 → 0,860 (examen con esa verdad) | Igual: 354 posesiones frente a 343–409, mismas paradas |
| Autoentrenamiento con las predicciones seguras del propio partido (`autoentreno.py`) | 0,845 → 0,853; las pseudo-etiquetas aciertan el 93–98 % | Poca cobertura: el modelo solo está seguro en el 56 % de los fotogramas (82 % en los de muestra) |

Lo que enseñaron las cuatro: lo que faltaba era la distancia entre los partidos de muestra y
este tracking, y no se cerraba con retoques del modelo. Las cuatro siguen en el código, apagadas
por defecto.

**Revisión del 16–17 nov.** Reconstruimos el modelo desde cero (misma validación y mismo
archivo, fotograma a fotograma), comprobamos con agregados que la dirección de ataque se
normaliza bien en las dos partes del partido entregado y medimos esa distancia entre dominios
(sección 2). De ahí salen los dos cambios de esta entrega:

| Prueba (16–17 nov) | Resultado | Decisión |
|---|---|---|
| `D` antes del saque inicial de cada parte (Smart Tagging, sección 4) | Con el modelo de entonces: 409 → 388 posesiones en el partido entregado; todos los fotogramas que cambian están antes del saque de la 2ª parte. No toca la validación en los partidos de muestra | **Adoptada** |
| Ruido de detección del balón en los partidos de entrenamiento (`--ruido-balon 0.8 30 0.06`), calibrado sin etiquetas con `calibrar_ruido.py` | Tablas de arriba: 0,58 → 0,68 con el ruido en el examen, robusto a la calibración; pierde con el examen sin ruido | **Adoptada** |
| Orden del post-procesado: fusión → paradas → fusión | *Leave-one-out* idéntico en los 9 exámenes (medido con el modelo de entonces) | No adoptada (neutra) |

## 6. Limitaciones

- **La verdad de entrenamiento** sale de eventos de otros partidos, no de tracking corregido
  como la oficial; los criterios finos (cuándo un toque es "control") pueden diferir.
- **El ruido de entrenamiento es una simulación.** Está calibrado con agregados del partido,
  pero si el ruido real tiene otra estructura (desvíos sistemáticos, detecciones falsas del
  balón) la mejora medida en los partidos de muestra puede no trasladarse entera. Con un examen
  sin ruido este modelo pierde 0,13 frente al entrenado sin ruido: apostamos a que el partido se
  parece al examen con ruido, que es lo que dicen las distancias medidas.
- **Funde algunas posesiones.** En los exámenes con ruido predice un 13 % menos de posesiones que
  la verdad (913 frente a 1.053 en los tres partidos); el modelo entrenado sin ruido predecía un
  14 % de más.
- **Paradas con el balón visible:** son el error más probable. El modelo las detecta por el
  ritmo de los jugadores y un balón quieto; un juego muy lento (circulación del Barça atrás)
  puede parecerse, y una parada con el balón en movimiento (saque rápido) puede pasar como
  juego.
- **Cambios con el balón no visto:** los decide el movimiento colectivo y pueden llegar tarde.
- **Saque inicial de la 2ª parte:** la regla deja al modelo desde el inicio de la fase
  `Kick Off (Start)`, unos 3 s antes del saque real, así que la primera posesión de esa parte
  puede empezar un poco antes de tiempo.
- **Ventanas futuras:** las variables usan información posterior al fotograma. El método es
  *offline* sobre el partido completo, como permite la tarea.
- **Con más tiempo:** calibrar también la correlación temporal del ruido con un estadístico que
  la determine, simular paradas con el balón llevado por un jugador y construir una verdad de
  entrenamiento desde tracking, como la oficial.

## 7. Reproducción

```powershell
git clone https://github.com/Sanetc/Getche-MTAH-2027   # el código; los datos van en ..\datos, fuera del repo
cd Getche-MTAH-2027
.\setup.ps1                                            # venv + dependencias + datos de muestra de Metrica
.\reproducir.ps1 -Equipo Getche -Oficial ..\datos\oficiales\data
```
En Linux/macOS: `./setup.sh` y `./reproducir.sh Getche ../datos/oficiales/data`.

`reproducir` entrena el modelo con los datos de muestra y el ruido de balón calibrado
(`--ruido-balon 0.8 30 0.06`; ~15 min, 16 GB de RAM), predice la grabación táctica (carpeta
`tactical` del bundle, con su ATD y su Smart Tagging), escribe `entregas\Getche.csv`, lo pasa
por `validate_submission.py` si el repo oficial está clonado en `..\datos\mtah-2027-oficial` y
muestra su SHA-256, que debe coincidir con el de arriba. La calibración del ruido se repite con
`python -I calibrar_ruido.py --data ..\datos\sample-data\data --oficial ..\datos\oficiales\data --entrega <predicción sin ruido>`
(solo imprime cuantiles).

**Entorno de referencia:** Windows 11 x64, Python 3.11.15 y las versiones exactas de
`requirements.txt` (pandas 3.0.5, numpy 2.4.6, lightgbm 4.6.0, scikit-learn 1.9.1,
scipy 1.17.1, kloppy 3.19.1); datos de muestra en el commit `e706dd5`. En ese entorno el
resultado es determinista: semillas fijas y LightGBM en modo determinista, también con distinto
número de hilos, y el CSV se escribe siempre con fin de línea CRLF. En otra arquitectura o con
otras versiones el entrenamiento puede dar un modelo ligeramente distinto (en una Mac con otras
versiones obtuvimos otro archivo): el archivo entregado es el de este entorno.
