# MTAH 2027 — Posesión por fotograma (Fase 1)

Código del equipo para la Fase 1 de **More Than a Hack 2027** (Mobile World Capital +
Barça Innovation Hub + Metrica Sports): estimar qué equipo tiene la posesión en cada
fotograma de un partido a partir del tracking. Ranking automático, pasan los 20 mejores.
Fase 1: 20 oct – 17 nov 2026. Contexto completo del concurso en `CLAUDE.md`.

## En qué vamos (17 nov 2026)

- **Entrega: modelo entrenado con ruido de balón + regla del saque inicial.** `reproducir.ps1` /
  `reproducir.sh` regeneran `Getche.csv` con el SHA-256 de [`Getche.md`](Getche.md) (`b31952fb…`,
  generado el 17 nov en Windows). En el partido completo: 334 posesiones y `D` en el 46,0% de lo
  evaluado. Comprobación del pipeline en los 5 min públicos: spellF1 0,600 / macroF1 0,815
  (mejor baseline oficial 0,424); con 17 posesiones no mide el modelo ni sirve para comparar
  versiones.
- **Modelo:** LightGBM + Viterbi + fusión de posesiones cortas. Se entrena con los sample
  data en condiciones oficiales: sin identidades, 15% de paradas con balón visible, balón
  interpolado y, desde el 17 nov, ruido de detección del balón calibrado sin etiquetas con el
  ATD oficial (`--ruido-balon 0.8 30 0.06`). Leave-one-out con ese ruido en el examen:
  spellF1 ≈ 0,68 (el modelo anterior, "modelo C", da 0,58 en ese examen y 0,845 en el examen
  sin ruido, donde el nuevo da 0,715).
- **Regla del saque inicial (17 nov):** en cada parte, los fotogramas anteriores al inicio de la
  fase `SET PIECES` de tipo `Kick Off (Start)` del Smart Tagging son `D`. La 2ª parte se evalúa
  desde antes de su saque y el modelo C ponía ahí 21 posesiones falsas (409 → 388);
  `--sin-saque` la apaga.
- **Probado del 30 oct al 9 nov y descartado** (detalle en `Getche.md`, sección 5; todo queda en
  el código, apagado por defecto): `--entre-paradas`, `--features-parada`, `--verdad control` y
  `autoentreno.py`. Mejoraban o empataban en los sample data, pero los agregados sin etiquetas
  del partido oficial no mejoraban (uno empeoraba). En esas pruebas también miramos la
  puntuación en los 5 min públicos (0,615 y 0,626 frente a 0,630: ruido con 17 posesiones); las
  reglas no permiten usar esa ventana para elegir entre versiones, así que no es un filtro.
  Cada cambio tiene que pasar dos filtros: (1) leave-one-out en los sample data y (2) agregados
  sin etiquetas del partido oficial plausibles (`D` con balón visible ~13%, número de paradas y
  posesiones). Los 5 min públicos se pasan solo sobre el archivo que se va a subir, para
  comprobar que el pipeline es correcto (alineación, formato, significado de las etiquetas), no
  para decidir.
- **Revisión del 16–17 nov:** las reglas oficiales cambiaron el 22–23 oct (el documento técnico
  debe traer una sección de datos de entrenamiento; los 5 min públicos no son conjunto de
  ajuste). Medimos sin etiquetas la distancia entre dominios: en los sample data el balón va
  pegado al jugador (mediana 0,3–1,5 m al más cercano) y en el ATD oficial está a 1,85 m y
  tiembla. De ahí sale `--ruido-balon` (ruido de detección del balón al entrenar, calibrado con
  `calibrar_ruido.py`): en leave-one-out con ese ruido en el examen sube de 0,58 a 0,68 (gana en
  los 9 exámenes y con otros niveles de ruido, de +0,03 a +0,14) y pierde con el examen limpio
  (0,85 → 0,71). Lo adoptamos porque las distancias medidas dicen que el partido no se parece al
  examen limpio. Detalle en `Getche.md`, secciones 2 y 5. La entrega anterior (modelo C + regla
  del saque, `2c4b8193…`) se regenera quitando `--ruido-balon` de `reproducir.*`.
- **Reproducibilidad:** el entorno de referencia es Windows 11 x64 con Python 3.11.15 y las
  versiones exactas de `requirements.txt`; el CSV se escribe siempre con fin de línea CRLF. En
  una Mac con otras versiones (PR #1, Poetry y Docker) el entrenamiento dio otro archivo
  (`efe2bb9b…`), y la diferencia no es solo el fin de línea: el archivo entregado es el de
  Windows.
- **Código público (17 nov):** https://github.com/Sanetc/Getche-MTAH-2027, generado desde la rama `publico` (un solo commit, sin el
  historial de trabajo ni las notas internas de planificación). Este repo de trabajo sigue
  privado. **Pendiente:** subir la entrega por el formulario.

> Las secciones que siguen son notas fechadas del desarrollo (octubre): cuentan cómo se llegó
> hasta aquí y algunas cifras y planes ya no están vigentes. El método y los números actuales
> están arriba y en [`Getche.md`](Getche.md).

## Qué hacemos y cómo (leer primero)

### Los datos
Todavía no hay datos oficiales (llegan el 20 oct). Trabajamos con los **3 partidos de
muestra públicos de Metrica**, anónimos ("Home"/"Away", "Player1…"), que tienen la misma
pinta que lo que esperamos recibir:

- **Tracking:** 25 veces por segundo, la posición (x, y) de los 22 jugadores y del balón
  sobre el campo. Un partido son ~140.000 fotogramas. No hay video que mirar: solo
  coordenadas. Games 1 y 2 vienen en CSV; Game 3 en formato EPTS (`metadata.xml` +
  `tracking.txt`), que es el formato actual de Metrica y el más probable para los oficiales.
- **Eventos:** lista de pases, conducciones, recuperaciones, tiros, faltas, balón fuera…,
  cada uno con equipo y fotograma de inicio/fin. En el concurso llegarán como
  "Smart Tagging" automático.
- **Homografía:** la relación video ↔ coordenadas de campo. En los datos de muestra ya
  está aplicada (las coordenadas vienen en campo), así que de momento no la usamos.

Cosas que hubo que normalizar: convertir coordenadas 0–1 a metros (campo 105 × 68, origen
al centro) y **voltear el campo cada periodo para que el local ataque siempre hacia +x**
(se detecta por dónde está el portero local). Sin esto, "el balón va hacia la portería
rival" significa cosas distintas en cada parte.

### Qué es "posesión" aquí
Tres clases por fotograma: **Local**, **Visitante** o **Ninguno**. La idea de fondo es:
tiene la posesión el equipo del último jugador que controló el balón, y nadie la tiene
cuando el juego está parado (saque pendiente, falta, celebración) o el balón está fuera.

No tenemos etiquetas reales. Construimos una **verdad aproximada desde los eventos**:
después de un pase/conducción/recuperación/saque/tiro, la posesión es de ese equipo hasta
el siguiente evento; después de un gol, falta, balón fuera o final de parte, es Ninguno
hasta el siguiente saque; los tramos ambiguos (balón perdido en disputa, rechace tras un
tiro) se excluyen de la métrica. Es nuestra regla, no la de la organización: los números
sirven para comparar versiones, no como cifra absoluta.

### Cómo lo predecimos (tres niveles)
1. **Jugador más cercano:** si hay un jugador a ≤ 1,5 m del balón, la posesión es de su
   equipo; si no hay nadie, Ninguno. Solo con esto, 51–61%: falla en cada pase y no
   entiende las paradas.
2. **Reglas con memoria:** la posesión se mantiene mientras nadie controle el balón; para
   cambiar de equipo, el rival tiene que estar controlando varios fotogramas seguidos
   (histéresis); si el balón va muy rápido (pase largo, despeje) nadie lo controla; balón
   fuera del campo = Ninguno. 92–96%.
3. **Modelo LightGBM + suavizado:** ~60 variables por fotograma (distancias de los 3
   jugadores más cercanos de cada equipo, cuántos hay a 3/5/10 m, velocidad y dirección
   del balón, lo mismo en ventanas de 0,2 / 0,6 / 2 s hacia atrás y hacia adelante,
   velocidad media de los 22 jugadores, fotogramas sin ver el balón, salida de la regla
   del nivel 2). Un clasificador aprende de ellas; un Viterbi quita los parpadeos y las
   posesiones de menos de medio segundo se funden con la anterior. **spellF1 0,86** de
   media validado en partidos que el modelo no ha visto (93,5% de fotogramas).

Lo que más ha movido la aguja hasta ahora: (a) etiquetar bien las paradas (`D`) en la
verdad; (b) el suavizado, porque la métrica oficial cuenta posesiones enteras y un modelo
que acierta más fotogramas pero parpadea puntúa peor; (c) el balón ausente en juego, que
es el gran problema de los datos oficiales (41%) y que simulamos ocultándolo nosotros.

### Cómo se mide
Con el evaluador oficial (`spellF1`, `macroF1`) y validación *leave-one-out*: para evaluar
cada partido, todo (parámetros de la regla, modelo, suavizado) se ajusta solo con los otros
dos. Así el número que reportamos es el que esperaríamos en un partido nuevo.

### Qué hay en el repo
Instalar con `setup.ps1`, reproducir los resultados con los dos primeros comandos de
abajo (~10 min en total), y entregar con `entregar.py`. Todo lo que depende del formato
de datos está en `datos.py`, para que el 20 de octubre solo haya que tocar ahí.

## Instalación
```powershell
.\setup.ps1                     # Windows: crea .venv, instala dependencias, baja datos a ..\datos
.\.venv\Scripts\python.exe -I baseline_posesion.py --data ..\datos\sample-data\data --out salida
.\.venv\Scripts\python.exe -I modelo_posesion.py   --data ..\datos\sample-data\data --out salida
```
En Linux/macOS: `./setup.sh` y `python -I ...` con el venv activado.

El baseline tarda ~1 min (la carga EPTS del Game 3 con kloppy son 30 s); el modelo ~8 min.
Si en Windows pandas falla al importar con "Una directiva de Control de aplicaciones bloqueó
este archivo", es Smart App Control bloqueando un `.pyd` sin firma de pandas 3.0.6;
`requirements.txt` excluye esa versión (3.0.5 carga bien).
Generan `salida/<juego>_posesion.csv` y `salida/<juego>_posesion_modelo.csv`, una fila por fotograma.

## Estructura
| Archivo | Qué hace |
|---|---|
| `datos.py` | Carga de tracking (CSV Raw o EPTS, detección automática) y eventos, dirección de ataque normalizada (local siempre hacia +x), verdad aproximada desde eventos, métricas, `exportar()` con `FORMATO_ENTREGA`. **Es lo único que debería cambiar con los datos oficiales.** |
| `baseline_posesion.py` | Niveles 1 y 2: jugador más cercano + histéresis + filtro de velocidad. Búsqueda de parámetros en malla. |
| `modelo_posesion.py` | Nivel 3: LightGBM (~60 features + colectivas) + Viterbi + fusión de posesiones cortas, todo ajustado por `spellF1`. Validación leave-one-out entre partidos. `--enmascarar 0.41 --balon interpolar --sin-identidades` simula las condiciones oficiales; `--huecos` y `--smart-tagging` son los experimentos descritos abajo. |
| `tracking_sin_id.py` | Simula el ATD oficial sin identidades: 46 track slots por partido con cortes, huecos, duplicados e intercambios de pista (`quitar_identidades`). |
| `smart_tagging.py` | Simula las fases `SET PIECES` del Smart Tagging desde los eventos de muestra y las usa como ancla de saques o como features. |
| `evaluacion.py` | Envuelve el evaluador oficial (`oficial/possession_metrics.py`, copia MIT del repo del concurso): `spellF1`, `macroF1`, y el post-procesado `fusionar_rachas_cortas()`. |
| `balon_estimado.py` | Simula el balón ausente en juego (`enmascarar_balon`) y lo estima desde el movimiento colectivo + la historia del balón (`estimar_balon`). |
| `entregar.py` | Entrena el modelo final con todos los partidos (opcionalmente con balón oculto y estimador dentro) y lo guarda; predice una carpeta de partidos y escribe los CSV de entrega `frame,label` con H/A/D/X. |
| `analiza_errores.py` | Clasifica los fotogramas mal predichos (cambio de posesión, balón en vuelo, disputa, sin balón...) y mide las rachas de error. |
| `reproducir.ps1` / `reproducir.sh` | Regeneran la entrega de punta a punta (entrenar → predecir → validar → SHA-256), de forma determinista. |
| `Getche.md` | Documento técnico de la entrega (enfoque, datos usados y descartados, reglas de posesión, validación, limitaciones, reproducción), con el SHA-256 de `Getche.csv`. Se sube tal cual al formulario. |
| `comprobar_oficial.py` | Guion del día de los datos: estadísticas agregadas del bundle (periodos frente a los oficiales, slots visibles, balón ausente, Smart Tagging, homografía) y, con `--entrega`, el validador oficial más `run_evaluation.py` en los 5 min públicos junto a los baselines. Solo imprime agregados. |
| `calibrar_ruido.py` | Elige los tres parámetros de `balon_estimado.ruido_balon` (ruido de detección del balón para los partidos de entrenamiento) comparando cuantiles sin etiquetas entre los sample data y el ATD oficial. Solo imprime agregados. `reproducir.*` pasa `--ruido-balon 0.8 30 0.06` desde el 17 nov (ver `Getche.md`, secciones 2 y 5). |
| `autoentreno.py` | Autoentrenamiento con las predicciones seguras del propio partido: `validar` (leave-one-out en los sample data) y `oficial` (partido oficial, sin pseudo-etiquetar los 5 min públicos). Probado el 9 nov y descartado: no sale del ruido en el partido oficial. |
| `visor.py` | Genera un HTML local que anima el tracking (jugadores por equipo y balón) con la posesión predicha, la verdad si la hay y las fases del Smart Tagging, con una línea de tiempo para saltar entre cambios. El HTML lleva datos del partido: se escribe en `../datos/visor/` y no se sube a ningún sitio. |

## Resultados con los datos de muestra, métricas oficiales (12 oct 2026)
Desde el 12 oct todo se mide y se ajusta con **`spellF1`**, la métrica del ranking (posesiones
enteras emparejadas por solapamiento temporal; `evaluacion.py` envuelve el evaluador oficial
copiado en `oficial/`), y la verdad de los sample data sigue la **definición oficial**: con el
balón en juego siempre posee un equipo, en balón suelto lo conserva el que lo tenía, `D` solo
con el balón muerto. Cada partido se evalúa con todo (regla, modelo, `p_quedarse`, duración
mínima de posesión) ajustado **solo en los otros dos**. Promedio de los tres partidos:

| Salida | spellF1 | macroF1 | exactitud | posesiones pred./reales | duración media (s) |
|---|---|---|---|---|---|
| Nivel 1: jugador más cercano | 0.30 | 0.46 | 57% | 919 / 351 | 6.3 / 10.6 |
| Regla N2 (histéresis) | 0.705 | 0.905 | 90.6% | 313 / 351 | 11.4 / 10.6 |
| LightGBM sin suavizar | 0.729 | 0.945 | 94.5% | 511 / 351 | 7.0 / 10.6 |
| + Viterbi | 0.870 | 0.945 | 94.5% | 362 / 351 | 9.9 / 10.6 |
| **+ fusión de posesiones < 12 fotogramas** | **0.869** | 0.945 | 94.5% | 357 / 351 | 10.0 / 10.6 |

Por partido (salida final): Game 1 0.869, Game 2 0.854, Game 3 0.883 (ejecución del 13 oct con
el código actual; el 12 oct, antes de hacer las velocidades robustas a saltos, daba 0.873: la
diferencia es ruido). Sin las features colectivas (`--sin-colectivas`) el modelo da 0.863. El floor oficial es 0.243, pero sobre
tracking sin corregir y con el balón ausente el 41% del tiempo en juego; no son comparables.
Dos lecciones de la tabla: LightGBM solo acierta más fotogramas que la regla pero puntúa peor
en `spellF1` porque parpadea (511 posesiones donde hay 351), y el Viterbi es lo que convierte
fotogramas en posesiones. La fusión de rachas cortas aporta poco encima.

### El balón ausente en juego: simulación del 41% oficial
En los sample data el balón falta casi solo en paradas (en juego: 2–10% según el partido).
Para prepararnos, `balon_estimado.enmascarar_balon()` oculta el balón en ventanas aleatorias
(duración log-normal, mediana 1 s, hasta 20 s) hasta el 41% de los fotogramas en juego, en
entrenamiento y en prueba. `modelo_posesion.py --enmascarar 0.41 --balon {ninguno,interpolar,estimar}`:

| Balón ausente (41% en juego) | spellF1 | macroF1 | exactitud | posesiones LightGBM sin suavizar |
|---|---|---|---|---|
| Sin rellenar (NaN) | 0.649 | 0.870 | 87.0% | 1011 |
| **Interpolación lineal entre la última y la siguiente vez visto** | **0.692** | 0.891 | 89.3% | 744 |
| Estimador colectivo (ver abajo), estimaciones dentro de muestra en el entreno | 0.667 | 0.872 | 87.3% | 1426 |
| Estimador colectivo con cross-fitting | 0.662 | 0.882 | 88.2% | 1472 |
| Estimador con cross-fitting + corrección suavizada por hueco (interpolación pura si ≤ 1 s) | 0.672 | 0.887 | 88.8% | 1092 |
| **Interpolación + features colectivas en el modelo de posesión** | **0.761** | 0.912 | 91.3% | 600 |

El balón ausente cuesta ~0.17 de `spellF1` sin hacer nada. Rellenarlo por interpolación
recupera un cuarto. El estimador de balón, aunque acierta mejor la posición, **no mejora la
posesión**: sus variantes quedan por debajo de la interpolación y dentro del ruido (< 0.04)
entre sí, porque la estimación salta entre fotogramas y LightGBM parpadea el doble (1472
posesiones frente a 744); suavizarla reduce el parpadeo (1092) pero no lo elimina.

Lo que sí funciona es **darle al modelo de posesión las mismas features colectivas
directamente** (centroides y su velocidad, líneas, dispersión, centroide ponderado por
velocidad, jugadores más rápidos, y el balón relativo al bloque de cada equipo): con el balón
oculto, 0.692 → **0.761**, bien por encima del ruido, y recupera dos tercios de lo que cuesta el
balón ausente. Sin ocultar el balón también suma, pero poco (0.863 → 0.873). Es decir: el
modelo no necesita que le reconstruyamos el balón; necesita ver cómo se mueven los equipos.
**Configuración por defecto desde el 12 oct: features colectivas + interpolación.**

### `X` en huecos largos: probado y descartado (13 oct)
El evaluador oficial rellena cada `X` con la última etiqueta de esa parte antes de emparejar
posesiones, así que en `spellF1` entregar `X` dentro de un hueco del balón equivale a
**mantener la etiqueta anterior al hueco**; en `macroF1` la `X` además cuenta como fallo. Por
eso la prueba es "mantener la etiqueta" (`evaluacion.mantener_en_huecos()`,
`modelo_posesion.py --huecos`), en tres variantes y para huecos de ≥ 1, 2, 5 y 10 s, aplicada
entre el Viterbi y la fusión. Promedio leave-one-out (spellF1):

| Huecos de ≥ 1 s / ≥ 10 s | Balón oculto 41% | Sin ocultar |
|---|---|---|
| Salida final actual (sin tocar los huecos) | **0.761** | **0.873** |
| Mantener solo si el modelo cambia de equipo dentro del hueco (`D` se respeta) | 0.737 / 0.750 | 0.872 / 0.873 |
| Mantener en todo el hueco (= entregar `X`) | 0.689 / 0.705 | 0.845 / 0.844 |
| Sustituir solo los `D` dentro del hueco | 0.710 / 0.714 | 0.845 / 0.844 |

Empeora en todas las variantes y en los tres partidos. Con las features colectivas el modelo ya
sitúa bien los cambios de posesión que ocurren mientras el balón no se ve; mantener la etiqueta
anterior los retrasa hasta que el balón reaparece, y un retraso de 1 s ya baja el IoU de esa
posesión a ~0,65. Pisar los `D` es peor todavía porque muchos huecos son paradas reales.
Conclusión: no entregamos `X` dentro de los fotogramas evaluados; `X` queda solo para los
fotogramas fuera de los tramos evaluados.

### Tracking sin identidades: simulado y casi gratis (13 oct)
El ATD oficial no dice quién es cada jugador: 46 "track slots" por partido, con equipo pero sin
persona, que aparecen, desaparecen y se intercambian (20,5 objetos visibles de media en juego, a
veces más de 22). `tracking_sin_id.quitar_identidades()` lo simula sobre los sample data: cada
pista se parte en tracklets de ~20 s separados por huecos (mediana 1 s), un 10% de los cortes
deja un duplicado 2 s, y cuando dos compañeros se cruzan a < 1,5 m sus pistas se intercambian
con probabilidad 0,3. Resultado: 46 slots, ~6.500 tracklets y 40–64 intercambios por partido,
20,7 jugadores visibles de media (oficial: 20,5), más de 22 en el 5% de los fotogramas.

Dos cosas dependían de la identidad y se han cambiado para siempre: la dirección de ataque
(antes "la columna con la media más lejana es el portero"; ahora, por fotograma, el local más
alejado del centro, y la mediana del periodo decide; decide igual que antes en los 6 periodos
con identidades y, sin ellas, la antigua fallaba en uno) y las velocidades por jugador (un salto
de > 1 m entre fotogramas ya no es un movimiento sino un cambio de persona en el slot: se corta
la pista ahí). Leave-one-out, spellF1 medio (`--sin-identidades`):

| | Con identidades | Sin identidades (slots simulados) |
|---|---|---|
| Balón completo | **0.869** | 0.857 |
| Balón oculto 41% + interpolación | **0.760** | 0.750 |

La pérdida es de 0,01, dentro del ruido: todas las features son colectivas o "del más
cercano", y a eso le da igual quién sea quién. La condición oficial completa (sin identidades y
sin balón el 41% del tiempo en juego) queda en **0.750**.

### Smart Tagging como ancla de saques: simulado, no aporta (13 oct)
Las fases `SET PIECES` oficiales (59 en el partido: 26 saques de banda **sin equipo**, 12 córners,
10 saques de puerta, 5 faltas, 6 saques de centro; un partido tiene ~100 reinicios) dicen que ahí
hubo un reinicio y, salvo en los saques de banda, de quién. `smart_tagging.py` las simula desde
los eventos de muestra con ese recall por tipo, ventanas [inicio − 1..5 s, inicio + 5..15 s],
5 fases falsas por partido y un 5% de equipo equivocado, y las usa de dos formas
(`modelo_posesion.py --smart-tagging {ancla,features,ambos}`), en condiciones oficiales
(sin identidades, balón oculto 41%):

| | spellF1 |
|---|---|
| Sin Smart Tagging | 0.750 |
| Ancla: el primer reinicio `D` → equipo que el modelo predice en la ventana de una fase pasa al equipo de la fase | 0.746 |
| Features (dentro de qué fase, de qué equipo y tipo, segundos desde su inicio) | 0.752 |
| Features + ancla | 0.748 |

Todo dentro del ruido, y el ancla cambia menos de 100 fotogramas por partido: **el modelo ya
acierta el equipo de los saques** (ve quién está junto al balón parado), así que la etiqueta solo
puede estropear con sus falsos positivos. Ninguna columna `st_*` entra en el top 10 del modelo.
Se queda como opción para repetir con las fases reales el 20 oct (las simuladas no reproducen
la duración ni los errores de las oficiales), no por defecto.

### Modelo híbrido (balón cuando se ve, lo demás cuando no): peor (13 oct)
`modelo_posesion.py --hibrido` (`ModeloHibrido`): dos especialistas mezclados fotograma a
fotograma, uno con todas las features entrenado y aplicado donde el balón se ve, y otro **sin
ninguna feature del balón** (solo colectivas, velocidad de los jugadores, cuánto lleva sin verse)
aplicado donde no se ve; luego el mismo Viterbi. En condiciones oficiales: **0.750 → 0.661**.
LightGBM sin suavizar pasa de 711 a 1.189 posesiones (hay 351): el especialista sin balón pierde
incluso la interpolación, que en los huecos cortos (la mayoría) es casi exacta, y el relevo entre
dos modelos mete discontinuidades que el Viterbi no cierra. El modelo único ya hace ese reparto
por dentro: tiene `balon_nan` y los fotogramas sin balón hacia atrás y hacia delante como
features, y parte por ahí cuando le conviene.

También probado el híbrido solo en huecos largos (`--hibrido-hueco N`: el especialista sin balón
solo entra en huecos de más de N fotogramas; en los cortos manda el modelo completo con el balón
interpolado):

| Especialista sin balón en... | spellF1 | posesiones LightGBM sin suavizar |
|---|---|---|
| Ningún hueco (modelo único actual) | **0.750** | 711 |
| Huecos > 3 s | 0.684 | 892 |
| Huecos > 1 s | 0.669 | 985 |
| Todos los huecos | 0.661 | 1.189 |

Cuanto más se usa el especialista, peor, también en los huecos largos: el modelo único ahí
sigue sacando partido del balón interpolado (dice por dónde anda el juego aunque no sea exacto)
y de las distancias al jugador más cercano, y el especialista sin nada de eso parpadea. Descartado.

### Estimar el balón desde el movimiento de los equipos (`balon_estimado.py`)
Features colectivas por fotograma (centroides y su velocidad, líneas, dispersión, centroide
ponderado por velocidad, los 3 jugadores más rápidos, "punto caliente", ventanas ±0,4 / 1 s) más
la historia del balón (última y siguiente posición vista, velocidad, extrapolación,
interpolación). Dos LightGBM aprenden la corrección sobre la interpolación. Error medio en
metros sobre el balón oculto (leave-one-out, 40% oculto):

| Método | todos | hueco ≤ 1 s | 1–3 s | 3–10 s | > 10 s |
|---|---|---|---|---|---|
| Última posición vista | 12.8 | 2.4 | 6.4 | 14.0 | 25.5 |
| Interpolación lineal | 6.1 | **0.3** | **1.8** | 6.4 | 14.9 |
| **Modelo (interpolación + corrección colectiva)** | **5.1** | 0.9 | 1.8 | **5.2** | **12.1** |

En huecos cortos la interpolación basta; el movimiento colectivo ayuda a partir de 3 s. Las
features que más usa son la interpolación, la duración del hueco y la velocidad del balón al
desaparecer, seguidas de la dispersión y la diferencia entre centroides.

### Dónde está el error ahora (`analiza_errores.py`, verdad oficial, sin ocultar balón)
| Categoría (fotogramas) | G1 | G2 | G3 |
|---|---|---|---|
| Sin balón trackeado | 1653 | 8747 | 4210 |
| A ±10 fotogramas de un cambio de posesión | 1013 | 762 | 1000 |
| Nadie a menos de 3 m | 1229 | 681 | 225 |
| Balón en vuelo (> 10 m/s) | 346 | 52 | 59 |
| Disputa (los dos equipos a ≤ 2 m) | 154 | 198 | 191 |
| Otro | 1369 | 559 | 915 |
| **Total** | 5764 (4.0%) | 10999 (7.8%) | 6600 (4.6%) |

Con la definición oficial, el error dominante pasa a ser el **balón no trackeado con el juego en
marcha** (el modelo lo toma por parada y dice `D`), justo lo que el 41% oficial va a multiplicar.
Es el frente de trabajo. Las rachas de error más largas (hasta 2 min) son tramos sin balón que
los eventos no marcan como parada; habrá que ver con la verdad pública oficial qué hace el
tracking corregido ahí.

### Con el bundle real: paradas con balón visible en vez de ocultar el balón (22 oct)
En el ATD oficial (agregados) el balón ausente es casi todo parada: con los jugadores casi
quietos (< 1,5 m/s de mediana) en el 87% de esos fotogramas, y ~13% del balón visible también
es parada. La simulación anterior (`--enmascarar 0.41`: ocultar el balón en juego) enseñaba lo
contrario de lo que hacía falta. Ahora `balon_estimado.mostrar_balon_en_paradas` deja el balón
visible y quieto en el punto del saque al final de un 15% de las paradas
(`--paradas-visibles 0.15`), y `evaluacion.quitar_paradas_cortas` elimina las paradas de menos
de `min_d` fotogramas entre dos posesiones (elegido fuera de muestra: 3 s). Leave-one-out sin
identidades, el mismo examen para las dos variantes (`--prueba-enmascarar 0
--prueba-paradas-visibles {0.05,0.15,0.30}`):

| Paradas visibles en el examen | anterior (`--enmascarar 0.41`) spellF1 / macroF1 | **nueva (`--enmascarar 0 --paradas-visibles 0.15`)** |
|---|---|---|
| 5% | 0.847 / 0.932 | 0.850 / 0.935 |
| 15% | 0.818 / 0.909 | **0.845 / 0.933** |
| 30% | 0.776 / 0.870 | **0.839 / 0.928** |

Es la configuración por defecto desde el 22 oct.

## Entrega
Un solo comando regenera el archivo exacto desde cero (entrena con los sample data,
predice el bundle oficial, lo renombra a `<equipo>.csv`, pasa el validador oficial y
muestra el SHA-256):
```powershell
.\reproducir.ps1 -Equipo <equipo> -Oficial ..\datos\oficiales\data     # Windows
./reproducir.sh <equipo> ../datos/oficiales/data                         # Linux / macOS
```
**Es determinista**: LightGBM va con `random_state=0`, `deterministic=True` y
`force_col_wise=True`, y toda la simulación (balón oculto, slots sin identidad) usa
`numpy.random.default_rng` con semilla fija. Comprobado el 13 oct: dos entrenamientos
completos seguidos dan los mismos parámetros de regla y suavizado, árboles idénticos
(`model_to_string` igual) y un CSV con el mismo SHA-256; sobre un partido, también con 1 y
4 hilos. Las reglas piden que el repo público regenere el archivo entregado: esto es ese
comando. (Dos entrenamientos a la vez agotan los 16 GB de RAM: de uno en uno.)

Paso a paso:
```powershell
# entrena con los tres sample data en las condiciones oficiales (balón nativo + 15% de paradas con balón visible, ruido de detección del balón, interpolación, sin identidades) y guarda el modelo (~15 min)
.\.venv\Scripts\python.exe -I entregar.py entrenar --data ..\datos\sample-data\data --modelo modelos\posesion.pkl --enmascarar 0 --paradas-visibles 0.15 --balon interpolar --sin-identidades --ruido-balon 0.8 30 0.06
# predice el bundle oficial (carpeta data\ con tactical\BAR-ATH_tactical_atd.xml …), pone D antes del saque inicial de cada parte (Smart Tagging) y escribe tactical.csv con frame,label (199.319 filas)
.\.venv\Scripts\python.exe -I entregar.py predecir --data ..\datos\oficiales\data --juegos tactical --modelo modelos\posesion.pkl --out entregas
# valida el archivo exacto que se va a subir con el script oficial (renombrado a <equipo>.csv)
.\.venv\Scripts\python.exe ..\datos\mtah-2027-oficial\evaluation\validate_submission.py entregas\<equipo>.csv
```
El modelo guardado lleva dentro los parámetros de la regla, `p_quedarse`, la duración mínima
de posesión, el estimador de balón (si se entrenó) y el orden de columnas. El formato del CSV
se define en `datos.FORMATO_ENTREGA` (ya es el oficial: `frame,label`, H/A/D/X).

### El cargador de los formatos oficiales (listo el 13 oct, sin datos reales todavía)
`datos.py` reconoce el bundle oficial por el `*_atd.xml` de la carpeta (`detectar_formato` →
`"oficial"`; `listar_juegos` ve la subcarpeta `tactical/`) y sigue exactamente lo que hace el
`starter/` del repo oficial:
- **ATD**: kloppy `metrica.load_tracking_epts`, el mismo camino que el Game 3 de muestra. `Frame`
  = `frame_id − 1` = fotograma del video = fila de la entrega. Los slots sin identidad entran como
  jugadores `H_<id>` / `A_<id>`; nada del pipeline usa la identidad. Los fotogramas fuera de los
  periodos (antes del saque, descanso, final) llevan `Period 0` y se entregan como `D`.
- **Smart Tagging**: `cargar_smart_tagging()` lee el XML SportsCode (fotograma = `round(s × 25)`,
  `Team` → H si contiene "Barcelona", A si es otro club, sin equipo si N/A; `Type` → los tipos
  de `smart_tagging.py`). `entregar.py predecir --ancla-saques` aplica el ancla con las fases
  reales (opcional: con las simuladas no aportó).
- **Homografía**: `cargar_homografia()` (frame, h00…h22, reprojection_error, valid). No la usa el
  modelo.
- **Exportación**: `exportar(..., n_frames=199319)` garantiza las filas `0…199318` en orden; un
  fotograma sin registro de tracking saldría como `D`.

Probado de punta a punta con un bundle sintético (el Game 3 de muestra renombrado al formato
oficial más un Smart Tagging y una homografía de juguete): `entregar.py predecir` produce un
archivo que **pasa `validate_submission.py`**. Lo que no se puede comprobar hasta el 20 oct: que
kloppy lea el metadata oficial de 46 slots igual que el de muestra, que los periodos del
metadata empiecen en el saque, y la alineación con los 5 min públicos (`run_evaluation.py`).

## La especificación oficial (12 oct 2026)
El repositorio oficial https://github.com/metrica-sports/more-than-a-hack-2027 (clonado en
`../datos/mtah-2027-oficial`) ya publica las reglas, el evaluador y 5 minutos de verdad. Responde
casi todas las preguntas que teníamos abiertas, y varias respuestas cambian el plan:

| Pregunta | Respuesta oficial | Qué cambia para nosotros |
|---|---|---|
| Formato del tracking | EPTS (`_atd.xml` + `_atd.txt`), kloppy lo lee. **Sin identidades de jugador**: 46 slots por equipo que aparecen, desaparecen y se intercambian | Quitar la heurística del portero para la dirección de ataque; features colectivas, no por jugador |
| Eventos | Smart Tagging en XML SportsCode: fases (Build Up, Progression, Set Pieces…) con equipo y parte, no pases ni recuperaciones | `verdad_desde_eventos()` no sirve con los datos oficiales; el Smart Tagging es un prior, no una verdad |
| Formato de entrega | `frame,label`, 199.319 filas, etiquetas `H` / `A` / `D` / `X`, nombre `<equipo>.csv`, validado con `validate_submission.py` | Ajustar `FORMATO_ENTREGA`; `X` fuera de las dos partes |
| Balón parado | `D` = balón muerto. Nadie pide quién "posee" un balón muerto | Coincide con nuestro Ninguno |
| ¿Qué es "Ninguno" en juego? | No existe: con el balón en juego siempre posee un equipo; balón suelto = el último que la tenía; cambia solo con control del rival | Nuestra verdad marcaba sin etiqueta los balones perdidos; hay que alinearla |
| Métrica | **`spellF1`**: posesiones enteras emparejadas por IoU temporal. `macroF1` por fotograma desempata. Floor: 0,2433 | Nuestro 98,6% de exactitud no predice el ranking. Medir `spellF1` desde ya |
| Etiquetas del partido | Ninguna para entrenar ni para ajustar: los 5 min públicos (17 posesiones) son solo para comprobar el pipeline y el resto es privado | Entrenar con sample data; los 5 min solo para comprobar el pipeline, nunca para entrenar, elegir parámetros ni elegir entre versiones |
| ¿Offline? | Sí: un CSV del partido completo | Las ventanas futuras son válidas |
| Envíos | Un formulario, una respuesta por equipo (editable), sin leaderboard ni feedback. Cierre 17 nov | Hay que confiar en nuestra validación interna |
| Coordenadas | 0–1 normalizadas, campo 105,3 × 68; homografía solo para proyectar a video | Como los sample data |
| ¿Pierde el balón en juego? | **Sí: falta en el 41% de los fotogramas evaluados**, sobre todo en los interesantes (áreas, duelos, entradas) | Lo contrario de los sample data. Hacen falta features sin balón |
| Deliverables | CSV + **repositorio público** que regenere el mismo archivo (sin datos) + documento técnico de pocas páginas | Hacer público el repo al entregar; nada de datos dentro |
| LLMs | Permitidos, pero **prohibido subir datos del partido o extractos a LLMs en la nube** | Claude Code no debe leer los archivos de datos oficiales; solo código y agregados |

Sigue pendiente del equipo: inscribirse con el mismo nombre de equipo, cuenta de Google para el
formulario, y reproducir los resultados en cada máquina.

## Al llegar los datos oficiales (20 oct)
1. Cargar con `datos.py` (ya preparado para EPTS con slots, Smart Tagging XML y homografía CSV, ver "El cargador de los formatos oficiales") y comprobar la alineación con los 5 min públicos (`run_evaluation.py`).
2. Primer archivo válido la primera semana: `validate_submission.py` sobre el CSV exacto.
3. Verdad de entrenamiento: sample data con la definición oficial (en juego siempre posee alguien; cambio solo con control).
4. Features sin balón para el 41%: movimiento colectivo, última posición del balón, Smart Tagging como prior de equipo y de saque.
5. Optimizar `spellF1`, no exactitud: suavizado que no parta posesiones, cambios de posesión bien situados (la `X` en huecos largos ya está probada y descartada, ver arriba).

## Siguientes mejoras del modelo
- Balón ausente: ~~medir el estimador con cross-fitting; probar `X` en huecos largos; features colectivas directamente en el modelo~~ (los tres probados el 12–13 oct: solo las features colectivas ayudan, van por defecto).
- Simular también el tracking sin identidades (46 slots con intercambios) y la pérdida de jugadores.
- Smart Tagging `SET PIECES` como ancla de saques (empieza posesión, termina `D`); fases con equipo como prior.
- Decidir mejor los cambios de posesión (control vs. rebote): features del jugador poseedor (velocidad jugador vs. balón, orientación).
- ~~Revisar las rachas de error largas sin balón contra la verdad pública oficial~~ No: la verdad pública es solo para comprobar el pipeline; el análisis de errores se hace en los sample data.
