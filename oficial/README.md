# Evaluador oficial

`possession_metrics.py` es una copia sin modificar de
https://github.com/metrica-sports/more-than-a-hack-2027/blob/b429693/evaluation/possession_metrics.py
(licencia MIT, ver `LICENSE`). Define `spellF1` y `macroF1`, las métricas con las
que se puntúa la fase 1. Nuestro `evaluacion.py` lo envuelve para usarlo con las
etiquetas internas (Ninguno / Local / Visitante) y con los sample data, que no
tienen hueco de fotogramas entre partes.

Si la organización actualiza el archivo, copiarlo de nuevo y anotar aquí el commit.
