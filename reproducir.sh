#!/usr/bin/env bash
# Regenera la entrega de la fase 1 desde cero: entrena el modelo con los sample
# data de Metrica, predice el bundle oficial y valida el CSV. Determinista: el
# mismo archivo (mismo SHA-256) en cada ejecución y con distinto número de hilos.
#
#   ./reproducir.sh <nombre_del_equipo> [carpeta_del_bundle_oficial]
#
# Los datos no están en el repo: los sample data los descarga setup.sh y el
# bundle oficial lo entrega la organización (carpeta con tactical/*_atd.xml).
# En Windows usar reproducir.ps1.
set -euo pipefail
cd "$(dirname "$0")"
EQUIPO="${1:-Getche}"
OFICIAL="${2:-../datos/oficiales/data}"
MUESTRA="../datos/sample-data/data"
VALIDADOR="../datos/mtah-2027-oficial/evaluation/validate_submission.py"
export PYTHONIOENCODING=utf8
PY=".venv/bin/python"

[ -x "$PY" ] && [ -d "$MUESTRA" ] || bash ./setup.sh

echo "1/3 Entrenando el modelo con los sample data (condiciones oficiales, ~15 min)..."
"$PY" -I entregar.py entrenar --data "$MUESTRA" --modelo modelos/posesion.pkl \
    --enmascarar 0 --paradas-visibles 0.15 --balon interpolar --sin-identidades \
    --ruido-balon 0.8 30 0.06

echo "2/3 Prediciendo el bundle oficial..."
# Solo la grabación táctica: es la referencia de la entrega (broadcast va en otra línea de tiempo)
"$PY" -I entregar.py predecir --data "$OFICIAL" --juegos tactical --modelo modelos/posesion.pkl --out entregas
cp -f entregas/tactical.csv "entregas/$EQUIPO.csv"

echo "3/3 Validando entregas/$EQUIPO.csv..."
if [ -f "$VALIDADOR" ]; then
    "$PY" "$VALIDADOR" "entregas/$EQUIPO.csv"
else
    echo "  (sin validador oficial en $VALIDADOR; clona metrica-sports/more-than-a-hack-2027 en ../datos/mtah-2027-oficial)"
fi
echo
echo "Entrega: entregas/$EQUIPO.csv"
if command -v sha256sum >/dev/null 2>&1; then
    SHA=$(sha256sum "entregas/$EQUIPO.csv" | cut -d' ' -f1)
else                                   # macOS
    SHA=$(shasum -a 256 "entregas/$EQUIPO.csv" | cut -d' ' -f1)
fi
echo "SHA-256: $SHA"
