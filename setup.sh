#!/usr/bin/env bash
# Instala dependencias y descarga los datos de muestra de Metrica (fuera del código).
# En Windows usar setup.ps1. En macOS, LightGBM necesita libomp: brew install libomp
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
ver=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[ "$ver" = "3.11" ] || echo "AVISO: el entorno de referencia es Python 3.11 (probado con 3.11.15); este venv usa $ver y el archivo regenerado puede no ser idéntico." >&2
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p ../datos
[ -d ../datos/sample-data ] || git clone --depth 1 https://github.com/metrica-sports/sample-data.git ../datos/sample-data
commit=$(git -C ../datos/sample-data rev-parse --short=7 HEAD)
[ "$commit" = "e706dd5" ] || echo "AVISO: los datos de muestra están en el commit $commit; la entrega se generó con e706dd5." >&2
echo
echo "Listo. Para correr:"
echo "  source .venv/bin/activate"
echo "  python -I baseline_posesion.py --data ../datos/sample-data/data --out salida"
echo "  python -I modelo_posesion.py   --data ../datos/sample-data/data --out salida"
