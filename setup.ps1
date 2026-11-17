# Instala dependencias y descarga los datos de muestra de Metrica (fuera del código).
# Equivalente de setup.sh para Windows PowerShell.
$ErrorActionPreference = "Continue"   # ver reproducir.ps1; los errores se comprueban con $LASTEXITCODE
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) { python -m venv .venv }
if (-not (Test-Path .\.venv\Scripts\python.exe)) { throw "no se pudo crear .venv: instalen Python 3.11 y agreguenlo al PATH" }
$ver = & .\.venv\Scripts\python.exe -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($ver -ne "3.11") {
    Write-Warning "El entorno de referencia es Python 3.11 (probado con 3.11.15); este venv usa $ver y el archivo regenerado puede no ser identico."
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "error en pip install -r requirements.txt" }

New-Item -ItemType Directory -Force ..\datos | Out-Null
if (-not (Test-Path ..\datos\sample-data)) {
    git clone --depth 1 https://github.com/metrica-sports/sample-data.git ..\datos\sample-data
    if ($LASTEXITCODE -ne 0) { throw "error al clonar sample-data" }
}
$commit = git -C ..\datos\sample-data rev-parse --short=7 HEAD
if ($commit -ne "e706dd5") {
    Write-Warning "Los datos de muestra estan en el commit $commit; la entrega se genero con e706dd5."
}

Write-Host ""
Write-Host "Listo. Para correr:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -I baseline_posesion.py --data ..\datos\sample-data\data --out salida"
Write-Host "  python -I modelo_posesion.py   --data ..\datos\sample-data\data --out salida"
