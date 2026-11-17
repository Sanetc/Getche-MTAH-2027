# Regenera la entrega de la fase 1 desde cero: entrena el modelo con los sample
# data de Metrica, predice el bundle oficial y valida el CSV. Determinista: el
# mismo archivo (mismo SHA-256) en cada ejecución y con distinto número de hilos.
#
#   .\reproducir.ps1 -Equipo <nombre_del_equipo> [-Oficial ..\datos\oficiales\data]
#
# Los datos no están en el repo: los sample data los descarga setup.ps1 y el
# bundle oficial lo entrega la organización (carpeta con tactical\*_atd.xml).
param(
    [string]$Equipo = "Getche",
    [string]$Oficial = "..\datos\oficiales\data",
    [string]$Muestra = "..\datos\sample-data\data",
    [string]$Validador = "..\datos\mtah-2027-oficial\evaluation\validate_submission.py"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf8"
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $py) -or -not (Test-Path $Muestra)) { & .\setup.ps1 }

Write-Host "1/3 Entrenando el modelo con los sample data (condiciones oficiales, ~15 min)..."
& $py -I entregar.py entrenar --data $Muestra --modelo modelos\posesion.pkl `
    --enmascarar 0 --paradas-visibles 0.15 --balon interpolar --sin-identidades `
    --ruido-balon 0.8 30 0.06
if ($LASTEXITCODE -ne 0) { throw "falló el entrenamiento" }

Write-Host "2/3 Prediciendo el bundle oficial..."
# Solo la grabación táctica: es la referencia de la entrega (broadcast va en otra línea de tiempo)
& $py -I entregar.py predecir --data $Oficial --juegos tactical --modelo modelos\posesion.pkl --out entregas
if ($LASTEXITCODE -ne 0) { throw "falló la predicción" }
Copy-Item "entregas\tactical.csv" "entregas\$Equipo.csv" -Force

Write-Host "3/3 Validando entregas\$Equipo.csv..."
if (Test-Path $Validador) {
    & $py $Validador "entregas\$Equipo.csv"
    if ($LASTEXITCODE -ne 0) { throw "el validador oficial rechaza el archivo" }
} else {
    Write-Host "  (sin validador oficial en $Validador; clona metrica-sports/more-than-a-hack-2027 en ..\datos\mtah-2027-oficial)"
}
$hash = (Get-FileHash "entregas\$Equipo.csv" -Algorithm SHA256).Hash.ToLower()
Write-Host ""
Write-Host "Entrega: entregas\$Equipo.csv"
Write-Host "SHA-256: $hash"
