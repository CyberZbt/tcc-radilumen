# Compila trabalho.tex com referencias ABNT (pdflatex + bibtex + pdflatex x2)
Set-Location $PSScriptRoot

Write-Host "1/4 pdflatex (1a passagem)..."
pdflatex -interaction=nonstopmode trabalho.tex | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Warning "pdflatex retornou codigo $LASTEXITCODE" }

Write-Host "2/4 bibtex..."
bibtex trabalho
if ($LASTEXITCODE -ne 0) { Write-Warning "bibtex retornou codigo $LASTEXITCODE" }

if (-not (Test-Path "trabalho.bbl") -or (Get-Item "trabalho.bbl").Length -lt 100) {
    Write-Error "trabalho.bbl vazio ou ausente. Abortando."
    exit 1
}

Write-Host "3/4 pdflatex (2a passagem)..."
pdflatex -interaction=nonstopmode trabalho.tex | Out-Null

Write-Host "4/4 pdflatex (3a passagem)..."
pdflatex -interaction=nonstopmode trabalho.tex | Out-Null

$undef = (Select-String -Path trabalho.log -Pattern "Citation.*undefined" -ErrorAction SilentlyContinue).Count
if ($undef -gt 0) {
    Write-Warning "Ainda ha $undef citacoes indefinidas."
    exit 1
}

if (-not (Select-String -Path trabalho.bbl -Pattern "\\bibitem" -Quiet)) {
    Write-Error "Bibliografia vazia em trabalho.bbl. Verifique referencias.bib."
    exit 1
}

Write-Host "OK: citacoes resolvidas."

if (Test-Path trabalho.pdf) {
    Write-Host "PDF gerado: $(Resolve-Path trabalho.pdf)"
}
