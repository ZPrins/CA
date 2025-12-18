# PowerShell script to rebuild the Docker image
Write-Host "Building Docker image 'simulation-app'..." -ForegroundColor Cyan
docker build -t simulation-app .

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful!" -ForegroundColor Green
    Write-Host "You can run the container using: docker run -p 5000:5000 simulation-app" -ForegroundColor Yellow
} else {
    Write-Host "Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
