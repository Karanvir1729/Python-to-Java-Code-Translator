# Check if Python is installed
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not found in PATH. Please install Python." -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install dependencies." -ForegroundColor Red
    exit 1
}

# Run the Streamlit app
Write-Host "Starting AVATAR Translation UI..." -ForegroundColor Green
python -m streamlit run app.py
