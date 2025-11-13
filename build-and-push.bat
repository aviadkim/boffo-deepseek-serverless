@echo off
echo ============================================================
echo BOFFO DeepSeek OCR - Docker Build and Push
echo ============================================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo Step 1: Docker is running - OK!
echo.

REM Prompt for Docker Hub username
set /p DOCKER_USERNAME="Enter your Docker Hub username (or press Enter to use 'aviadkim'): "
if "%DOCKER_USERNAME%"=="" set DOCKER_USERNAME=aviadkim

echo.
echo Using Docker Hub username: %DOCKER_USERNAME%
echo.

REM Login to Docker Hub
echo Step 2: Login to Docker Hub...
echo (Enter your Docker Hub password when prompted)
echo.
docker login -u %DOCKER_USERNAME%

if errorlevel 1 (
    echo.
    echo ERROR: Docker login failed!
    echo.
    echo If you don't have a Docker Hub account:
    echo 1. Go to https://hub.docker.com/signup
    echo 2. Create free account
    echo 3. Come back and run this script again
    pause
    exit /b 1
)

echo.
echo Step 3: Building Docker image (this takes 10-15 minutes)...
echo Downloading DeepSeek model (8GB)...
echo.
docker build -t %DOCKER_USERNAME%/boffo-deepseek-ocr:latest .

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo Step 4: Pushing to Docker Hub...
echo.
docker push %DOCKER_USERNAME%/boffo-deepseek-ocr:latest

if errorlevel 1 (
    echo.
    echo ERROR: Push failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! Docker image built and pushed!
echo ============================================================
echo.
echo Your Docker image: %DOCKER_USERNAME%/boffo-deepseek-ocr:latest
echo.
echo NEXT STEPS:
echo 1. Go to RunPod Templates
echo 2. Create new template with this image:
echo    %DOCKER_USERNAME%/boffo-deepseek-ocr:latest
echo 3. Update your endpoint to use this template
echo.
echo Press any key to exit...
pause >nul
