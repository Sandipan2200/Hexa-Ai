@echo off
REM Hexa Voice Assistant Launcher
REM Usage: run_hexa.bat [--check | --text | --help]

set "PROJECT_DIR=%~dp0.."
set "VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo ERROR: Virtual environment not found at %VENV_PYTHON%
    echo Run: py -3.12 -m venv .venv && .venv\Scripts\pip.exe install -r requirements.txt
    exit /b 1
)

cd /d "%PROJECT_DIR%"
set "PYTHONPATH=%PROJECT_DIR%\hexa"

if "%~1"=="" (
    echo Starting Hexa service...
    "%VENV_PYTHON%" -m hexa
) else if "%~1"=="--check" (
    echo Running health check...
    "%VENV_PYTHON%" -m hexa --check
) else if "%~1"=="--text" (
    echo Starting Hexa text mode...
    "%VENV_PYTHON%" -m hexa --text
) else if "%~1"=="--help" (
    echo Hexa Voice Assistant Launcher
    echo.
    echo Usage: run_hexa.bat [--check | --text | --help]
    echo.
    echo   (no args)   Start full service (API + voice)
    echo   --check     Run health checks
    echo   --text      Interactive text mode
    echo   --help      Show this help
) else (
    echo Unknown argument: %~1
    echo Use --help for usage
    exit /b 1
)