@echo off
REM Hexa Voice Assistant Launcher
REM Usage: run_hexa.bat [--check | --text | --voice | --service | --help]

set "HEXA_ROOT=D:\work - personal\Hexa Ai\hexa"
set "VENV_PYTHON=D:\work - personal\Hexa Ai\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo ERROR: Virtual environment not found at %VENV_PYTHON%
    echo Run: py -3.12 -m venv .venv && .venv\Scripts\pip.exe install -r requirements.txt
    exit /b 1
)

set "PYTHONPATH=%HEXA_ROOT%"

if "%~1"=="" (
    echo Starting Hexa service (API + voice)...
    "%VENV_PYTHON%" -m hexa --voice
) else (
    if "%~1"=="--check" (
        echo Running health check...
        "%VENV_PYTHON%" -m hexa --check
    ) else if "%~1"=="--text" (
        echo Starting Hexa text mode...
        "%VENV_PYTHON%" -m hexa --text
    ) else if "%~1"=="--voice" (
        echo Starting Hexa voice service (API + voice)...
        "%VENV_PYTHON%" -m hexa --voice
    ) else if "%~1"=="--service" (
        echo Starting Hexa API service only...
        "%VENV_PYTHON%" -m hexa --service
    ) else if "%~1"=="--help" (
        echo Hexa Voice Assistant Launcher
        echo.
        echo Usage: run_hexa.bat [--check ^| --text ^| --voice ^| --service ^| --help]
        echo.
        echo   (no args)   Start full service (API + voice)
        echo   --check     Run health checks
        echo   --text      Interactive text mode
        echo   --voice     Start full service with voice (default)
        echo   --service   Start API service only (no voice)
        echo   --help      Show this help
    ) else (
        echo Unknown argument: %~1
        echo Use --help for usage
        exit /b 1
    )
)