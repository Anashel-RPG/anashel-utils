@echo off
:: Redirect errors to a log file
set LOGFILE=script_log.txt
echo Logging errors to %LOGFILE%
echo If you run this script for the first time, it may take some time.
echo ------------------------------------------------------------------

:: Check if Python is installed
where python >nul 2>nul || (
    echo Python is not installed. Please install Python 3.6+ to continue.
    echo %date% %time% - Python not installed >> %LOGFILE%
    pause
    exit /b 1
)

:: Check if virtual environment exists
if exist venv\Scripts\activate (
    echo Virtual environment already exists. Skipping creation.
) else (
    echo Creating virtual environment...
    python -m venv venv >> %LOGFILE% 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment. Make sure Python 3.6+ is properly installed.
        echo %date% %time% - Failed to create virtual environment >> %LOGFILE%
        pause
        exit /b 1
    )
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate >> %LOGFILE% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate the virtual environment.
    echo %date% %time% - Failed to activate the virtual environment >> %LOGFILE%
    pause
    exit /b 1
)

:: Upgrade pip with the correct command
echo Upgrading pip...
python -m pip install --upgrade pip >> %LOGFILE% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Failed to upgrade pip. Please check your Python and pip installation.
    echo %date% %time% - Failed to upgrade pip >> %LOGFILE%
    pause
    exit /b 1
)

:: Check and install requirements
echo Checking installed packages...
pip list --format=freeze > installed_packages.txt
findstr /V /G:installed_packages.txt requirements.txt > missing_packages.txt
if exist missing_packages.txt (
    echo Installing missing requirements... Please wait!
    pip install -r requirements.txt >> %LOGFILE% 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install some dependencies. Please check the requirements file.
        echo %date% %time% - Failed to install dependencies >> %LOGFILE%
        pause
        exit /b 1
    )
) else (
    echo All requirements are already installed.
)

:: Run the main script
echo Running main.py...
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo An error occurred while running main.py. Please check the script for errors.
    echo %date% %time% - Error running main.py >> %LOGFILE%
    pause
    exit /b 1
)

echo Script executed successfully!
pause