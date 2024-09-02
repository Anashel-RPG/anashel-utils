@echo off

:: Check if Python is installed
where python >nul 2>nul || (
    echo Python is not installed. Please install Python 3.6+ to continue.
    exit /b 1
)

:: Create a virtual environment
echo Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment. Make sure Python 3.6+ is properly installed.
    exit /b 1
)

:: Activate the virtual environment
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate the virtual environment.
    exit /b 1
)

:: Upgrade pip
echo Upgrading pip...
pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo Failed to upgrade pip. Please check your Python and pip installation.
    exit /b 1
)

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies. Please ensure requirements.txt exists and is correctly formatted.
    exit /b 1
)

:: Run the main script
echo Running main.py...
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo An error occurred while running main.py. Please check the script for errors.
    pause
    exit /b 1
)

echo Script executed successfully!
pause