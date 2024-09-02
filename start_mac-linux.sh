#!/bin/bash

# Log file
LOGFILE="script_log.txt"
echo "Logging errors to $LOGFILE"
echo "If you run this script for the first time, it may take some time."
echo "------------------------------------------------------------------"

# Check if Python is installed
if ! command -v python3 &>/dev/null; then
    echo "Python is not installed. Please install Python 3.6+ to continue."
    echo "$(date) - Python not installed" >> $LOGFILE
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    echo "Creating virtual environment..."
    python3 -m venv venv >> $LOGFILE 2>&1
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Make sure Python 3.6+ is properly installed."
        echo "$(date) - Failed to create virtual environment" >> $LOGFILE
        read -n 1 -s -r -p "Press any key to exit..."
        exit 1
    fi
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate >> $LOGFILE 2>&1
if [ $? -ne 0 ]; then
    echo "Failed to activate the virtual environment."
    echo "$(date) - Failed to activate the virtual environment" >> $LOGFILE
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
fi

# Check current pip version and upgrade if needed
CURRENT_PIP_VERSION=$(pip --version | awk '{print $2}')
LATEST_PIP_VERSION=$(pip install --upgrade pip 2>&1 | grep 'from version' | awk '{print $6}')
if [ "$CURRENT_PIP_VERSION" != "$LATEST_PIP_VERSION" ]; then
    echo "Upgrading pip from $CURRENT_PIP_VERSION to $LATEST_PIP_VERSION..."
    pip install --upgrade pip >> $LOGFILE 2>&1
    if [ $? -ne 0 ]; then
        echo "Failed to upgrade pip. Please check your Python and pip installation."
        echo "$(date) - Failed to upgrade pip" >> $LOGFILE
        read -n 1 -s -r -p "Press any key to exit..."
        exit 1
    fi
else
    echo "Pip is already up-to-date."
fi

# Check and install requirements
echo "Checking and installing requirements..."
pip freeze > installed_packages.txt
comm -23 <(sort requirements.txt) <(sort installed_packages.txt) > missing_packages.txt
if [ -s missing_packages.txt ]; then
    echo "Installing missing requirements... Please wait!"
    pip install -r requirements.txt >> $LOGFILE 2>&1
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies. Please ensure requirements.txt exists and is correctly formatted."
        echo "$(date) - Failed to install dependencies" >> $LOGFILE
        read -n 1 -s -r -p "Press any key to exit..."
        exit 1
    fi
else
    echo "All requirements are already installed."
fi

# Run the main script
echo "Running main.py..."
python main.py
if [ $? -ne 0 ]; then
    echo "An error occurred while running main.py. Please check the script for errors."
    echo "$(date) - Error running main.py" >> $LOGFILE
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
fi

echo "Script executed successfully!"