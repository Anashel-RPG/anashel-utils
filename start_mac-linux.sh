#!/bin/bash

# Check if Python is installed
if ! command -v python3 &>/dev/null; then
    echo "Python is not installed. Please install Python 3.6+ to continue."
    exit 1
fi

# Create and activate a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment. Make sure Python 3.6+ is properly installed."
    exit 1
fi

# Activate the virtual environment
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate the virtual environment."
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Failed to upgrade pip. Please check your Python and pip installation."
    exit 1
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies. Please ensure requirements.txt exists and is correctly formatted."
    exit 1
fi

# Run the main script
echo "Running main.py..."
python main.py
if [ $? -ne 0 ]; then
    echo "An error occurred while running main.py. Please check the script for errors."
    exit 1
fi

echo "Script executed successfully!"