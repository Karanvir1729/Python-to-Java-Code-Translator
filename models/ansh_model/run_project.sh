#!/bin/bash

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "python3 could not be found. Please install Python."
    exit 1
fi

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    exit 1
fi

# Run the Streamlit app
echo "Starting AVATAR Translation UI..."
python3 -m streamlit run app.py
