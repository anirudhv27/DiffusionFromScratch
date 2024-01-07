#!/bin/bash

# Specify the Google Drive URL and the target directory
GOOGLE_DRIVE_URL="https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
SCRIPT_DIR=$(dirname "$0")
TARGET_DIRECTORY="$SCRIPT_DIR/../../data"
ZIP_FILE="$TARGET_DIRECTORY/data_download.zip" # Name of the downloaded zip file

# Check if gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found, installing it now..."
    pip install gdown
fi

# Confirming gdown installation
if command -v gdown &> /dev/null
then
    echo "gdown is installed, proceeding with the download..."
    # Use gdown to download the file to the specified directory
    gdown "$GOOGLE_DRIVE_URL" -O "$ZIP_FILE"

    echo "Download complete, unzipping the file..."
    # Unzip the downloaded file
    unzip "$ZIP_FILE" -d "$TARGET_DIRECTORY"

    echo "Unzip complete, removing the original zip file..."
    # Remove the original zip file
    rm "$ZIP_FILE"
else
    echo "Error: gdown installation failed."
fi