#!/bin/bash

# Specify the directory containing the .tar.gz files
DIR="."

# Loop through all .tar.gz files in the directory
for TAR_FILE in "$DIR"/*.tar.gz; do
    # Extract the .tar.gz file (this will create a folder)
    tar -xzvf "$TAR_FILE" -C "$DIR"
    
    # Find the extracted folder's name
    EXTRACTED_FOLDER=$(tar -tzf "$TAR_FILE" | head -1 | cut -f1 -d"/")
    EXTRACTED_PATH="$DIR/$EXTRACTED_FOLDER"
    
    # Move all contents out of the extracted folder to the main directory
    if [ -d "$EXTRACTED_PATH" ]; then
        mv "$EXTRACTED_PATH"/* "$DIR"
        
        # Remove the extracted folder
        rm -r "$EXTRACTED_PATH"
        echo "Processed $TAR_FILE: contents moved, folder deleted."
    else
        echo "No folder found for $TAR_FILE. Skipping..."
    fi

    # Delete the original .tar.gz file
    rm "$TAR_FILE"
    echo "Deleted $TAR_FILE after extraction."
done

echo "All .tar.gz files processed."
