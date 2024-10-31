#!/bin/bash

# Define the tar.gz file
TAR_FILE="bboxes_annotations.tar.gz"

# Check if the tar file exists
if [ ! -f "$TAR_FILE" ]; then
    echo "Error: $TAR_FILE not found!"
    exit 1
fi

# Extract the tar.gz file
tar -xzf "$TAR_FILE"

# Remove the tar.gz file
rm "$TAR_FILE"

# Find and extract any nested compressed files
for file in *; do
    case "$file" in
        *.tar.gz)
            echo "Extracting $file..."
            tar -xzf "$file"
            rm "$file"
            ;;
        *.tar)
            echo "Extracting $file..."
            tar -xf "$file"
            rm "$file"
            ;;
        *.zip)
            echo "Extracting $file..."
            unzip "$file"
            rm "$file"
            ;;
    esac
done

echo "Extraction and cleanup completed."

