#!/bin/bash
# Shell script to rebuild the Docker image
echo "Building Docker image 'simulation-app'..."
docker build -t simulation-app .

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "You can run the container using: docker run -p 5000:5000 simulation-app"
else
    echo "Build failed!"
    exit 1
fi
