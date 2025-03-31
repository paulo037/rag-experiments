#!/bin/bash

# Install the package in development mode
pip install -e .

# Print success message
echo "Installation complete!"
echo "You can now run the demo with: python3 demo.py"
echo "Or run the tests with: python3 -m unittest discover tests" 