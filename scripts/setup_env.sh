#!/bin/bash
# scripts/setup_env.sh

# Sets up the development environment.
# - Creates virtual environment
# - Installs dependencies
# - Downloads models

echo "Setting up Glimpse3D environment..."
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
python scripts/download_models.py
echo "Setup complete."
