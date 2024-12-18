#!/bin/bash

echo "Setting up the conda environment..."
conda env create -f environment.yaml

echo "Activating the environment..."
source activate amortized-decision-making

echo "Environment setup is complete."
