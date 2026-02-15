#!/bin/bash

# install torch, torchvision, fastapi[standard], python-multipart, ...
cd benchmark && python run_experiments.py && python visualize_results.py && cd ..