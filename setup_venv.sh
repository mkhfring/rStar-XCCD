#!/bin/bash
module load python/3.11 cuda/12.2 cudnn

rm -rf ~/.local/share/virtualenv
virtualenv --no-download venv
source venv/bin/activate
