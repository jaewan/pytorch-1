#! /bin/bash

pushd ../../
pip uninstall torch
python setup.py clean
python setup.py develop && python -c "import torch"
popd
