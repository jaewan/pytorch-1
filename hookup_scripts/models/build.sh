#! /bin/bash

pushd ../../
python setup.py develop && python -c "import torch"
popd
