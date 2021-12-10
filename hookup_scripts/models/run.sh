#! /bin/bash

./build.sh
rm ~/pytorchLog
PYTHON_JIT=0  python a.py
vi ~/pytorchLog
