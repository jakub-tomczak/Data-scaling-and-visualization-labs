#!/bin/bash

# Usage: ./runner.sh myPythonScript.py

python $1 -svd custom -k 5 -f img/pink-floyd.jpg
python $1 -svd custom -k 30 -f img/pink-floyd.jpg
python $1 -svd custom -k 30 -f img/wittgenstein.jpg
python $1 -svd custom -k 70 -f img/clouds.jpg
python $1 -svd scikit -k 10 -f img/pink-floyd.jpg

