#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    echo "Expecting: "
    echo "1) path to training config file,"
    echo "2) a colorname (in double quotes if more than one word),"
    echo "3) flag --no_plot (optional),"
    echo "4) flag --gpu (optional)."
    exit 1
fi


export PYTHONPATH=$(pwd)
pipenv run python color_generator/color_predictor.py $1 $2 $3 $4


