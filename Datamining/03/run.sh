#!/usr/bin/env bash

rm -rf output
mkdir output

python run.py

python specs/test_practical3.py
