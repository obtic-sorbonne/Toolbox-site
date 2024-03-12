#!/bin/bash

packages=(
    "flask"
    "flask_wtf"
    "flask_babel"
    "bs4"
    "lxml"
    "scikit-learn"
    "pandas"
    "sem"
    "semtagger"
    "renard"
    "renard.pipeline"
    "flair"
)

for package in "${packages[@]}"; do
    pip install "$package"
done

