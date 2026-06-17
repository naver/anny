#!/bin/bash
# This script is used to compile the tutorials of the project.

# Retrieve file parent directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$DIR/src

# Build the tutorials from their jupytext py:percent sources.
# Each .py is converted to a throwaway notebook, executed, rendered to HTML, then removed.
rm -r docs/build
mkdir -p docs/build
for py in tutorials/*.py; do
    base=$(basename "$py" .py)
    nb="docs/build/$base.ipynb"
    jupytext --to ipynb "$py" -o "$nb"
    jupyter nbconvert --to html --execute "$nb" --output-dir=docs/build
    rm "$nb"
done

# Anonymize the documentation by replacing lines containing '/home' by 'some_path'
echo "Anonymizing documentation..."
find docs/build -type f -name "*.html" -exec sed -i '/\/home/c\some_path' {} +


