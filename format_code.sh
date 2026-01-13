#!/bin/bash

for folder in include applications; do
    find $folder -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i
done