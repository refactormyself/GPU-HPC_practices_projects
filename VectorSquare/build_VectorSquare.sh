#!/usr/bin/env/ bash

mkdir -p build
nvcc VectorSquare.cu -o ./build/vector_square && ./build/vector_square