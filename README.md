# cuwh
Implementation of the relativistic ray-tracing algorithm 
for wormhole spacetimes described in "Visualizing Interstellar's Wormhole" by James, von Tunzelmann, Franklin, and Thorne (arxiv:1502:03809)

Simple build:
- gcc -c cuda-wh.cpp
- nvcc -c kernels.cu
- g++ -o cuwh kernels.o cuda-wh.o `pkg-config --cflags --libs opencv` -L/usr/local/cuda/lib64 -lcuda -lcudart -lm

### Work in progress
Bezier curves for camera trajectory / orientation.  
Procedurally generated celestial sphere backgrounds.  
Motion blur