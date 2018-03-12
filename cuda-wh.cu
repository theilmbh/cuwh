#include <iostream>
#include <math.h>
#include <fstream>
#include "cuwh.h"

using namespace std;


State * get_frame_ics(int Nx, int Ny, double thetaFOV, double phiFOV,
                      double cam_l, double cam_phi, double cam_theta)
{

    int N = Nx*Ny;
    State *ret = (State *)malloc(N*sizeof(State));

    int i,j;
    double pi = 3.14159;
    double deg2rad = pi/180;

    double mintheta = pi/2 - deg2rad*thetaFOV/2;
    double maxtheta = pi/2 + deg2rad*thetaFOV/2;
    double minphi = -deg2rad*phiFOV/2;
    double maxphi = deg2rad*phiFOV/2;

    double thetacs, phics;
    double nx, ny, nz;

    for(i=0; i<Nx; i++)  // phi
    {
        for(j=0; j<Ny; j++) // theta
        {
            thetacs = mintheta + (((double) j)/Ny)*(maxtheta - mintheta);
            phics = minphi + (((double) i)/Nx)*(maxphi - minphi);
            nx = sin(thetacs)*cos(phics);
            ny = sin(thetacs)*sin(phics);
            nz = -1.0*cos(thetacs);

            ret[i*Nx +j].r = cam_l;
            ret[i*Nx +j].theta = cam_theta;
            ret[i*Nx +j].phi = cam_phi;
            ret[i*Nx +j].pr = nx;
            ret[i*Nx +j].ptheta = cam_r*nz;
            ret[i*Nx +j].b = cam_r*sin(cam_theta)*ny;
            ret[i*Nx +j].Bsq = pow(cam_r, 2)*(pow(nz,2) + pow(ny,2));
        }
    }

    return ret;

}

int compute_wh(State *ics, int Nx, int Ny)
{
    State *states_device;
    int N = Nx*Ny;
    cudaError_t err = cudaMalloc((void**) &states_device, N*sizeof(State));

    if (err != cudaSuccess) {
        cout << "CUDA Memory Allocation Error" << endl;
        return -1;
    }

    err = cudaMemcpy(states_device, &ics[0], N*sizeof(State), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << "CUDA Memory Copy Error" << endl;
        return -1;
    }

    int blockSize = Nx;
    int numBlocks = N / blockSize;

    // Integrate
    double dt = 1e-3;
    double t = 0.0;
    double tend = 100.0;
    int k = 0;
    while(t < tend)
    {
        curk4(blockSize, numblocks, N, states_device, 1.0, dt);
        t += dt;
        k += 1;

        if((k%1000) == 0)
        {
            cout << "Time: " << k*dt << endl;
        }
    }

    // Retrieve Results
    err = cudaMemcpy(states_host, states_device, N*sizeof(State), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cout << "Retrieval error" << endl;
        return -1;
    }
    cudaFree(states_device);
    return 0;

}

int main(void)
{

    int Nx = 512;
    int Ny = 512;  // Pixes

    int N = Nx*Ny;  // total pixels in image.

    // Allocate Device array for initial conditions
    State *states_device;
    cudaError_t err = cudaMalloc((void**) &states_device, N*sizeof(State));
    if (err != cudaSuccess)
    {
        cout << "Allocation error" << endl;
    }

    // Setup Initial conditions
    double thetaFOV = 20;
    double phiFOV = 20;

    double cam_l = -10.0;
    double cam_r = sqrt(1.0 + pow(cam_l, 2));
    double cam_phi = 0.0;
    double cam_theta = pi/2;

    states_host = get_frame_ics(Nx, Ny, thetaFOV, phiFOV, cam_l, cam_phi, cam_theta);
    int err = compute_wh(states_host, Nx, Ny);
    if (err < 0) {
        cout << "Computation Failed" << endl;
        return -1;
    }

    ofstream outfile("mapout.dat", ios::binary);
    outfile.write((char *)states_host, N*sizeof(State));
    outfile.close();

    free(states_host);
    return 0;

}
