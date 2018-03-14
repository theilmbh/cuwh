#include <iostream>
#include <math.h>
#include "cuwh.h"

using namespace std;

__device__ State state_add(const State &s1, const State &s2)
{
    State s;
    s.r = s1.r + s2.r;
    s.theta = s1.theta + s2.theta;
    s.phi = s1.phi + s2.phi;

    s.pr = s1.pr + s2.pr;
    s.ptheta = s1.ptheta + s2.ptheta;
    s.b = s1.b + s2.b;
    s.Bsq = s1.Bsq + s2.Bsq;
    return s;
}

__device__ State state_mul(double g, const State &s)
{
    State rs;
    rs.r = g*s.r;
    rs.theta = g*s.theta;
    rs.phi = g*s.phi;
    rs.pr = g*s.pr;
    rs.ptheta = g*s.ptheta;
    rs.b = g*s.b;
    rs.Bsq = g*s.Bsq;
    return rs;
}

__device__ double l(double r, double rhosq)
{
    
    return sqrt(rhosq + pow(r, 2));
}

__device__ double l_dneg(double r, double a, double rho, double M)
{
    if (abs(r) < a) {
        return rho;
    }

    double x = (2*(abs(r) - a)) / (pi*M);
    return rho + M*(x*atan(x) - log(1+pow(x,2))/2);
}

__device__ double dldr_dneg(double r, double a, double rho, double M)
{
    if (abs(r) < a) {
        return 0.0;
    }

    double dldr = (2/pi)*atan(2*(abs(r)-a)/(pi*M));
    if (r < 0) {
        dldr *= -1;
    }
    return dldr;
}

__device__ double dldr(double r, double rhosq)
{
    return r/l(r, rhosq) ;
}

__device__ State rhs(const State &s, double a, double rho, double M)
{

    State ds;
    double rsq = pow(l_dneg(s.r, a, rho, M), 2);

    ds.r = s.pr;
    ds.theta = s.ptheta / rsq;
    ds.phi = s.b / (rsq*pow(sin(s.theta), 2));
    ds.pr = s.Bsq*(dldr_dneg(s.r, a, rho, M) / (pow(l_dneg(s.r, a, rho, M), 3)));
    ds.ptheta = (pow(s.b, 2)/rsq) * cos(s.theta)/pow(sin(s.theta), 3);
    ds.b = 0.0;
    ds.Bsq = 0.0;

    return ds;
}

__global__ void rk4_step(int N, State *states, double a, double rho, double M, double h)
{
    uint indx = blockIdx.x*blockDim.x + threadIdx.x;

    State s = states[indx];
    State y = s;

    State k1 = rhs(s, a, rho, M);
    s = state_add(y, state_mul(h/2, k1));
    State k2 = rhs(s, a, rho, M);
    s = state_add(y, state_mul(h/2, k2));
    State k3 = rhs(s, a, rho, M);
    s = state_add(y, state_mul(h, k3));
    State k4 = rhs(s, a, rho, M);

    s = state_add(y, state_mul(h/6, state_add(k4, state_add(state_mul(2.0, state_add(k2, k3)), k1))));
    states[indx] = s;
}

void curk4(int NA, int NB, int N, State *states, double a, double rho, double M, double h)
{
    rk4_step<<<NB, NA>>>(N, states, a, rho, M, h);
    cudaDeviceSynchronize();
}


int compute_wh(State *ics, int Nx, int Ny, double a, double rho, double M)
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

    int blockSize = 4096;
    //int numBlocks = 512;
    //blockSize = N / numBlocks;
    int numBlocks = N / blockSize;
    numBlocks = 1;
    blockSize = N;
    blockSize = Nx;
    numBlocks = (N/blockSize);

    // Integrate
    double dt = 1e-2;
    double t = 0.0;
    double tend = 50.0;
    int k = 0;
    while(t < tend)
    {
        curk4(blockSize, numBlocks, N, states_device, a, rho, M, dt);
        t += dt;
        k += 1;

        // if((k%1000) == 0)
        // {
        //     cout << "Time: " << k*dt << endl;
        // }
    }

    // Retrieve Results
    err = cudaMemcpy(ics, states_device, N*sizeof(State), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cout << "Retrieval error" << endl;
        return -1;
    }
    cudaFree(states_device);
    return 0;

}