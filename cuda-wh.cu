#include <iostream>
#include <math.h>

using namespace std;

typedef struct {
	float r;
	float theta;
	float phi;
	float pr;
	float ptheta;
	float pphi;
	float b;
	float Bsq;
} State;        // Dynamical System State Vector


__device__ State state_add(const State &s1, const State &s2)
{
	State s;
	s.r = s1.r + s2.r;
	s.theta = s1.theta + s2.theta;
	s.phi = s1.phi + s2.phi;

	s.pr = s1.pr + s2.pr;
	s.ptheta = s1.ptheta + s2.ptheta;
	s.pphi = s1.pphi + s2.pphi;
	return s;
}

__device__ State state_mul(float g, const State &s)
{
	State rs;
	rs.r = g*s.theta;
	rs.theta = g*s.theta;
	rs.phi = g*s.phi;
	rs.pr = g*s.pr;
	rs.ptheta = g*s.ptheta;
	rs.pphi = g*s.pphi;
	return rs;
}

__device__ float l(float r)
{
	return r;
}

__device__ float dldr(float r)
{
	return 1.0;
}

__device__ State rhs(const State &s, float gamma)
{

	State ds;
	float rsq = pow(l(s.r), 2);

	ds.r = s.pr;
	ds.theta = s.ptheta / rsq;
	ds.phi = s.b / (rsq*pow(sin(s.theta), 2));
	ds.pr = s.Bsq*(dldr(s.r) / (rsq*l(s.r)));
	ds.ptheta = (pow(s.b, 2)/rsq) * cos(s.theta)/pow(sin(s.theta), 3);
	ds.pphi = 0.0;
	return ds;
}

__global__ void rk4_step(int N, State *states, float gamma, float h)
{
	uint indx = blockIdx.x*blockDim.x + threadIdx.x;

	State s = states[indx];
	const State y = s;

	State k1 = rhs(s, gamma);
	s = state_add(y, state_mul(h/2, k1));
	State k2 = rhs(s, gamma);
	s = state_add(y, state_mul(h/2, k2));
	State k3 = rhs(s, gamma);
	s = state_add(y, state_mul(h, k3));
	State k4 = rhs(s, gamma);

	s = state_add(y, state_mul(h/6, state_add(k4, state_add(state_mul(2, state_add(k2, k3)), k1))));
	states[indx] = s;
}

int main(void)
{

	int Nx = 512;
	int Ny = 512;  // Pixes

	int N = Nx*Ny;  // total pixels in image.

	// Allocate Host array for initial conditions:
	State *states_host = (State *)malloc(N*sizeof(State));
	State temps[16];

	// Allocate Device array for initial conditions
	State *states_device;
	cudaError_t err = cudaMalloc((void**) &states_device, N*sizeof(State));
	if (err != cudaSuccess)
	{
		cout << "Allocation error" << endl;
	}

	// Setup Initial conditions
	int i,j;
	float pi = 3.14159;
	float deg2rad = pi/180;

	float thetaFOV = 20;
	float phiFOV = 20;

	float mintheta = pi/2 - deg2rad*thetaFOV/2;
	float maxtheta = pi/2 + deg2rad*thetaFOV/2;
	float minphi = -deg2rad*phiFOV/2;
	float maxphi = deg2rad*phiFOV/2;

	float thetacs, phics;
	float nx, ny, nz;

	float cam_l = -10.0;
	float cam_r = l(cam_l);
	float cam_phi = 0.0;
	float cam_theta = pi/2;

	for(i=0; i<Nx; i++)  // phi
	{
		for(j=0; j<Ny; j++) // theta
		{
			thetacs = mintheta + (((float) j)/Ny)*(maxtheta - mintheta);
			phics = minphi + (((float) i)/Nx)*(maxphi - minphi);
			nx = sin(thetacs)*cos(phics);
			ny = sin(thetacs)*sin(phics);
			nz = -1.0*cos(thetacs);

			states_host[i*Nx +j].r = cam_l;
			states_host[i*Nx +j].theta = cam_theta;
			states_host[i*Nx +j].phi = cam_phi;
			states_host[i*Nx +j].pr = nx;
			states_host[i*Nx +j].ptheta = cam_r*nz;
			states_host[i*Nx +j].pphi = cam_r*sin(cam_theta)*ny;
			states_host[i*Nx +j].b = cam_r*sin(cam_theta)*ny;
			states_host[i*Nx +j].Bsq = pow(cam_r, 2)*(pow(nz,2) + pow(ny,2));
		}
	}

	// Copy initial conditions to device
	err = cudaMemcpy(states_device, &states_host[0], N*sizeof(State), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		cout << "Allocation error" << endl;
	}


	int blockSize = 256;
	int numBlocks = N / blockSize;

	// Integrate
	float dt = 1e-3;
	float t = 0.0;
	float tend = 100.0;
	int k = 0;
	while(t < tend)
	{
		rk4_step<<<blockSize, numBlocks>>>(N, states_device, 1.0, dt);
		t += dt;
		k +=1;

		if((k%1000) == 0)
		{
			err = cudaMemcpy(&temps[0], states_device, 16*sizeof(State), cudaMemcpyDeviceToHost);
			if (err != cudaSuccess)
			{
				cout << "Allocation error" << endl;
			}

			for (int j=0; j< 1; j++)
			{
				cout << "UPDATE   " <<  temps[3].r << endl;
			}
		}

	}


}