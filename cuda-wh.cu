#include <iostream>
#include <math.h>

using namespace std;

typedef struct {
	float x;
	float y;
} State;        // Dynamical System State Vector


__device__ State state_add(const State &s1, const State &s2)
{
	State s;
	s.x = s1.x + s2.x;
	s.y = s1.y + s2.y;
	return s;
}

__device__ State state_mul(float g, const State &s)
{
	State rs;
	rs.x = g*s.x;
	rs.y = g*s.y;
	return rs;
}

__device__ State rhs(const State &s, float gamma)
{
	State ds;
	ds.x = s.y;
	ds.y = -gamma*s.x;
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
	for(i=0; i<Nx; i++)
	{
		for(j=0; j<Ny; j++)
		{
			states_host[i*Nx +j].x = ((float) i)/Nx;
			states_host[i*Nx+j].y = ((float) j)/Ny;
			//cout << states_host[i*Nx +j].x << endl;

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
				cout << "UPDATE   " <<  temps[3].x << endl;
			}
		}

	}


}