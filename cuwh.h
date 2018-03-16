#ifndef CUWHH
#define CUWHH

const float pi = 3.14159;
const float deg2rad = pi/180;

typedef struct {
    float r;
    float theta;
    float phi;
    float pr;
    float ptheta;
    float b;
    float Bsq;
} State;        // Dynamical System State Vector

State * get_frame_ics(int Nx, int Ny, float thetaFOV, float phiFOV,
                      float cam_l, float cam_phi, float cam_theta,
                      float a, float rho, float M);

int compute_wh(State *ics, int Nx, int Ny, float a, float rho, float M);
void curk4(int NA, int NB, int N, State *states, float a, float rho, float M, float h);


#endif