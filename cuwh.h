#ifndef CUWHH
#define CUWHH

const double pi = 3.14159;
const double deg2rad = pi/180;

typedef struct {
    double r;
    double theta;
    double phi;
    double pr;
    double ptheta;
    double b;
    double Bsq;
} State;        // Dynamical System State Vector

State * get_frame_ics(int Nx, int Ny, double thetaFOV, double phiFOV,
                      double cam_l, double cam_phi, double cam_theta,
                      double a, double rho, double M);

int compute_wh(State *ics, int Nx, int Ny, double a, double rho, double M);
void curk4(int NA, int NB, int N, State *states, double a, double rho, double M, double h);


#endif