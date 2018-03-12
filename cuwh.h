#ifndef CUWHH
#define CUWHH

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
                      double cam_l, double cam_phi, double cam_theta);

int compute_wh(State *ics, int Nx, int Ny, double rhosq);
void curk4(int NA, int NB, int N, State *states, double rhosq, double h);


#endif