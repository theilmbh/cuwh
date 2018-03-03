#include <iostream>
#include <math.h>
#include <fstream>
#include <cstdlib>

using namespace std;

typedef struct {
    double r;
    double theta;
    double phi;
    double pr;
    double ptheta;
    double b;
    double Bsq;
} State;        // Dynamical System State Vector


State state_add(const State &s1, const State &s2)
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

State state_mul(double g, const State &s)
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

double l(double r)
{
    double rhosq = 0.01;
    return sqrt(rhosq + pow(r, 2));
}

double dldr(double r)
{
    return r/l(r) ;
}

State rhs(const State &s, double gamma)
{

    State ds;
    double rsq = pow(l(s.r), 2);

    ds.r = s.pr;
    ds.theta = s.ptheta / rsq;
    ds.phi = s.b / (rsq*pow(sin(s.theta), 2));
    ds.pr = s.Bsq*(dldr(s.r) / (pow(l(s.r), 3)));
    ds.ptheta = (pow(s.b, 2)/rsq) * cos(s.theta)/pow(sin(s.theta), 3);
    ds.b = 0.0;
    ds.Bsq = 0.0;

    return ds;
}

void rk4_step(int N, State *states, double gamma, double h, int indx)
{

    State s = states[indx];
    State y = s;

    if(s.pr < 0) {
        int a = 1;
    }

    State k1 = rhs(s, gamma);
    s = state_add(y, state_mul(h/2, k1));
    State k2 = rhs(s, gamma);
    s = state_add(y, state_mul(h/2, k2));
    State k3 = rhs(s, gamma);
    s = state_add(y, state_mul(h, k3));
    State k4 = rhs(s, gamma);

    s = state_add(y, state_mul(h/6, state_add(k4, state_add(state_mul(2.0, state_add(k2, k3)), k1))));
    states[indx] = s;
}

int main(void)
{

    int Nx = 8;
    int Ny = 8;  // Pixes

    int N = Nx*Ny;  // total pixels in image.

    // Allocate Host array for initial conditions:
    State *states_host = (State *)malloc(N*sizeof(State));
    if (states_host == NULL)
    {
        cout << "Allocation error" << endl;
    }

    // Setup Initial conditions
    int i,j;
    double pi = 3.14159;
    double deg2rad = pi/180;

    double thetaFOV = 20;
    double phiFOV = 20;

    double mintheta = pi/2 - deg2rad*thetaFOV/2;
    double maxtheta = pi/2 + deg2rad*thetaFOV/2;
    double minphi = -deg2rad*phiFOV/2;
    double maxphi = deg2rad*phiFOV/2;

    double thetacs, phics;
    double nx, ny, nz;

    double cam_l = -10.0;
    double cam_r = sqrt(1.0 + pow(cam_l, 2));
    double cam_phi = 0.0;
    double cam_theta = pi/2;

    for(i=0; i<Nx; i++)  // phi
    {
        for(j=0; j<Ny; j++) // theta
        {
            thetacs = mintheta + (((double) j)/Ny)*(maxtheta - mintheta);
            phics = minphi + (((double) i)/Nx)*(maxphi - minphi);
            nx = sin(thetacs)*cos(phics);
            ny = sin(thetacs)*sin(phics);
            nz = -1.0*cos(thetacs);

            states_host[i*Nx +j].r = cam_l;
            states_host[i*Nx +j].theta = cam_theta;
            states_host[i*Nx +j].phi = cam_phi;
            states_host[i*Nx +j].pr = nx;
            states_host[i*Nx +j].ptheta = cam_r*nz;
            states_host[i*Nx +j].b = cam_r*sin(cam_theta)*ny;
            states_host[i*Nx +j].Bsq = pow(cam_r, 2)*(pow(nz,2) + pow(ny,2));
        }
    }
        int ind = 4*Nx + 4;
        ind =0;
        cout << states_host[ind].r << " " << states_host[ind].theta << " " <<
             states_host[ind].phi << " " << " " << states_host[ind].pr << " " 
             << states_host[ind].b << endl;

    // Integrate
    double dt = 1e-3;
    double t = 0.0;
    double tend = 100.0;
    int k = 0;
    while(t < tend)
    {
        for (int m = 0; m < N; m++){
            rk4_step(N, states_host, 1.0, dt, m);
        }
        
        t += dt;
        k +=1;

        int ind = 4*Nx+4;
        ind =0;
        for (int j=0; j< 1; j++)
        {
            cout << "UPDATE   " <<  states_host[ind].r << " "
                << states_host[ind].theta << " " << states_host[ind].phi 
                << " " << states_host[ind].pr << " " << states_host[ind].b << endl;
        }
    }

}
