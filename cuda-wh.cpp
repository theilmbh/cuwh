#include <iostream>
#include <cmath>
#include <fstream>
#include "cuwh.h"
#include <cstdlib>
#include <stdio.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;



double r_l(double r, double rhosq)
{
    return sqrt(rhosq + pow(r, 2));
}

double l_dneg_cpu(double r, double a, double rho, double M)
{
    if (abs(r) < a) {
        return rho;
    }

    double x = (2*(abs(r) - a)) / (pi*M);
    return rho + M*(x*atan(x) - log(1+pow(x,2))/2);
}

double cam_rot(double t)
{
    //return pi/(1+exp(-4*(t - 0.5)));
    return pi/2;
}

void cam_trajectory(double *r, double *phi, double *theta, double t)
{
    double DR = 20;
    double c = 4*DR/(2+sin(2));
    double vphi = -2*pi;
    *r = c*(t/2.0 + sin(2*t)/(4)) - 10;
    *r = -0.01;
    *phi = vphi*t;
    *theta = pi/2;
}

State * get_frame_ics(int Nx, int Ny, double thetaFOV, double phiFOV,
                      double cam_l, double cam_phi, double cam_theta, 
                      double t, double a, double rho, double M)
{

    int N = Nx*Ny;
    State *ret = (State *)malloc(N*sizeof(State));

    int i,j;


    double mintheta = pi/2 - deg2rad*thetaFOV/2;
    double maxtheta = pi/2 + deg2rad*thetaFOV/2;
    double minphi = -deg2rad*phiFOV/2;
    minphi *= 1;
    double maxphi = deg2rad*phiFOV/2;
    maxphi *= 1;
    double cam_r = l_dneg_cpu(cam_l, a, rho, M);

    double thetacs, phics;
    double nx, ny, nz;
    double phieps = 0.00001;

    double rot = cam_rot(t);

    for(i=0; i<Nx; i++)  // phi
    {
        for(j=0; j<Ny; j++) // theta
        {
            thetacs = mintheta + (((double) j)/Ny)*(maxtheta - mintheta);
            phics = minphi + (((double) i)/Nx)*(maxphi - minphi) + phieps + rot;
            nx = sin(thetacs)*cos(phics);
            ny = sin(thetacs)*sin(phics);
            nz = -1.0*cos(thetacs);

            ret[i*Ny +j].r = cam_l;
            ret[i*Ny +j].theta = cam_theta;
            ret[i*Ny +j].phi = cam_phi;
            ret[i*Ny +j].pr = nx;
            ret[i*Ny +j].ptheta = cam_r*nz;
            ret[i*Ny +j].b = cam_r*sin(cam_theta)*ny;
            ret[i*Ny +j].Bsq = pow(cam_r, 2)*(pow(nz,2) + pow(ny,2));
        }
    }

    return ret;

}

double remap_phi(double phi)
{

    // remap phi to [-pi, pi]
    double sgn = 1.0;
    if (phi < 0)
    {
        sgn = -1.0;
    }
    while(abs(phi) >= 2*pi)
    {
        phi -= sgn*2*pi;
    }
    return phi;
}

double remap_theta(double theta)
{
    // remap theta to [0, pi]
    double sgn = 1.0;
    if (theta < 0)
    {
        sgn = -1.0;
    }
    while((theta >= pi) || (theta <=0))
    {
        theta -= sgn*pi;
    }
    return theta;
}

Vec3b bilinear_interpolate(const Mat& img, int pxf, int pyf, int pxc, int pyc, double px, double py)
{
    Vec3b res;
    Vec3b y1 = (pxc - px)/(pxc - pxf) * img.at<Vec3b>(pyf, pxf) +
                (px - pxf)/(pxc-pxf) * img.at<Vec3b>(pyf, pxc);

    Vec3b y2 = (pxc - px)/(pxc - pxf) * img.at<Vec3b>(pyc, pxf) +
                (px - pxf)/(pxc-pxf) * img.at<Vec3b>(pyc, pxc);

    res = (pyc - py)/(pyc - pyf) * y1 + (py - pyf)/(pyc - pyf) * y2;
    return res;
}

void map_image(int Nx, int Ny, State *res, int framenum)
{

    Mat cs1;
    Mat cs2;

    cs1 = imread("/home/brad/wormhole/cuwh/saturn.jpeg", 1);
    cs2 = imread("/home/brad/wormhole/cuwh/gargantua.jpeg", 1);

    Mat out_image = Mat::zeros(Ny, Nx, cs1.type());

    int pxf, pyf, pxc, pyc,
        x, y;
    double phi, theta, s, px, py;
    Vec3b pix;

    //cout << "Mapping..." << endl;
    for (x=0; x<Nx; x++)
    {
        for (y=0; y<Ny; y++)
        {
            //cout << "x, y: " << x << y << endl;
            //cout << res[x*Ny + y].phi << " " << res[x*Ny +y].theta << endl;
            phi = remap_phi(res[x*Ny + y].phi);
            theta = remap_theta(res[x*Ny +y].theta);
            //cout << phi << " " << theta << endl;
            s = res[x*Ny+y].r / abs(res[x*Ny+y].r);
            px = (phi)/(2*pi) * cs1.cols;
            py = (theta/pi) * cs1.rows;
            pxf = floor(px);
            pyf = floor(py);
            pxc = ceil(px);
            pyc = ceil(py);
            //cout << "interpolating" << endl;
            if (s < 0) {
                pix = bilinear_interpolate(cs1, pxf, pyf, pxc, pyc, px, py);
            } else {
                pix = bilinear_interpolate(cs2, pxf, pyf, pxc, pyc, px, py);
            }

            for (int c = 0; c < 3; c++)
            {
                out_image.at<Vec3b>(y, x)[c] = pix[c];
            }
       
        }
    }

    // namedWindow("Wormhole", WINDOW_AUTOSIZE);
    // imshow("Wormhole", out_image);
    // waitKey();
    cout << "Writing image..." << endl;
    char fname[128];
    sprintf(fname, "./frame%d.tiff", framenum);
    imwrite(fname, out_image);
}

int main(void)
{


    int Nx = 320;
    int Ny = 180;  // Pixes

    int N = Nx*Ny;  // total pixels in image.

    // Setup Initial conditions
    double thetaFOV = 60;  // Vertical field of view
    double phiFOV = thetaFOV*((double)Nx/(double)Ny);  // Keep aspect ratio
    

    double cam_l = -10.0;
    double cam_phi = pi;
    double cam_theta = pi/2;
    double rhosq = pow(4.0, 2);

    int nframes = 120;
    double dphi = 2*pi/nframes;
    double dtheta = pi/12;
    double t = 0;

    // Wormhole parameters
    double rho = 4.0;
    double a = 0.01*rho/2;
    double W = 0.05*rho;
    double M = W / 1.42953;
    double dr = 0.5;


    State * states_host;

    for(int frame=0; frame<nframes; frame++)
    {
        t = (double)frame/(double)nframes;
        // cam_phi += dphi;
        // cam_theta = pi/2 + dtheta*sin(2*pi*6*t);
        // cam_l += dr;
        cam_trajectory(&cam_l, &cam_phi, &cam_theta, t);

        states_host = get_frame_ics(Nx, Ny, thetaFOV, phiFOV, cam_l, cam_phi, cam_theta, t, a, rho, M);
        int err = compute_wh(states_host, Nx, Ny, a, rho, M);
        if (err < 0) {
            cout << "Computation Failed" << endl;
            return -1;
        }
        cout << "Compute frame: " << frame+1 << endl;
        map_image(Nx, Ny, states_host, frame+1);
        cout << "Completed frame: " << frame+1 << endl;
    }

    ofstream outfile("mapout.dat", ios::binary);
    outfile.write((char *)states_host, N*sizeof(State));
    outfile.close();
    free(states_host);
    return 0;

}
