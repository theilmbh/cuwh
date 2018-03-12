#include <iostream>
#include <math.h>
#include <fstream>
#include "cuwh.h"
#include <cstdlib>
#include <stdio.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const double pi = 3.14159;
const double deg2rad = pi/180;

double r_l(double r, double rhosq)
{
    return sqrt(rhosq + pow(r, 2));
}

State * get_frame_ics(int Nx, int Ny, double thetaFOV, double phiFOV,
                      double cam_l, double cam_phi, double cam_theta, double rhosq)
{

    int N = Nx*Ny;
    State *ret = (State *)malloc(N*sizeof(State));

    int i,j;


    double mintheta = pi/2 - deg2rad*thetaFOV/2;
    double maxtheta = pi/2 + deg2rad*thetaFOV/2;
    double minphi = -deg2rad*phiFOV/2;
    minphi *= 0.5;
    double maxphi = deg2rad*phiFOV/2;
    maxphi *=0.5;
    double cam_r = r_l(cam_l, rhosq);

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

    cs2 = imread("/home/brad/wormhole/cuwh/saturn.jpeg", 1);
    cs1 = imread("/home/brad/wormhole/cuwh/stars.jpeg", 1);
    Mat out_image = Mat::zeros(Ny, Nx, cs1.type());

    int pxf, pyf, pxc, pyc,
        x, y;
    double phi, theta, s, px, py;
    Vec3b pix;

    for (x=0; x<Nx; x++)
    {
        for (y=0; y<Ny; y++)
        {
            phi = remap_phi(res[x*Ny + y].phi);
            theta = remap_theta(res[x*Ny +y].theta);
            s = res[x*Ny+y].r / abs(res[x*Ny+y].r);
            px = (phi)/(2*pi) * cs1.cols;
            py = (theta/pi) * cs1.rows;
            pxf = floor(px);
            pyf = floor(py);
            pxc = ceil(px);
            pyc = ceil(py);

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
    char fname[128];
    sprintf(fname, "./frame%d.jpg", framenum);
    imwrite(fname, out_image);
}

int main(void)
{

    int Nx = 2048;
    int Ny = 1024;  // Pixes

    int N = Nx*Ny;  // total pixels in image.

    // Setup Initial conditions
    double thetaFOV = 90;
    double phiFOV = 360;

    double cam_l = -10.0;
    double cam_phi = pi;
    double cam_theta = pi/2;
    double rhosq = pow(4.0, 2);

    int nframes = 120;
    double dphi = 2*pi/nframes;

    State * states_host;

    for(int frame=0; frame<nframes; frame++)
    {
        cam_phi += dphi;
        states_host = get_frame_ics(Nx, Ny, thetaFOV, phiFOV, cam_l, cam_phi, cam_theta, rhosq);
        int err = compute_wh(states_host, Nx, Ny, rhosq);
        if (err < 0) {
            cout << "Computation Failed" << endl;
            return -1;
        }
        map_image(Nx, Ny, states_host, frame+1);
        cout << "Completed frame: " << frame+1 << endl;
    }



    ofstream outfile("mapout.dat", ios::binary);
    outfile.write((char *)states_host, N*sizeof(State));
    outfile.close();

    
    free(states_host);
    return 0;

}
