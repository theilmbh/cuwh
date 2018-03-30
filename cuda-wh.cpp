#include <iostream>
#include <cmath>
#include <fstream>
#include "cuwh.h"
#include <cstdlib>
#include <stdio.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;



float r_l(float r, float rhosq)
{
    return sqrt(rhosq + pow(r, 2));
}

float l_dneg_cpu(float r, float a, float rho, float M)
{
    if (abs(r) < a) {
        return rho;
    }

    float x = (2*(abs(r) - a)) / (pi*M);
    return rho + M*(x*atan(x) - log(1+pow(x,2))/2);
}

float cam_rot(float t)
{
    //return pi/(1+exp(-4*(t - 0.5)));
    return pi/2;
}

void cam_trajectory(float *r, float *phi, float *theta, float t, float a, float b)
{
    float DR = 20;
    float c = 4*DR/(2+sin(2));
    //float c = 20;
    float V = 2.0/24; // km/frame (2 km/s)
    float vphi = -2*pi;
    //*r = c*(t/2.0 + sin(2*t)/(4)) - 10;
    //*r = c*t - 10;

    if (t < a)
    {
        *r = (10/a)*t - 10;
    }
    if (t > b)
    {
        *r = (10/(1-b))*t - (10*b)/(1-b);
    }
    if(t < b && t > a)
    {
        *r = 0;
    }

    *r = 20*tan(t-0.5);
    *r = (20*pow(0.5, -3)*pow(t-0.5, 3));
    //*r = -0.01;
    //*phi = V*log(-10+c*t)/c;
    *phi = vphi*t;
    *theta = pi/2;
}

State * get_frame_ics(int Nx, int Ny, float thetaFOV, float phiFOV,
                      float cam_l, float cam_phi, float cam_theta, 
                      float t, float a, float rho, float M)
{

    int N = Nx*Ny;
    State *ret = (State *)malloc(N*sizeof(State));

    int i,j;


    float mintheta = pi/2 - deg2rad*thetaFOV/2;
    float maxtheta = pi/2 + deg2rad*thetaFOV/2;
    float minphi = -deg2rad*phiFOV/2;
    minphi *= 1;
    float maxphi = deg2rad*phiFOV/2;
    maxphi *= 1;
    float cam_r = l_dneg_cpu(cam_l, a, rho, M);

    float thetacs, phics;
    float nx, ny, nz;
    float phieps = 0.00001;

    float rot = cam_rot(t);

    for(i=0; i<Nx; i++)  // phi
    {
        for(j=0; j<Ny; j++) // theta
        {
            thetacs = mintheta + (((float) j)/Ny)*(maxtheta - mintheta);
            phics = minphi + (((float) i)/Nx)*(maxphi - minphi) + phieps + rot;
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

float remap_phi(float phi)
{

    // remap phi to [-pi, pi]

    // compute mod 2pi
    float r = phi - 2*pi*floor(phi/(2*pi));

    if (r > pi)
    {
        r -= 2*pi;
    }
    return r;
    // float sgn = 1.0;
    // if (phi < 0)
    // {
    //     sgn = -1.0;
    // }
    // while(abs(phi) >= 2*pi)
    // {
    //     phi -= sgn*2*pi;
    // }
    // return phi;
}

float remap_theta(float theta)
{
    // remap theta to [0, pi]

    float r = theta - 2*pi*floor(theta/(2*pi));
    return r;
    // float sgn = 1.0;
    // if (theta < 0)
    // {
    //     sgn = -1.0;
    // }
    // while((theta >= pi) || (theta <=0))
    // {
    //     theta -= sgn*pi;
    // }
    // return theta;
}

void remap_phi_theta(float phi, float theta, float *phi2, float *theta2)
{
    float x = sin(theta)*cos(phi);
    float y = sin(theta)*sin(phi);
    float z = cos(theta);

    *theta2 = acos(z)/pi;
    *phi2 = atan2(y,x)/(2*pi);

}

Vec3b bilinear_interpolate(const Mat& img, int pxf, int pyf, int pxc, int pyc, float px, float py)
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

    cs1 = imread("./saturn.jpeg", 1);
    cs2 = imread("./gargantua.jpeg", 1);

    Mat out_image = Mat::zeros(Ny, Nx, cs1.type());

    int pxf, pyf, pxc, pyc,
        x, y;
    float phi, theta, s, px, py;
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
    cout << "Writing image..." << endl;
    char fname[128];
    sprintf(fname, "./imgs/frame%d.tiff", framenum);
    imwrite(fname, out_image);
}

int main(void)
{


    int Nx = 640;
    int Ny = 360;  // Pixes

    int N = Nx*Ny;  // total pixels in image.

    // Setup Initial conditions
    float thetaFOV = 60;  // Vertical field of view
    float phiFOV = thetaFOV*((float)Nx/(float)Ny);  // Keep aspect ratio
    

    float cam_l = -10.0;
    float cam_phi = pi;
    float cam_theta = pi/2;
    float rhosq = pow(4.0, 2);

    int nframes = 15*24;
    float dphi = 2*pi/nframes;
    float dtheta = pi/12;
    float t = 0;

    // Wormhole parameters
    float rho = 4.0;
    float a = 0.01*rho/2;
    float W = 0.05*rho;
    float M = W / 1.42953;
    float dr = 0.5;

    float a1 = 3./5.;
    float b1 = 4./5.;


    State * states_host;
    int dev = 3;

    for(int frame=0; frame<nframes; frame++)
    {
        t = (float)frame/(float)nframes;
        // cam_phi += dphi;
        // cam_theta = pi/2 + dtheta*sin(2*pi*6*t);
        // cam_l += dr;
        cam_trajectory(&cam_l, &cam_phi, &cam_theta, t, a1, b1);
        cout << t<< " " <<cam_l << " " << cam_phi << endl;

        states_host = get_frame_ics(Nx, Ny, thetaFOV, phiFOV, cam_l, cam_phi, cam_theta, t, a, rho, M);
        int err = compute_wh(dev, states_host, Nx, Ny, a, rho, M);
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
