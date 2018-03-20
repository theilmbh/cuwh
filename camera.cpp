// camera.cpp  Camera trajectory and orientation
//
// Brad Theilman 2018 03 19

#include <vector>

vector<float> bezier(const vector<float>& P0, const vector<float>& P1, 
                     const vector<float>& P2, const vector<float>& P3,
                     float t)
{
    // return cubic bezier point
    vector<float> res;
    float c0 = pow(1-t, 3);
    float c1 = 3*pow(1-t, 2)*t;
    float c2 = c1;
    float c3 = pow(t, 3);
    for (int i=0; i<= P0.size(), i++)
    {
        res.push_back(c0*P0[i] + c1*P1[i] + c2*P1[i] + c3*P3[i]);
    }
    return res;
}


vector<float> bezier_prime(const vector<float>& P0, const vector<float>& P1, 
                           const vector<float>& P2, const vector<float>& P3,
                           float t)
{
    vector<float> res;
    float c0 = 3*pow(1-t, 2);
    float c1 = 6*(1-t)*t;
    float c2 = 3*t*t;
    for (int i=0; i<= P0.size(), i++)
    {
        res.push_back(c0*(P1[i] -P0[i]) + c1*(P2[i] - P1[i]) + c3*(P3[i] - P2[i]));
    }
    return res;
}

vector<float> bezier_pprime(const vector<float>& P0, const vector<float>& P1, 
                           const vector<float>& P2, const vector<float>& P3,
                           float t)
{
    vector<float> res;
    float c0 = 6*(1-t); 
    float c1 = 6*t;
    for (int i=0; i<= P0.size(), i++)
    {
        res.push_back(c0*(P2[i] - 2*P1[i] + P0[i]) + c1*(P3[i] - 2*P2[i] + P1[i]));
    }
    return res;
}

float norm(const vector<float>& P)
{
    return sqrt(P[0]*P[0] + P[1]*P[1] + P[2]*P[2]);
}

vector<float> cross(const vector<float>& P0, const vector<float>& P1)
{
    vector<float> res;
    res.push_back(P0[1]*P1[2] - P0[2]*P1[1]);
    res.push_back(P0[2]*P1[0] - P0[0]*P1[2]);
    res.push_back(P0[0]*P1[1] - P0[1]*P1[0]);
    return res;
}

vector<float> TNB_T(const vector<float>& r_prime)
{
    vector<float> res;
    float nm = norm(r_prime);
    res.push_back(r_prime[0]/nm);
    res.push_back(r_prime[1]/nm);
    res.push_back(r_prime[2]/nm);
    return res;
}

vector<float> TNB_N(const vector<float>& r_prime, const vector<float>& r_pprime)
{
    vector<float> b = cross(r_pprime, r_prime);
    vector<float> res = cross(r_prime, b);
    float nmb = norm(b);
    float nmt = norm(r_prime);

    res[0] /= (nmb*nmt);
    res[1] /= (nmb*nmt);
    res[2] /= (nmb*nmt);
    
    return res;
}

vector<float> TNB_B(const vector<float>& r_prime, const vector<float>& r_pprime)
{
    vector<float> res = cross(r_prime, r_pprime);
    float nmb = norm(res);
    
    res[0] /= (nmb);
    res[1] /= (nmb);
    res[2] /= (nmb);
    return res;
}



