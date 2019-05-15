#include <iostream>
#include <fstream>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <vector>

using namespace std;
using namespace Eigen;

class Camera {
    public:
        double f;
        double u0, v0;
        MatrixXd R;
        Vector3d t;
        MatrixXd K;
        MatrixXd uv_a, uv_n;
        int id;

        Camera(double _f, double _u0, double _v0, 
               MatrixXd& _R, Vector3d& _t, 
               MatrixXd& _uv_a, MatrixXd& _uv_n, int _id) {
            f = _f; u0 = _u0; v0 =_v0;
            R = _R; t = _t;
            uv_a = _uv_a; uv_n = _uv_n; id = _id;
            MatrixXd _K(3, 3);
            _K << _f, 0, _u0,
                    0, _f, _v0,
                    0, 0, 1;
            K = _K;
        }

        void reproject(MatrixXd& XYZ) {
            int _n = XYZ.cols();

            MatrixXd XYZ_1 = MatrixXd::Ones(4, _n);
            XYZ_1.block<3, 1000>(0, 0) = XYZ; // custom block size <3, n>
            
            MatrixXd R_t(3, 4);
            R_t << R, t;

            MatrixXd local = R_t * XYZ_1;

            for(int i = 0; i < _n; i++) {
                local(0, i) = local(0, i) / local(2, i);
                local(1, i) = local(1, i) / local(2, i);
                local(2, i) = local(2, i) / local(2, i);
            } 

            uv_n =  K * local;
        }
};

MatrixXd makeJacobiMatrix(vector<Camera>& C, MatrixXd& XYZ, int m, int n) {
    vector<vector<MatrixXd>> F; // [m, n]
    vector<vector<Vector3d>> p; // [m ,n]
    vector<Vector3d> XYZ_stdvec;

    for (int i = 0; i < n; i++) {
        Vector3d xyz;
        xyz << XYZ(0, i), XYZ(1, i), XYZ(2, i);
        XYZ_stdvec.push_back(xyz);
    }

    for (int i = 0 ; i < m; i++) {
        vector<Vector3d> tp;
        vector<MatrixXd> tF;

        for (int j = 0 ; j < n; j++) {
            Vector3d ttp;
            ttp = C[i].R * XYZ_stdvec[j] + C[i].t;
            tp.push_back(ttp);

            MatrixXd ttF(2, 3);
            ttF << 1/ttp[2], 0, -ttp[0]/((ttp[2])*(ttp[2])),
                   0, 1/ttp[2], -ttp[1]/((ttp[2])*(ttp[2]));
            tF.push_back(ttF);
        }
        p.push_back(tp);
        F.push_back(tF);
    }

    MatrixXd J = MatrixXd::Zero(2*n*m, 6*m + 3*n);

    for (int i = 0 ; i < m; i++) {
        for (int j = 0 ; j < n; j++) {
            J.block<2, 3>(2*n*i+2*j, 6*i) = C[i].f * F[i][j];
            
            Vector3d a = C[i].R * XYZ_stdvec[j];
            MatrixXd A(3, 3);
            A << 0, -a(2), a(1),
                a(2), 0, -a(0),
                -a(1), a(0), 0;
            J.block<2, 3>(2*n*i+2*j, 6*i+3) = - C[i].f * F[i][j] * A;

            J.block<2, 3>(2*n*i+2*j, 6*m+3*j) = C[i].f * F[i][j] * C[i].R;
        }
    }

    return J;
}

VectorXd funcVec(vector<Camera>& C, int m, int n) {
    VectorXd e(2*n*m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            e(2*n*i + 2*j)   = C[i].uv_n(0, j) - C[i].uv_a(0, j);
            e(2*n*i + 2*j + 1) = C[i].uv_n(1, j) - C[i].uv_a(1, j);
        }
    }

    return e;
}

double reprojectionError(VectorXd& funcvec) {
    return funcvec.dot(funcvec) / 2;
}

void outXYZ(MatrixXd& XYZ, string path) {
    ofstream file;
    file.open(path);
    for (int i = 0; i < XYZ.cols(); i++) {
        file << XYZ(0, i) << " " << XYZ(1, i) << " " << XYZ(2, i) << "\n";
    }
    file.close();
}