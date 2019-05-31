#include "methods.hpp"
#include <eigen3/Eigen/Dense>
#include <string>
#include <cassert>
#include <algorithm>
#include <functional>

// global variable
int n, m; // n = number of points, m = number of cameras.
MatrixXd XYZ;// points cloud shaped matrix(3, n).
MatrixXd XYZ_new;
vector<Camera> C; // list of cameras.
vector<Camera> C_new;


void readData(string path) {
    ifstream file;
    file.open(path);
    file >> n;
    MatrixXd _XYZ(3, n);
    for (int i = 0; i < n; i++) {
        file >> _XYZ(0, i) >> _XYZ(1, i) >> _XYZ(2, i);
    }
    XYZ = _XYZ;
    file >> m;
    for (int i = 0; i < m; i++) {
        int id;
        double f, u0, v0;
        MatrixXd R(3, 3);
        Vector3d t;
        MatrixXd uv_a(3, n);
        MatrixXd uv_n(3, n);
        file >> id;
        file >> f >> u0 >> v0;
        file >> R(0, 0) >> R(0, 1) >> R(0, 2) 
             >> R(1, 0) >> R(1, 1) >> R(1, 2)
             >> R(2, 0) >> R(2, 1) >> R(2, 2);
        file >> t(0) >> t(1) >> t(2);
        for (int j = 0; j < n; j++) {
            file >> uv_a(0, j) >> uv_a(1, j) >> uv_n(0, j) >> uv_n(1, j);
            uv_a(2, j) = 1;
            uv_n(2, j) = 1;
        }
        Camera c(f, u0, v0, R, t, uv_a, uv_n, id);
        C.push_back(c);
    }
    file.close();
}

int main(int argc,char *argv[]) {

    readData(argv[1]);

    outXYZ(XYZ, argv[2]); // output initial points cloud.

    XYZ_new = XYZ;

    double E, E_new;
    VectorXd e, e_new;
    VectorXd a;
    VectorXd x(6*m - 7 + 3*n);
    VectorXd x_new(6*m - 7 + 3*n);
    VectorXd dx;
    MatrixXd _J, J, Jt, JtJ, L;
    double damp, gainfactor;

    double Beta, Gamma, q1, q2, nu;
    Beta = 2; Gamma = 3; q1 = 0.25; q2 = 0.75; nu = 0;
    double e1, e2, e3, e4;
    e1 =  e2 = e3 = 10e-12; e4 = 0;
    bool stop = false;

    damp = 1;

    // make x vector
    for (int i = 0; i < m - 1; i++) {
        if (i == 0) {
            x(0) = C[1].t(1);
            x(1) = C[1].t(2);
        }
        else {
            x(6*i - 1) = C[i+1].t(0);
            x(6*i)     = C[i+1].t(1);
            x(6*i + 1) = C[i+1].t(2);
        }
        x(6*i + 2) = 0;
        x(6*i + 3) = 0;
        x(6*i + 4) = 0;
    }
    for (int i = 0; i < n; i++) {
        x(6*m - 7 + 3*i) = XYZ(0, i);
        x(6*m - 6 + 3*i) = XYZ(1, i);
        x(6*m - 5 + 3*i) = XYZ(2, i);
    }

    for (int i = 0; i < m; i++) {
        C[i].reproject(XYZ);
    }
    e = funcVec(C, m, n);
    E = reprojectionError(e);

    cout << "Reprojection Error : " << E << endl;

    _J = makeJacobiMatrix(C, XYZ, m, n);      
    J = _J.block<2000, 623>(0, 7); // custom block <2mn, 6m+3n-7>
    Jt = J.transpose();
    JtJ = Jt * J;
    a = - Jt * e;

    int iter = 0;
    for (int k = 0; k < 100; k++) {
        iter ++;

        L = JtJ + damp * MatrixXd::Identity(623, 623); // custom row and column (6m+3n-7, 6m+3n-7)
        dx = L.partialPivLu().solve(a);

        if ( dx.norm() / (x.norm() + e2) <= e2 ) {
            stop = true;
        } else {
            C_new = C;
            XYZ_new = XYZ;

            // renew C_new prameters.
            for (int i = 0; i < m - 1; i++) {
                if (i == 0) {
                    C_new[1].t(1) += dx(0);
                    C_new[1].t(2) += dx(1);
                } else {
                    C_new[i+1].t(0) += dx(6*i - 1);
                    C_new[i+1].t(1) += dx(6*i);
                    C_new[i+1].t(2) += dx(6*i + 1);
                }

                Vector3d w;
                MatrixXd wx(3, 3);
                MatrixXd exp_wx(3, 3);

                w << dx(6*i + 2), dx(6*i + 3), dx(6*i + 4);
                double theta = sqrt(w.dot(w));

                wx << 0, -w(2), w(1),
                    w(2), 0, -w(0),
                    -w(1), w(0), 0;

                exp_wx = MatrixXd::Identity(3, 3) + (sin(theta) / theta)*wx + ((1-cos(theta)) / (theta*theta))*wx*wx;

                C_new[i+1].R = exp_wx * C_new[i+1].R;
            }

            // renew XYZ_new prameters.
            for (int i = 0; i < n; i++) {
                XYZ_new(0, i) += dx(6*m - 7 + 3*i);
                XYZ_new(1, i) += dx(6*m - 6 + 3*i);
                XYZ_new(2, i) += dx(6*m - 5 + 3*i);
            }
            
            // make funcvec e_new and reprojection Error E_new.
            for (int i = 0; i < m; i++) {
                C_new[i].reproject(XYZ_new);
            }
            e_new = funcVec(C_new, m, n);
            E_new = reprojectionError(e_new);

            gainfactor = (E - E_new) / (dx.dot(damp * dx + a) / 2);

            if (gainfactor > 0) {
                stop = ((e.norm() - e_new.norm()) / e.norm() < e4);
                C = C_new;
                XYZ = XYZ_new;

                // make x vector
                for (int i = 0; i < m - 1; i++) {
                    if (i == 0) {
                        x(0) = C[1].t(1);
                        x(1) = C[1].t(2);
                    }
                    else {
                        x(6*i - 1) = C[i+1].t(0);
                        x(6*i)     = C[i+1].t(1);
                        x(6*i + 1) = C[i+1].t(2);
                    }
                    x(6*i + 2) = 0;
                    x(6*i + 3) = 0;
                    x(6*i + 4) = 0;
                }
                for (int i = 0; i < n; i++) {
                    x(6*m - 7 + 3*i) = XYZ(0, i);
                    x(6*m - 6 + 3*i) = XYZ(1, i);
                    x(6*m - 5 + 3*i) = XYZ(2, i);
                }

                for (int i = 0; i < m; i++) {
                    C[i].reproject(XYZ);
                }
                e = funcVec(C, m, n);
                E = reprojectionError(e);

                cout << "Reprojection Error : " << E << endl;

                _J = makeJacobiMatrix(C, XYZ, m, n);      
                J = _J.block<2000, 623>(0, 7); // custom block <2mn, 6m+3n-7>
                Jt = J.transpose();
                JtJ = Jt * J;
                a = - Jt * e;

                stop = (stop || (a.lpNorm<Infinity>() < e1));
                damp = damp * max( {1.0 / 3, 1 - pow(2 * gainfactor - 1, 3)} );
                nu = 2;
            } else {
                damp = damp * nu;
                nu = 2 * nu;
            }
        }
        if ( stop ) { break; }
    }

    cout << iter << endl;

    outXYZ(XYZ, argv[3]); // output result points cloud.

    return 0;
}