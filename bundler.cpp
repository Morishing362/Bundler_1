#include "methods.hpp"
#include <eigen3/Eigen/Dense>
#include <string>

// global variable
int n, m; // n = number of points, m = number of cameras.
MatrixXd XYZ;// points cloud shaped matrix(3, n).
MatrixXd XYZ_pls;
vector<Camera> C; // list of cameras.
vector<Camera> C_pls;


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

    string path = argv[1];
    readData(path);

    outXYZ(XYZ, argv[2]); // output initial points cloud.

    XYZ_pls = XYZ;

    double E, E_pls;
    VectorXd e, e_pls;
    VectorXd a;
    VectorXd dx;
    MatrixXd _J, J, Jt, JtJ, L;
    double damp, gainfactor;

    double Beta, Gamma, q1, q2;
    Beta = 2; Gamma = 3; q1 = 0.25; q2 = 0.75;

    damp = 1;

    int iter = 0;
    for (int k = 0; k < 100; k++) {
        iter ++;

        for (int i = 0; i < m; i++) {
            C[i].reproject(XYZ);
        }
        e = funcVec(C, m, n);
        E = reprojectionError(e);

        cout << "Reprojection Error : " << E << endl;
        // cout << damp << endl;

        _J = makeJacobiMatrix(C, XYZ, m, n);
        J = _J.block<2000, 623>(0, 7); // custom block <2mn, 6m+3n-7>
        Jt = J.transpose();

        cout << "making JtJ and grad..." << endl;
        JtJ = Jt * J;
        L = JtJ + damp * MatrixXd::Identity(623, 623); // custom row and column (6m+3n-7, 6m+3n-7)
        a = - Jt * e;

        cout << "solving..." << endl;
        dx = L.partialPivLu().solve(a);

        C_pls = C;
        XYZ_pls = XYZ;

        // renew C_pls prameters.
        for (int i = 0; i < m - 1; i++) {
            if (i == 0) {
                C_pls[1].t(1) += dx(0);
                C_pls[1].t(2) += dx(1);
            }
            else {
                C_pls[i+1].t(0) += dx(6*i - 1);
                C_pls[i+1].t(1) += dx(6*i);
                C_pls[i+1].t(2) += dx(6*i + 1);
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

            C_pls[i+1].R = exp_wx * C_pls[i+1].R;
        }

        for (int i = 0; i < n; i++) {
            XYZ_pls(0, i) += dx(6*m - 7 + 3*i);
            XYZ_pls(1, i) += dx(6*m - 6 + 3*i);
            XYZ_pls(2, i) += dx(6*m - 5 + 3*i);
        }

        for (int i = 0; i < m; i++) {
            C_pls[i].reproject(XYZ_pls);
        }
        e_pls = funcVec(C_pls, m, n);
        E_pls = reprojectionError(e_pls);

        gainfactor = (E - E_pls) / (dx.dot(damp * dx + a) / 2);

        if (gainfactor < q1) { 
            damp = damp * Beta; 
        }
        else if (q2 < gainfactor) {
            damp = damp / Gamma;
        }

        if (0 < gainfactor) {
            C = C_pls;
            XYZ = XYZ_pls;
        }

        if (a.norm() <= 10e-5) {
            break;
        }
    }

    cout << iter << endl;

    outXYZ(XYZ, argv[3]); // output result points cloud.

    return 0;
}