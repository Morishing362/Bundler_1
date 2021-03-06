#include "methods.hpp"
#include <eigen3/Eigen/Dense>
#include <string>

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

    string path = argv[1];
    readData(path);

    outXYZ(XYZ, argv[2]); // output initial points cloud.

    XYZ_new = XYZ;

    double E, E_pls, dE;
    VectorXd e, e_pls;
    VectorXd a;
    VectorXd x(6*m - 7 + 3*n);
    VectorXd dx;
    MatrixXd _J, J, Jt, JtJ, L;
    double damp, gainfactor;

    double Alpha, Beta, Gamma, q1, q2;
    Alpha = 1; Beta = 2; Gamma = 3; q1 = 0.25; q2 = 0.75;

    damp = 1;
    dE = 0;

    int iter = 0;
    for (int k = 0; k < 100; k++) {
        iter ++;

        // make x vector (column - 7)
        x = CreateXvec_from_CandXYZ(C, XYZ, m, n);

        e = funcVec(C, m, n);
        E = reprojectionError(e);

        // cout << E << endl;
        cout << damp << endl;
        // cout << dE << endl;

        _J = makeJacobiMatrix(C, XYZ, m, n);      
        J = _J.block<2000, 623>(0, 7); // custom block <2mn, 6m+3n-7>
        Jt = J.transpose();

        // cout << "making JtJ and grad..." << endl;
        JtJ = Jt * J;
        L = JtJ + damp * MatrixXd::Identity(623, 623); // custom row and column (6m+3n-7, 6m+3n-7)
        a = - Jt * e;

        // cout << "solving..." << endl;
        dx = Alpha * L.partialPivLu().solve(a);

        if ( dx.norm() / x.norm() < 10e-12 ) {
            break;
        }

        // renew C_new prameters.
        C_new = Create_C_new(C, dx, m);
        XYZ_new = Create_XYZ_new(XYZ, dx, m, n);

        for (int i = 0; i < m; i++) {
            C_new[i].reproject(XYZ_new);
        }
        e_pls = funcVec(C_new, m, n);
        E_pls = reprojectionError(e_pls);

        gainfactor = (E - E_pls) / (dx.dot(damp * dx + a) / 2);

        if (gainfactor < q1) { 
            damp = damp * Beta; 
        } else if (q2 < gainfactor) {
            damp = damp / Gamma;
        }

        if (damp > 1) {
            Alpha = 0.5;
        } else {
            Alpha = 1;
        }

        if (0 < gainfactor) {
            C = C_new;
            XYZ = XYZ_new;
        }

        if (a.norm() <= 10e-12 ) {
            break;
        }

        dE = E_pls - E;
    }

    cout << iter << endl;

    outXYZ(XYZ, argv[3]); // output result points cloud.

    return 0;
}