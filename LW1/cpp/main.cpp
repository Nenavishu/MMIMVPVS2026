#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>

using namespace std;


double kernel(double z) {
    if (abs(z) <= 1.0)
        return 0.75 * (1.0 - z * z);
    return 0.0;
}


double npr_predict(const vector<double>& x,
    const vector<double>& y,
    int leave_out_index,
    double beta) {

    int n = x.size();
    double h = beta * pow(n, -1.0 / 5.0);

    double numerator = 0.0;
    double denominator = 0.0;

    for (int j = 0; j < n; j++) {

        if (j == leave_out_index)
            continue;

        double z = (x[leave_out_index] - x[j]) / h;
        double w = kernel(z);

        numerator += w * y[j];
        denominator += w;
    }

    if (denominator == 0.0)
        return 0.0;

    return numerator / denominator;
}


double loo_mse(const vector<double>& x,
    const vector<double>& y,
    double beta) {

    int n = x.size();
    double mse = 0.0;

    for (int i = 0; i < n; i++) {
        double y_pred = npr_predict(x, y, i, beta);
        double error = y[i] - y_pred;
        mse += error * error;
    }

    return mse / n;
}


int main() {

    setlocale(LC_ALL, "Russian");
    const int n = 80;

    vector<double> x(n);
    vector<double> y(n);

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<> dist_x(-3.0, 7.0);
    normal_distribution<> dist_noise(0.0, 0.3);


    for (int i = 0; i < n; i++) {
        x[i] = dist_x(gen);
        double noise = dist_noise(gen);
        y[i] = 0.1 * (x[i] - 4.0) * cos(x[i]) + 0.5 * x[i] + noise;
    }

    double best_beta = 0.0;
    double best_mse = numeric_limits<double>::max();


    for (double beta = 0.1; beta <= 2.0 + 1e-9; beta += 0.1) {

        double mse = loo_mse(x, y, beta);

        if (mse < best_mse) {
            best_mse = mse;
            best_beta = beta;
        }

        cout << "beta = " << beta << "  MSE = " << mse << endl;
    }

    cout << "\nОптимальное beta: " << best_beta << endl;
    cout << "Минимальный MSE: " << best_mse << endl;

    return 0;
}
