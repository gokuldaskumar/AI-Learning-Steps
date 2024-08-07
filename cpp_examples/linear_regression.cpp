#include <iostream>
#include <Eigen/Dense>
#include <random>

using namespace Eigen;
using namespace std;

int main() {
    // Set random seed for reproducibility
    srand(0);

    // Generate sample data
    int n = 100;  // Number of data points
    VectorXd X = VectorXd::Random(n);  // Input features
    // Create target values: y = 2 + 3x + noise
    VectorXd y = 2 + 3 * X + VectorXd::Random(n) * 0.1;

    // Prepare design matrix
    // X_design = [1, X] to include intercept term
    MatrixXd X_design(n, 2);
    X_design << VectorXd::Ones(n), X;

    // Compute coefficients using normal equation
    // coeffs = (X^T * X)^-1 * X^T * y
    VectorXd coeffs = (X_design.transpose() * X_design).ldlt().solve(X_design.transpose() * y);

    // Print results
    cout << "Intercept: " << coeffs(0) << endl;
    cout << "Slope: " << coeffs(1) << endl;

    // Make predictions
    VectorXd X_test(2);
    X_test << 1, 0.5;  // Predict for x = 0.5
    double y_pred = X_test.dot(coeffs);  // y = b0 + b1 * x
    cout << "Prediction for X=0.5: " << y_pred << endl;

    return 0;
}
