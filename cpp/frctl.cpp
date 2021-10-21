#include <complex>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

const int WIDTH = 640;
const int HEIGHT = 480;

const double EPSILON = 1e-30;
const int ITERATIONS = 30;

const complex<double> ONEC(1, 0); 

const int NROOTS = 3;
const complex<double> ROOTS[NROOTS] = {
    {1.0, 0.0},
    {-0.5, sqrt(3.0) / 2.0},
    {-0.5, -sqrt(3.0) / 2.0}
};

const Scalar COLOURS[NROOTS] = {
    Scalar(255, 0, 0),
    Scalar(0, 255, 0),
    Scalar(0, 0, 255)
};

Mat fractal_image(
    int w,
    int h,
    double xi,
    double xa,
    double yi,
    double ya,
    int n,
    double k
) {
    Mat image(h, w, CV_8UC3, Scalar(0, 0, 0));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            complex<double> z(
                xi + (xa - xi) * x / w,
                yi + (ya - yi) * y / h
            );

            for (int i = 0; i < n; i++) {
                z -= (pow(z, k) - ONEC) / (k * pow(z, k - 1.0));
            }

            for (int i = 0; i < NROOTS; i++) {
                if (abs(z - ROOTS[i]) < EPSILON) {
                    image.at<uchar>(y, x, i) = 0xFF;
                    break;
                }
            }
        }
    }

    return image;
}

int main(int argc, char** argv) {
    Mat image = fractal_image(
        WIDTH, HEIGHT, -2.5, 1.0, -1, 1.0, ITERATIONS, 3.0
    );

    imwrite("out.png", image);

    image.release();
}
