#include "mutual_information.h"

/*!
 * \brief Generate a 1D Gaussian kernel.
 * \param sigma Standard deviation.
 * \return A Gaussian filter kernel with radius `r` and size `2r + 1`.
 */
std::vector<double> gaussian_kernel(const double sigma)
{
    int r = std::ceil(2 * sigma); // filter radius
    std::vector<double> kernel (2 * r + 1);

    const double k = -1.0 / (2.0 * sigma * sigma);
    double sum = 0.0;
    for (int i = 0; i < 2*r + 1; ++i) {
        kernel[i] = std::exp(k * (i - r) * (i - r));
        sum += kernel[i];
    }

    // Normalise
    for (int i = 0; i < 2*r + 1; ++i) {
        kernel[i] /= sum;
    }

    return kernel;
}

