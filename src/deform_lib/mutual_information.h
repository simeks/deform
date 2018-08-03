#pragma once

#include <stk/common/assert.h>
#include <stk/image/volume.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>

#include <memory>
#include <tuple>
#include <vector>

std::vector<double> gaussian_kernel(const double sigma);

template<typename T>
class EntropyTerm {
public:

    /*!
     * \brief Build an entropy term object.
     *
     * This object represents a the contribution of a single voxel to
     * the entropy of an image.
     *
     * \param volume Image.
     * \param bins Number of bins.
     * \param sigma Kernel standard deviation.
     */
    EntropyTerm(const stk::VolumeHelper<T>& volume,
                const int bins,
                const double sigma) :
        bins(bins),
        sigma(sigma),
        data(bins)
    {
        update(volume);
    }

    /*!
     * \brief Compute the entropy on the given volume.
     *
     * Entropy is approximated with a first order Taylor polynomial and
     * the probability distribution is estimated with Parzen KDE, which
     * in turn is approximated with a smoothed histogram.
     *
     * This approximation was introuduced in:
     *   Kim, Junhwan et al. (2003): Visual correspondence using energy
     *   minimization and mutual information, Proceedings of the Ninth
     *   IEEE International Conference on Computer Vision, 1033–1040.
     *
     * \param volume Image.
     */
    void update(const stk::VolumeHelper<T>& volume)
    {
        stk::find_min_max(volume, min, max);
        T range = max - min;
        inv_bin_width = static_cast<T>(bins) / range;
        dim3 size = volume.size();

        // Histogram count
        std::fill(data.begin(), data.end(), 0.0);
        for (size_t z = 0; z < size.z; ++z) {
            for (size_t y = 0; y < size.y; ++y) {
                for (size_t x = 0; x < size.x; ++x) {
                    int i = static_cast<int>(inv_bin_width * (volume(x, y, z) - min));
                    ++at(i);
                }
            }
        }

        // To probability (term inside the log)
        const int N = size.x * size.y * size.z;
        for (auto& x : data) {
            x /= N;
        }

        // Approximate PDF of the term inside the log
        gaussian(sigma);

        // Log
        for (auto& x : data) {
            x = std::log(x < 1e-8 ? 1e-8 : x);
        }

        // Approximate PDF of the term outside the log
        gaussian(sigma);

        // To probability (term outside the log)
        for (auto& x : data) {
            x /= N;
        }
    }

    /*!
     * \brief Retrieve the entropy term for a given intensity.
     * \param x Intensity.
     * \return Entropy term.
     */
    double operator()(const T x) const {
        const int i = bounded(static_cast<int>(inv_bin_width * (x - min)));
        return data.data()[i];
    }

private:
    const int bins;           /*!< Number of bins. */
    const double sigma;       /*!< Kernel standard deviation. */
    std::vector<double> data; /*!< Binned data. */
    T min, max;               /*!< Intensity extrema. */
    T inv_bin_width;          /*!< Inverse of the bin width. */

    /*!
     * \brief Bound a bin index within a valid range.
     * \param i Input coordinate.
     * \return Closest value to `i` in `{0, ..., bins}`.
     */
    int bounded(const int i) const {
        return std::min<int>(std::max<int>(0, i), bins - 1);
    };

    /*!
     * \brief Return a reference to the bin.
     * \param i Bin index.
     * \return A reference to the bin.
     */
    double& at(const int i) {
        return data.data()[bounded(i)];
    }

    /*!
     * \brief Gaussian convolution.
     * \param sigma Kernel badwidth.
     */
    void gaussian(const double sigma)
    {
        if (sigma <= 0.0) {
            return;
        }

        // Compute a 1D filter kernel of adaptive size
        std::vector<double> kernel = gaussian_kernel(sigma);
        int r = kernel.size() / 2;

        // Apply the filter
        std::vector<double> tmp(bins);
        #pragma omp parallel for
        for (int i = 0; i < bins; ++i) {
            double val = 0.0;
            for (int t = -r; t < r + 1; ++t) {
                val += kernel[t+r] * data.data()[bounded(i+t)];
            }
            tmp[i] = val;
        }

        data = std::move(tmp);
    }
};


template<typename T>
class JointEntropyTerm {
public:

    /*!
     * \brief Build a joint entropy term object.
     *
     * This object represents a the contribution of a single voxel to
     * the joint entropy of the images.
     *
     * \param volume1 First image.
     * \param volume2 Second image.
     * \param bins Number of bins on each dimension.
     * \param sigma Kernel standard deviation.
     */
    JointEntropyTerm(const stk::VolumeHelper<T>& volume1,
                     const stk::VolumeHelper<T>& volume2,
                     const int bins,
                     const double sigma) :
        bins(bins),
        sigma(sigma),
        data(bins * bins)
    {
        update(volume1, volume2);
    }

    /*!
     * \brief Compute the joint entropy of two images.
     *
     * Joint entropy is approximated with a first order
     * Taylor polynomial and the probability distribution is estimated
     * with Parzen KDE, which in turn is approximated with a smoothed
     * histogram.
     *
     * This approximation was introuduced in:
     *   Kim, Junhwan et al. (2003): Visual correspondence using energy
     *   minimization and mutual information, Proceedings of the Ninth
     *   IEEE International Conference on Computer Vision, 1033–1040.
     *
     * \param volume1 First image.
     * \param volume2 Second image.
     */
    void update(const stk::VolumeHelper<T>& volume1, const stk::VolumeHelper<T>& volume2)
    {
        stk::find_min_max(volume1, min1, max1);
        stk::find_min_max(volume2, min2, max2);
        T range1 = max1 - min1;
        T range2 = max2 - min2;
        inv_bin_width_1 = static_cast<T>(bins) / range1;
        inv_bin_width_2 = static_cast<T>(bins) / range2;
        dim3 size = volume1.size();

        // Joint histogram count
        std::fill(data.begin(), data.end(), 0.0);
        for (size_t z = 0; z < size.z; ++z) {
            for (size_t y = 0; y < size.y; ++y) {
                for (size_t x = 0; x < size.x; ++x) {
                    // [1] -> [world] -> [2]
                    float3 p1 {float(x), float(y), float(z)};
                    float3 pw = volume1.origin() + p1 * volume1.spacing();
                    float3 p2 = (pw - volume2.origin()) / volume2.spacing();

                    T v1 = volume1(x, y, z);
                    T v2 = volume2.linear_at(p2, stk::Border_Replicate);

                    int i1 = static_cast<int>(inv_bin_width_1 * (v1 - min1));
                    int i2 = static_cast<int>(inv_bin_width_2 * (v2 - min2));

                    ++at(i1, i2);
                }
            }
        }

        // To probability (term inside the log)
        const int N = size.x * size.y * size.z;
        for (auto& x : data) {
            x /= N;
        }

        // Approximate PDF of the term inside the log
        gaussian(sigma);

        // Log
        for (auto& x : data) {
            x = std::log(x < 1e-8 ? 1e-8 : x);
        }

        // Approximate PDF of the term outside the log
        gaussian(sigma);

        // To probability (term outside the log)
        for (auto& x : data) {
            x /= N;
        }
    }

    /*!
     * \brief Retrieve the joint entropy term for a given combination of intensities.
     * \param x Intensity on the first image.
     * \param y Intensity on the second image.
     * \return Joint entropy term.
     */
    double operator()(const T x, const T y) const {
        const int i = bounded(static_cast<int>(inv_bin_width_1 * (x - min1)));
        const int j = bounded(static_cast<int>(inv_bin_width_2 * (y - min2)));
        return data.data()[bins * j + i];
    }

private:
    int bins;     /*!< Number of bins. */
    double sigma; /*!< Kernel standard deviation. */
    std::vector<double> data; /*!< Binned data. */
    T min1, max1, min2, max2; /*!< Extrema for the two intensities. */
    T inv_bin_width_1, inv_bin_width_2; /*!< Inverse of the bin widths for each direction. */

    /*!
     * \brief Bound a bin index within a valid range.
     * \param i Input coordinate.
     * \return Closest value to `i` in `{0, ..., bins}`.
     */
    int bounded(const int i) const {
        return std::min<int>(std::max<int>(0, i), bins - 1);
    };

    /*!
     * \brief Return a reference to the bin.
     * \param i First bin coordinate.
     * \param j Second bin coordinate.
     * \return A reference to the bin.
     */
    double& at(const int i, const int j) {
        return data.data()[bins * bounded(j) + bounded(i)];
    }

    /*!
     * \brief Gaussian convolution with decomposed filters.
     * \param sigma Standard deviation.
     */
    void gaussian(const double sigma)
    {
        if (sigma <= 0.0) {
            return;
        }

        // Compute a 1D filter kernel of adaptive size
        std::vector<double> kernel = gaussian_kernel(sigma);
        int r = kernel.size() / 2;

        // Apply the filter along the x direction
        double tmp[bins][bins];
        #pragma omp parallel for
        for (int i = 0; i < bins; ++i) {
            for (int j = 0; j < bins; ++j) {
                double val = 0.0;
                for (int t = -r; t < r + 1; ++t) {
                    val += kernel[t+r] * at(i, j+t);
                }
                tmp[j][i] = val;
            }
        }

        // Apply the filter along the y direction
        #pragma omp parallel for
        for (int i = 0; i < bins; ++i) {
            for (int j = 0; j < bins; ++j) {
                double val = 0.0;
                for (int t = -r; t < r + 1; ++t) {
                    val += kernel[t+r] * tmp[j][bounded(i+t)];
                }
                at(i, j) = val;
            }
        }
    }
};

