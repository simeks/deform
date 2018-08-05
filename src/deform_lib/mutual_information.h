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
        _bins(bins),
        _sigma(sigma),
        _data(bins)
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
        stk::find_min_max(volume, _min, _max);
        T range = _max - _min;
        _inv_bin_width = static_cast<T>(_bins) / range;
        dim3 size = volume.size();

        // Histogram count
        std::fill(_data.begin(), _data.end(), 0.0);
        for (size_t z = 0; z < size.z; ++z) {
            for (size_t y = 0; y < size.y; ++y) {
                for (size_t x = 0; x < size.x; ++x) {
                    int i = static_cast<int>(_inv_bin_width * (volume(x, y, z) - _min));
                    ++at(i);
                }
            }
        }

        // To probability (term inside the log)
        const int N = size.x * size.y * size.z;
        for (auto& x : _data) {
            x /= N;
        }

        // Approximate PDF of the term inside the log
        gaussian();

        // Log
        for (auto& x : _data) {
            x = std::log(x < 1e-8 ? 1e-8 : x);
        }

        // Approximate PDF of the term outside the log
        gaussian();

        // To probability (term outside the log)
        for (auto& x : _data) {
            x /= N;
        }
    }

    /*!
     * \brief Retrieve the entropy term for a given intensity.
     * \param x Intensity.
     * \return Entropy term.
     */
    double operator()(const T x) const {
        const int i = bounded(static_cast<int>(_inv_bin_width * (x - _min)));
        return _data.data()[i];
    }

private:
    const int _bins;           /*!< Number of bins. */
    const double _sigma;       /*!< Kernel standard deviation. */
    std::vector<double> _data; /*!< Binned data. */
    T _min, _max;               /*!< Intensity extrema. */
    T _inv_bin_width;          /*!< Inverse of the bin width. */

    /*!
     * \brief Bound a bin index within a valid range.
     * \param i Input coordinate.
     * \return Closest value to `i` in `{0, ..., bins}`.
     */
    int bounded(const int i) const {
        return std::min<int>(std::max<int>(0, i), _bins - 1);
    };

    /*!
     * \brief Return a reference to the bin.
     * \param i Bin index.
     * \return A reference to the bin.
     */
    double& at(const int i) {
        return _data.data()[bounded(i)];
    }

    /*!
     * \brief Gaussian convolution.
     */
    void gaussian(void)
    {
        if (_sigma <= 0.0) {
            return;
        }

        // Compute a 1D filter kernel of adaptive size
        std::vector<double> kernel = gaussian_kernel(_sigma);
        int r = kernel.size() / 2;

        // Apply the filter
        std::vector<double> tmp(_bins);
        #pragma omp parallel for
        for (int i = 0; i < _bins; ++i) {
            double val = 0.0;
            for (int t = -r; t < r + 1; ++t) {
                val += kernel[t+r] * _data.data()[bounded(i+t)];
            }
            tmp[i] = val;
        }

        _data = std::move(tmp);
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
        _bins(bins),
        _sigma(sigma),
        _data(bins * bins)
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
        stk::find_min_max(volume1, _min1, _max1);
        stk::find_min_max(volume2, _min2, _max2);
        T range1 = _max1 - _min1;
        T range2 = _max2 - _min2;
        _inv_bin_width_1 = static_cast<T>(_bins) / range1;
        _inv_bin_width_2 = static_cast<T>(_bins) / range2;
        dim3 size = volume1.size();

        // Joint histogram count
        std::fill(_data.begin(), _data.end(), 0.0);
        for (size_t z = 0; z < size.z; ++z) {
            for (size_t y = 0; y < size.y; ++y) {
                for (size_t x = 0; x < size.x; ++x) {
                    // [1] -> [world] -> [2]
                    float3 p1 {float(x), float(y), float(z)};
                    float3 pw = volume1.origin() + p1 * volume1.spacing();
                    float3 p2 = (pw - volume2.origin()) / volume2.spacing();

                    T v1 = volume1(x, y, z);
                    T v2 = volume2.linear_at(p2, stk::Border_Replicate);

                    int i1 = static_cast<int>(_inv_bin_width_1 * (v1 - _min1));
                    int i2 = static_cast<int>(_inv_bin_width_2 * (v2 - _min2));

                    ++at(i1, i2);
                }
            }
        }

        // To probability (term inside the log)
        const int N = size.x * size.y * size.z;
        for (auto& x : _data) {
            x /= N;
        }

        // Approximate PDF of the term inside the log
        gaussian();

        // Log
        for (auto& x : _data) {
            x = std::log(x < 1e-8 ? 1e-8 : x);
        }

        // Approximate PDF of the term outside the log
        gaussian();

        // To probability (term outside the log)
        for (auto& x : _data) {
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
        const int i = bounded(static_cast<int>(_inv_bin_width_1 * (x - _min1)));
        const int j = bounded(static_cast<int>(_inv_bin_width_2 * (y - _min2)));
        return _data.data()[_bins * j + i];
    }

private:
    int _bins;     /*!< Number of bins. */
    double _sigma; /*!< Kernel standard deviation. */
    std::vector<double> _data; /*!< Binned data. */
    T _min1, _max1, _min2, _max2; /*!< Extrema for the two intensities. */
    T _inv_bin_width_1, _inv_bin_width_2; /*!< Inverse of the bin widths for each direction. */

    /*!
     * \brief Bound a bin index within a valid range.
     * \param i Input coordinate.
     * \return Closest value to `i` in `{0, ..., bins}`.
     */
    int bounded(const int i) const {
        return std::min<int>(std::max<int>(0, i), _bins - 1);
    };

    /*!
     * \brief Return a reference to the bin.
     * \param i First bin coordinate.
     * \param j Second bin coordinate.
     * \return A reference to the bin.
     */
    double& at(const int i, const int j) {
        return _data.data()[_bins * bounded(j) + bounded(i)];
    }

    /*!
     * \brief Gaussian convolution with decomposed filters.
     */
    void gaussian(void)
    {
        if (_sigma <= 0.0) {
            return;
        }

        // Compute a 1D filter kernel of adaptive size
        std::vector<double> kernel = gaussian_kernel(_sigma);
        int r = kernel.size() / 2;

        // Apply the filter along the x direction
        double tmp[_bins][_bins];
        #pragma omp parallel for
        for (int i = 0; i < _bins; ++i) {
            for (int j = 0; j < _bins; ++j) {
                double val = 0.0;
                for (int t = -r; t < r + 1; ++t) {
                    val += kernel[t+r] * at(i, j+t);
                }
                tmp[j][i] = val;
            }
        }

        // Apply the filter along the y direction
        #pragma omp parallel for
        for (int i = 0; i < _bins; ++i) {
            for (int j = 0; j < _bins; ++j) {
                double val = 0.0;
                for (int t = -r; t < r + 1; ++t) {
                    val += kernel[t+r] * tmp[j][bounded(i+t)];
                }
                at(i, j) = val;
            }
        }
    }
};

