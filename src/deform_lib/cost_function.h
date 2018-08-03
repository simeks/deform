#pragma once

#include "config.h"
#include "registration/transform.h"
#include "mutual_information.h"

#include <stk/common/assert.h>
#include <stk/image/volume.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>

#include <memory>
#include <tuple>
#include <vector>

struct Regularizer
{
    Regularizer(float weight=0.0f, const float3& spacing={1.0f, 1.0f, 1.0f}) :
        _weight(weight), _spacing(spacing)
    {
    }

    void set_regularization_weight(float weight)
    {
        _weight = weight;
    }
    void set_fixed_spacing(const float3& spacing)
    {
        _spacing = spacing;
    }

    // Sets the initial displacement for this registration level. This will be
    //  the reference when computing the regularization energy. Any displacement 
    //  identical to the initial displacement will result in zero energy.
    void set_initial_displacement(const stk::VolumeFloat3& initial)
    {
        _initial = initial;
    }

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    void set_weight_map(stk::VolumeFloat& map) { _weight_map = map; }
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

    /// p   : Position in fixed image
    /// def0 : Deformation in active voxel [mm]
    /// def1 : Deformation in neighbor [mm]
    /// step : Direction to neighbor
    inline double operator()(const int3& p, const float3& def0,
                             const float3& def1, const int3& step)
    {
        float3 step_in_mm {
            step.x*_spacing.x, 
            step.y*_spacing.y, 
            step.z*_spacing.z
        };
        
        // The diff should be relative to the initial displacement diff
        float3 diff = (def0-_initial(p)) - (def1-_initial(p+step));
        
        float dist_squared = stk::norm2(diff);
        float step_squared = stk::norm2(step_in_mm);
        
        float w = _weight;

        #ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
            /*
                Tissue-specific regularization
                Per edge weight: Mean term of neighboring voxels 

                w = 0.5f*(weights(p) + weights(p+step)) 
            */

            if (_weight_map.valid())
                w = 0.5f*(_weight_map(p) + _weight_map(p+step));
        #endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

        return w * dist_squared / step_squared;
    }

    float _weight;
    float3 _spacing;

    stk::VolumeFloat3 _initial;

    #ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        stk::VolumeFloat _weight_map;
    #endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP
};

struct SubFunction
{
    /*!
     * \brief Cost term for a single voxel.
     * \param p Indices of the voxel in the reference image.
     * \param def Tentative displacement for the voxel.
     * \return The cost associated to displacing `p` by `def`.
     */
    virtual float cost(const int3& p, const float3& def) = 0;

    /*!
     * \brief A callback that is executed after each iteration of the solver.
     * \param iteration Number of the iteration just completed.
     * \param def Deformation field at the end of the iteration.
     */
    virtual void post_iteration_hook(const int iteration, const stk::VolumeFloat3& def)
    {
        (void) iteration;
        (void) def;
    }
};


struct SoftConstraintsFunction : public SubFunction
{
    SoftConstraintsFunction(const stk::VolumeUChar& constraint_mask,
                            const stk::VolumeFloat3& constraints_values) :
        _constraints_mask(constraint_mask),
        _constraints_values(constraints_values),
        _spacing(_constraints_values.spacing())
    {}

    float cost(const int3& p, const float3& def) 
    {
        if (_constraints_mask(p) != 0)
        {
            float3 diff = def - _constraints_values(p);
            
            // Distance^2 in [mm]
            float dist_squared = stk::norm2(diff);
            
            return std::min(dist_squared, 1000.0f); // Clamp to avoid explosion
        }
        return 0.0f;
    }
    stk::VolumeUChar _constraints_mask;
    stk::VolumeFloat3 _constraints_values;
    float3 _spacing;
};


template<typename T>
struct SquaredDistanceFunction : public SubFunction
{
    SquaredDistanceFunction(const stk::VolumeHelper<T>& fixed,
                            const stk::VolumeHelper<T>& moving) :
        _fixed(fixed),
        _moving(moving)
    {}

    float cost(const int3& p, const float3& def)
    {
        float3 fixed_p{
            float(p.x),
            float(p.y),
            float(p.z)
        };
        
        // [fixed] -> [world] -> [moving]
        float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
        float3 moving_p = (world_p + def - _moving.origin()) / _moving.spacing();

        T moving_v = _moving.linear_at(moving_p, stk::Border_Constant);

        // [Filip]: Addition for partial-body registrations
        if (moving_p.x<0 || moving_p.x>_moving.size().x || 
            moving_p.y<0 || moving_p.y>_moving.size().y || 
            moving_p.z<0 || moving_p.z>_moving.size().z) {
            return 0;
        }


        // TODO: Float cast
        float f = fabs(float(_fixed(p) - moving_v));
        return f*f;
    }

    stk::VolumeHelper<T> _fixed;
    stk::VolumeHelper<T> _moving;
};

template<typename T>
struct NCCFunction : public SubFunction
{
    NCCFunction(const stk::VolumeHelper<T>& fixed,
                const stk::VolumeHelper<T>& moving,
                const int radius) :
        _fixed(fixed),
        _moving(moving),
        _radius(radius)
    {}


    float cost(const int3& p, const float3& def)
    {
        float3 fixed_p{
            float(p.x),
            float(p.y),
            float(p.z)
        }; 
        
        // [fixed] -> [world] -> [moving]
        float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
        float3 moving_p = (world_p + def - _moving.origin()) / _moving.spacing();

        // [Filip]: Addition for partial-body registrations
        if (moving_p.x<0 || moving_p.x>_moving.size().x || 
            moving_p.y<0 || moving_p.y>_moving.size().y || 
            moving_p.z<0 || moving_p.z>_moving.size().z) {
            return 0;
        }

        double sff = 0.0;
        double smm = 0.0;
        double sfm = 0.0;
        double sf = 0.0;
        double sm = 0.0;
        size_t n = 0;

        for (int dz = -_radius; dz <= _radius; ++dz) {
            for (int dy = -_radius; dy <= _radius; ++dy) {
                for (int dx = -_radius; dx <= _radius; ++dx) {
                    // TODO: Does not account for anisotropic volumes
                    int r2 = dx*dx + dy*dy + dz*dz;
                    if (r2 > 4)
                        continue;

                    int3 fp{p.x + dx, p.y + dy, p.z + dz};
                    
                    if (!stk::is_inside(_fixed.size(), fp))
                        continue;

                    float3 mp{moving_p.x + dx, moving_p.y + dy, moving_p.z + dz};

                    T fixed_v = _fixed(fp);
                    T moving_v = _moving.linear_at(mp, stk::Border_Constant);

                    sff += fixed_v * fixed_v;
                    smm += moving_v * moving_v;
                    sfm += fixed_v*moving_v;
                    sm += moving_v;
                    sf += fixed_v;

                    ++n;
                }
            }
        }

        if (n == 0)
            return 0.0f;

        // Subtract mean
        sff -= (sf * sf / n);
        smm -= (sm * sm / n);
        sfm -= (sf * sm / n);
        
        double d = sqrt(sff*smm);

        if(d > 1e-14) {
            return 0.5f*(1.0f-float(sfm / d));
        }
        return 0.0f;
    }

    stk::VolumeHelper<T> _fixed;
    stk::VolumeHelper<T> _moving;
    const int _radius;
};


struct LandmarksFunction : public SubFunction
{
    LandmarksFunction(const std::vector<float3>& fixed_landmarks,
                      const std::vector<float3>& moving_landmarks,
                      const float3& fixed_origin,
                      const float3& fixed_spacing,
                      const dim3& fixed_size) :
        landmarks {fixed_landmarks},
        fixed_origin {fixed_origin},
        fixed_spacing {fixed_spacing},
        fixed_size {fixed_size}
    {
        ASSERT(fixed_landmarks.size() == moving_landmarks.size());
        for (size_t i = 0; i < fixed_landmarks.size(); ++i) {
            displacements.push_back(moving_landmarks[i] - fixed_landmarks[i]);
        }
    }

    // TODO: precompute distance map in the constructor
    float cost(const int3& p, const float3& def)
    {
        float cost = 0.0f;
        const float epsilon = 1e-6f;

        const float3 fixed_p{
            static_cast<float>(p.x),
            static_cast<float>(p.y),
            static_cast<float>(p.z)
        };

        const float3 world_p = fixed_origin + fixed_p * fixed_spacing;

        for (size_t i = 0; i < landmarks.size(); ++i) {
            cost += stk::norm2(def - displacements[i]) /
                    (stk::norm2(landmarks[i] - world_p) + epsilon);
        }

        return cost;
    }

    const std::vector<float3> landmarks;
    std::vector<float3> displacements;
    const float3 fixed_origin;
    const float3 fixed_spacing;
    const dim3 fixed_size;
};


template<typename T>
struct MIFunction : public SubFunction
{
    MIFunction(const stk::VolumeHelper<T>& fixed,
               const stk::VolumeHelper<T>& moving,
               const int bins,
               const double sigma) :
        _fixed(fixed),
        _moving(moving),
        _bins(bins),
        _sigma(sigma),
        joint_entropy(fixed, moving, bins, sigma),
        entropy(moving, bins, sigma)
    {
    }

    /*!
     * \brief Contribution of a single voxel to the mutual information.
     *
     * Mutual information is broken to a voxel-wise sum thanks to an
     * approximation of the entropy function based on Taylor polynomia
     * and Parzen KDE. The formula was introduced in:
     *   Kim, Junhwan et al. (2003): Visual correspondence using energy
     *   minimization and mutual information, Proceedings of the Ninth
     *   IEEE International Conference on Computer Vision, 1033–1040.
     *
     * Here, both the joint entropy and the entropy of the moving image
     * are approximated. The entropy of the fixed image is ignored
     * since it is constant with respect to the displacement, hence it
     * has no effect on the optimisation process.
     */
    float cost(const int3& p, const float3& def)
    {
        const float3 fixed_p{float(p.x), float(p.y), float(p.z)};

        // [fixed] -> [world] -> [moving]
        float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
        float3 moving_p = (world_p + def - _moving.origin()) / _moving.spacing();

        T i1 = _fixed(p);
        T i2 = _moving.linear_at(moving_p, stk::Border_Constant);

        // NOTE: the sign is inverted (minimising negated MI)
        return entropy(i2) - joint_entropy(i1, i2);
    }

    /*!
     * \brief Update the entropy term estimations.
     *
     * The entropy terms are approximated with a first order truncated
     * Taylor polynomial, which is a function of the displacement. The
     * approximation gets worse as the displacement gets larger. To
     * compensate for this, resample the moving volume and update the
     * entropy of the moving image and the joint entropy after each
     * iteration.
     */
    virtual void post_iteration_hook(const int iteration, const stk::VolumeFloat3& def)
    {
        (void) iteration;
        auto tmp = transform_volume(_moving, def, transform::Interp_NN);
        joint_entropy.update(_fixed, tmp);
        entropy.update(tmp);
    }

    stk::VolumeHelper<T> _fixed;
    stk::VolumeHelper<T> _moving;
    int _bins;
    double _sigma;

private:
    JointEntropyTerm<T> joint_entropy;
    EntropyTerm<T> entropy;
};


struct UnaryFunction
{
    struct WeightedFunction {
        float weight;
        std::unique_ptr<SubFunction> function;
    };

    UnaryFunction(float regularization_weight=0.0f) : 
        _regularization_weight(regularization_weight)
    {
    }
    void set_regularization_weight(float weight)
    {
        _regularization_weight = weight;
    }
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    void set_regularization_weight_map(stk::VolumeFloat& map) 
    {
        _regularization_weight_map = map;
    }
#endif

    void add_function(std::unique_ptr<SubFunction> fn, float weight)
    {
        functions.push_back({weight, std::move(fn)});
    }

    inline double operator()(const int3& p, const float3& def)
    {
        double sum = 0.0f;
        for (auto& fn : functions) {
            sum += fn.weight * fn.function->cost(p, def);
        }

        float w = _regularization_weight;
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        if (_regularization_weight_map.valid())
            w = _regularization_weight_map(p);
#endif

        return (1.0f-w)*sum;
    }

    void post_iteration_hook(const int iteration, const stk::VolumeFloat3& def) {
        for (auto& fn : functions) {
            fn.function->post_iteration_hook(iteration, def);
        }
    }

    float _regularization_weight;
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    stk::VolumeFloat _regularization_weight_map;
#endif

    std::vector<WeightedFunction> functions;
};


