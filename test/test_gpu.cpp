#include "catch.hpp"

#ifdef DF_USE_CUDA

#include <deform_lib/filters/resample.h>
#include <deform_lib/filters/gpu/resample.h>

#include <deform_lib/filters/gaussian_filter.h>
#include <deform_lib/filters/gpu/gaussian_filter.h>

#include <deform_lib/registration/transform.h>
#include <deform_lib/registration/gpu/gpu_displacement_field.h>
#include <deform_lib/registration/gpu/transform.h>

#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>
#include <stk/io/io.h>

TEST_CASE("gpu_gaussian_filter", "")
{
    stk::VolumeFloat src({16,16,16});

    for (int z = 0; z < (int)src.size().z; ++z)
    for (int y = 0; y < (int)src.size().y; ++y)
    for (int x = 0; x < (int)src.size().x; ++x) {

        int r = (x-7)*(x-7) + (y-7)*(y-7) + (z-7)*(z-7);
        if (r < 8*8) {
            src(x,y,z) = 1.0f;
        }
        else {
            src(x,y,z) = 0.0f;
        }
    }

    // Use CPU-version as ground truth
    stk::VolumeFloat out = filters::gaussian_filter_3d(src, 2.5f);

    stk::GpuVolume gpu_src(src);
    stk::VolumeFloat gpu_out = filters::gpu::gaussian_filter_3d(gpu_src, 2.5f).download();

    CHECK(gpu_out.spacing().x == Approx(out.spacing().x));
    CHECK(gpu_out.spacing().y == Approx(out.spacing().y));
    CHECK(gpu_out.spacing().z == Approx(out.spacing().z));

    CHECK(gpu_out.origin().x == Approx(out.origin().x));
    CHECK(gpu_out.origin().y == Approx(out.origin().y));
    CHECK(gpu_out.origin().z == Approx(out.origin().z));

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            CHECK(gpu_out.direction()(i, j) == Approx(out.direction()(i, j)));
        }
    }

    for (int z = 0; z < (int)src.size().z; ++z)
    for (int y = 0; y < (int)src.size().y; ++y)
    for (int x = 0; x < (int)src.size().x; ++x) {
        REQUIRE(gpu_out(x,y,z) == Approx(out(x,y,z)));
    }
}

TEST_CASE("gpu_downsample_gaussian", "")
{
    stk::VolumeFloat src({16,16,16});

    for (int z = 0; z < (int)src.size().z; ++z)
    for (int y = 0; y < (int)src.size().y; ++y)
    for (int x = 0; x < (int)src.size().x; ++x) {

        int r = (x-7)*(x-7) + (y-7)*(y-7) + (z-7)*(z-7);
        if (r < 8*8) {
            src(x,y,z) = 10.0f;
        }
        else {
            src(x,y,z) = 1.0f;
        }
    }

    // Use CPU-version as ground truth
    stk::VolumeFloat out = filters::downsample_volume_by_2(src);

    stk::GpuVolume gpu_src(src);
    stk::VolumeFloat gpu_out = filters::gpu::downsample_volume_by_2(gpu_src).download();

    CHECK(gpu_out.spacing().x == Approx(out.spacing().x));
    CHECK(gpu_out.spacing().y == Approx(out.spacing().y));
    CHECK(gpu_out.spacing().z == Approx(out.spacing().z));

    CHECK(gpu_out.origin().x == Approx(out.origin().x));
    CHECK(gpu_out.origin().y == Approx(out.origin().y));
    CHECK(gpu_out.origin().z == Approx(out.origin().z));

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            CHECK(gpu_out.direction()(i, j) == Approx(out.direction()(i, j)));
        }
    }

    for (int z = 0; z < (int)out.size().z; ++z)
    for (int y = 0; y < (int)out.size().y; ++y)
    for (int x = 0; x < (int)out.size().x; ++x) {
        REQUIRE(gpu_out(x,y,z) == Approx(out(x,y,z)));
    }
}

TEST_CASE("gpu_downsample_vectorfield", "")
{
    stk::VolumeFloat3 src({16,16,16});
    stk::VolumeFloat4 src4({16,16,16}); // CUDA do not support float3 so we add an empty channel

    for (int z = 0; z < (int)src.size().z; ++z)
    for (int y = 0; y < (int)src.size().y; ++y)
    for (int x = 0; x < (int)src.size().x; ++x) {

        int r = (x-7)*(x-7) + (y-7)*(y-7) + (z-7)*(z-7);
        if (r < 6) {
            src(x,y,z) = {2.0f, 1.0f, 0.5f};
        }
        else {
            src(x,y,z) = {1.0f, 1.0f, 1.0f};
        }
        src4(x,y,z) = {src(x,y,z).x, src(x,y,z).y, src(x,y,z).z, 0.0f};
    }

    // Use CPU-version as ground truth
    stk::VolumeFloat3 out = filters::downsample_vectorfield_by_2(src);

    stk::GpuVolume gpu_src(src4);
    stk::VolumeFloat4 gpu_out = filters::gpu::downsample_vectorfield_by_2(gpu_src).download();

    CHECK(gpu_out.spacing().x == Approx(out.spacing().x));
    CHECK(gpu_out.spacing().y == Approx(out.spacing().y));
    CHECK(gpu_out.spacing().z == Approx(out.spacing().z));

    CHECK(gpu_out.origin().x == Approx(out.origin().x));
    CHECK(gpu_out.origin().y == Approx(out.origin().y));
    CHECK(gpu_out.origin().z == Approx(out.origin().z));

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            CHECK(gpu_out.direction()(i, j) == Approx(out.direction()(i, j)));
        }
    }

    for (int z = 0; z < (int)out.size().z; ++z)
    for (int y = 0; y < (int)out.size().y; ++y)
    for (int x = 0; x < (int)out.size().x; ++x) {
        REQUIRE(gpu_out(x,y,z).x == Approx(out(x,y,z).x));
        REQUIRE(gpu_out(x,y,z).y == Approx(out(x,y,z).y));
        REQUIRE(gpu_out(x,y,z).z == Approx(out(x,y,z).z));
        REQUIRE(gpu_out(x,y,z).w == Approx(0.0f));
    }
}

TEST_CASE("gpu_upsample_vectorfield", "")
{
    stk::VolumeFloat3 src({8,8,8});
    stk::VolumeFloat4 src4({8,8,8}); // CUDA do not support float3 so we add an empty channel

    for (int z = 0; z < (int)src.size().z; ++z)
    for (int y = 0; y < (int)src.size().y; ++y)
    for (int x = 0; x < (int)src.size().x; ++x) {
        int r = (x-3)*(x-3) + (y-3)*(y-3) + (z-3)*(z-3);
        if (r < 6) {
            src(x,y,z) = {2.0f, 1.0f, 0.5f};
        }
        else {
            src(x,y,z) = {1.0f, 1.0f, 1.0f};
        }
        src4(x,y,z) = {src(x,y,z).x, src(x,y,z).y, src(x,y,z).z, 0.0f};
    }

    // Use CPU-version as ground truth
    stk::VolumeFloat3 out = filters::upsample_vectorfield(src, {16,16,16});

    stk::GpuVolume gpu_src(src4);
    stk::VolumeFloat4 gpu_out = filters::gpu::upsample_vectorfield(gpu_src, {16,16,16}).download();

    CHECK(gpu_out.spacing().x == Approx(out.spacing().x));
    CHECK(gpu_out.spacing().y == Approx(out.spacing().y));
    CHECK(gpu_out.spacing().z == Approx(out.spacing().z));

    CHECK(gpu_out.origin().x == Approx(out.origin().x));
    CHECK(gpu_out.origin().y == Approx(out.origin().y));
    CHECK(gpu_out.origin().z == Approx(out.origin().z));

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            CHECK(gpu_out.direction()(i, j) == Approx(out.direction()(i, j)));
        }
    }

    for (int z = 0; z < (int)out.size().z; ++z)
    for (int y = 0; y < (int)out.size().y; ++y)
    for (int x = 0; x < (int)out.size().x; ++x) {
        REQUIRE(gpu_out(x,y,z).x == Approx(out(x,y,z).x));
        REQUIRE(gpu_out(x,y,z).y == Approx(out(x,y,z).y));
        REQUIRE(gpu_out(x,y,z).z == Approx(out(x,y,z).z));
        REQUIRE(gpu_out(x,y,z).w == Approx(0.0f));
    }
}


TEST_CASE("gpu_transform", "")
{
    stk::VolumeFloat src({8,8,8});
    src.set_origin(float3{1, 2, 3});
    src.set_spacing(float3{1.1f, 1.2f, 1.3f});
    stk::VolumeFloat3 def3({8,8,8}); // CUDA do not support float3 so we add an empty channel
    stk::VolumeFloat4 def4({8,8,8}); // CUDA do not support float3 so we add an empty channel
    def3.set_direction({{
            {0.999f, 0.001f, 0.00f},
            {0.001f, 0.988f, 0.00f},
            {0.000f, 0.000f, 0.99f}
    }});
    def4.set_direction(def3.direction());

    for (int z = 0; z < (int)src.size().z; ++z)
    for (int y = 0; y < (int)src.size().y; ++y)
    for (int x = 0; x < (int)src.size().x; ++x) {
        int r = (x-3)*(x-3) + (y-3)*(y-3) + (z-3)*(z-3);
        if (r < 6) {
            src(x,y,z) = float(r);
        }
        else {
            src(x,y,z) = 1.0f;
        }

        def3(x,y,z) = {1.0f, 2.0f, 3.0f};
        def4(x,y,z) = {1.0f, 2.0f, 3.0f, 0.0f};
    }

    // Use CPU-version as ground truth
    stk::VolumeFloat out_lin = transform_volume(src, def3, transform::Interp_Linear);
    stk::VolumeFloat out_nn = transform_volume(src, def3, transform::Interp_NN);

    stk::GpuVolume gpu_src(src);
    stk::GpuVolume gpu_def(def4);
    stk::VolumeFloat gpu_out_lin = gpu::transform_volume(gpu_src, GpuDisplacementField(gpu_def), transform::Interp_Linear)
        .download();
    stk::VolumeFloat gpu_out_nn = gpu::transform_volume(gpu_src, GpuDisplacementField(gpu_def), transform::Interp_NN)
        .download();

    CHECK(gpu_out_lin.spacing().x == Approx(out_lin.spacing().x));
    CHECK(gpu_out_lin.spacing().y == Approx(out_lin.spacing().y));
    CHECK(gpu_out_lin.spacing().z == Approx(out_lin.spacing().z));

    CHECK(gpu_out_lin.origin().x == Approx(out_lin.origin().x));
    CHECK(gpu_out_lin.origin().y == Approx(out_lin.origin().y));
    CHECK(gpu_out_lin.origin().z == Approx(out_lin.origin().z));

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            CHECK(gpu_out_lin.direction()(i, j) == Approx(gpu_def.direction()(i, j)));
        }
    }

    CHECK(gpu_out_nn.spacing().x == Approx(out_nn.spacing().x));
    CHECK(gpu_out_nn.spacing().y == Approx(out_nn.spacing().y));
    CHECK(gpu_out_nn.spacing().z == Approx(out_nn.spacing().z));

    CHECK(gpu_out_nn.origin().x == Approx(out_nn.origin().x));
    CHECK(gpu_out_nn.origin().y == Approx(out_nn.origin().y));
    CHECK(gpu_out_nn.origin().z == Approx(out_nn.origin().z));

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            CHECK(gpu_out_nn.direction()(i, j) == Approx(gpu_def.direction()(i, j)));
        }
    }

    for (int z = 0; z < (int)out_nn.size().z; ++z)
    for (int y = 0; y < (int)out_nn.size().y; ++y)
    for (int x = 0; x < (int)out_nn.size().x; ++x) {
        REQUIRE(gpu_out_lin(x,y,z) == Approx(out_lin(x,y,z)));
        REQUIRE(gpu_out_nn(x,y,z) == Approx(out_nn(x,y,z)));
    }
}

#include <deform_lib/registration/gpu/cost_functions/binary_function.h>
#include <deform_lib/registration/gpu/cost_functions/cost_function.h>
#include <deform_lib/registration/gpu/cost_functions/cross_correlation.h>
#include <deform_lib/registration/gpu/cost_functions/unary_function.h>
#include <deform_lib/registration/gpu/gpu_displacement_field.h>
#include <deform_lib/registration/cost_functions/regularizer.h>
#include <deform_lib/registration/cost_functions/unary_function.h>
#include <deform_lib/registration/cost_functions/cross_correlation_sphere.h>
#include <deform_lib/registration/affine_transform.h>
#include <deform_lib/registration/displacement_field.h>
#include <deform_lib/registration/settings.h>
#include <deform_lib/make_unique.h>
#include <stk/cuda/stream.h>

namespace {
    stk::VolumeFloat4 volume_float3_to_float4(const stk::VolumeFloat3& df)
    {
        dim3 dims = df.size();
        stk::VolumeFloat4 out(dims);
        out.copy_meta_from(df);

        for (int z = 0; z < (int)dims.z; ++z) {
        for (int y = 0; y < (int)dims.y; ++y) {
        for (int x = 0; x < (int)dims.x; ++x) {
            out(x,y,z) = float4{df(x,y,z).x, df(x,y,z).y, df(x,y,z).z, 0.0f};
        }}}

        return out;
    }
}

TEST_CASE("gpu_binary_term", "")
{
    float3 delta {0.5f, 0.0f, 0.0f};
    dim3 dims { 32, 32, 32 };
    stk::VolumeFloat3 vectorfield(dims, float3{0, 0, 0});
    for (int3 p : dims) {
        vectorfield(p) = float3{
            (rand() % 100) / 10.0f,
            (rand() % 100) / 10.0f,
            (rand() % 100) / 10.0f
        };
    }
    stk::GpuVolume gpu_vectorfield(volume_float3_to_float4(vectorfield));

    Matrix3x3f matrix {
        float3{2, 0, 0},
        float3{0, 3, 0},
        float3{0, 0, 4}
    };

    AffineTransform affine_transform(matrix, float3{5,-6,7});

    Settings::UpdateRule update_rules[] = {
        Settings::UpdateRule_Additive,
        Settings::UpdateRule_Compositive
    };
    for (auto update_rule : update_rules) {

        stk::GpuVolume gpu_binary_cost_x(dims, stk::Type_Float4);
        stk::GpuVolume gpu_binary_cost_y(dims, stk::Type_Float4);
        stk::GpuVolume gpu_binary_cost_z(dims, stk::Type_Float4);

        GpuBinaryFunction gpu_binary_fn;
        gpu_binary_fn.set_fixed_spacing(float3{1, 1, 1});
        gpu_binary_fn.set_regularization_weight(0.25f);
        gpu_binary_fn.set_regularization_scale(1.0f);
        gpu_binary_fn.set_regularization_exponent(2.0f);

        gpu_binary_fn(
            GpuDisplacementField(gpu_vectorfield, affine_transform),
            delta,
            int3{0, 0, 0},
            int3{(int)dims.x, (int)dims.y, (int)dims.z},
            update_rule,
            gpu_binary_cost_x,
            gpu_binary_cost_y,
            gpu_binary_cost_z,
            stk::cuda::Stream::null()
        );

        stk::VolumeFloat4 binary_cost_x = gpu_binary_cost_x.download();
        stk::VolumeFloat4 binary_cost_y = gpu_binary_cost_y.download();
        stk::VolumeFloat4 binary_cost_z = gpu_binary_cost_z.download();

        DisplacementField df(vectorfield, affine_transform);

        Regularizer binary_fn;
        binary_fn.set_fixed_spacing(float3{1, 1, 1});
        binary_fn.set_regularization_weight(0.25f);
        binary_fn.set_regularization_scale(1.0f);
        binary_fn.set_regularization_exponent(2.0f);

        for (int3 p : dims) {
            float3 d1 = df.get(p);
            float3 d1d = df.get(p, delta,
                                update_rule == Settings::UpdateRule_Compositive);

            if (p.x + 1 < int(dims.x)) {
                int3 step {1, 0, 0};
                float3 d2 = df.get(p+step);
                float3 d2d = df.get(p+step, delta, 
                                    update_rule == Settings::UpdateRule_Compositive);

                double f00 = binary_fn(p, d1, d2, step);
                double f01 = binary_fn(p, d1, d2d, step);
                double f10 = binary_fn(p, d1d, d2, step);
                double f11 = binary_fn(p, d1d, d2d, step);

                double gpu_f00 = binary_cost_x(p).x;
                double gpu_f01 = binary_cost_x(p).y;
                double gpu_f10 = binary_cost_x(p).z;
                double gpu_f11 = binary_cost_x(p).w;

                REQUIRE(gpu_f00 == Approx(f00));
                REQUIRE(gpu_f01 == Approx(f01));
                REQUIRE(gpu_f10 == Approx(f10));
                REQUIRE(gpu_f11 == Approx(f11));
            }
            
            if (p.y + 1 < int(dims.y)) {
                int3 step {0, 1, 0};
                float3 d2 = df.get(p+step);
                float3 d2d = df.get(p+step, delta,
                                    update_rule == Settings::UpdateRule_Compositive);

                double f00 = binary_fn(p, d1, d2, step);
                double f01 = binary_fn(p, d1, d2d, step);
                double f10 = binary_fn(p, d1d, d2, step);
                double f11 = binary_fn(p, d1d, d2d, step);

                double gpu_f00 = binary_cost_y(p).x;
                double gpu_f01 = binary_cost_y(p).y;
                double gpu_f10 = binary_cost_y(p).z;
                double gpu_f11 = binary_cost_y(p).w;

                REQUIRE(gpu_f00 == Approx(f00));
                REQUIRE(gpu_f01 == Approx(f01));
                REQUIRE(gpu_f10 == Approx(f10));
                REQUIRE(gpu_f11 == Approx(f11));
            }

            if (p.z + 1 < int(dims.z)) {
                int3 step {0, 0, 1};
                float3 d2 = df.get(p+step);
                float3 d2d = df.get(p+step, delta,
                                    update_rule == Settings::UpdateRule_Compositive);

                double f00 = binary_fn(p, d1, d2, step);
                double f01 = binary_fn(p, d1, d2d, step);
                double f10 = binary_fn(p, d1d, d2, step);
                double f11 = binary_fn(p, d1d, d2d, step);

                double gpu_f00 = binary_cost_z(p).x;
                double gpu_f01 = binary_cost_z(p).y;
                double gpu_f10 = binary_cost_z(p).z;
                double gpu_f11 = binary_cost_z(p).w;

                REQUIRE(gpu_f00 == Approx(f00));
                REQUIRE(gpu_f01 == Approx(f01));
                REQUIRE(gpu_f10 == Approx(f10));
                REQUIRE(gpu_f11 == Approx(f11));
            }
        }
    } // for update_rule
}

TEST_CASE("gpu_unary_term", "")
{
    float3 delta {0.5f, 0.0f, 0.0f};
    dim3 dims { 32, 32, 32 };
    stk::VolumeFloat3 vectorfield(dims, float3{0, 0, 0});
    stk::VolumeFloat fixed(dims, 0.0f);
    stk::VolumeFloat moving(dims, 0.0f);
    for (int3 p : dims) {
        vectorfield(p) = float3{
            (rand() % 100) / 10.0f,
            (rand() % 100) / 10.0f,
            (rand() % 100) / 10.0f
        };
        fixed(p) = (rand() % 100) / 100.0f;
        moving(p) = (rand() % 100) / 100.0f;
    }

    stk::GpuVolume gpu_vectorfield(volume_float3_to_float4(vectorfield));
    stk::GpuVolume gpu_fixed(fixed);
    stk::GpuVolume gpu_moving(moving);
    
    Matrix3x3f matrix {
        float3{2, 0, 0},
        float3{0, 3, 0},
        float3{0, 0, 4}
    };

    AffineTransform affine_transform(matrix, float3{5,-6,7});

    Settings::UpdateRule update_rules[] = {
        Settings::UpdateRule_Additive,
        Settings::UpdateRule_Compositive
    };
    for (auto update_rule : update_rules) {
        stk::VolumeFloat2 unary_cost(dims, float2{0, 0});
        stk::GpuVolume gpu_unary_cost(unary_cost);

        GpuUnaryFunction gpu_unary_fn;

        std::unique_ptr<GpuCostFunction> fn = make_unique<GpuCostFunction_NCC>(
            gpu_fixed, gpu_moving, 2);
        gpu_unary_fn.add_function(
            fn,
            1.0f
        );

        GpuDisplacementField gpu_df(gpu_vectorfield, affine_transform);
        gpu_unary_fn(
            gpu_df,
            delta,
            int3{0, 0, 0},
            int3{(int)dims.x, (int)dims.y, (int)dims.z},
            update_rule,
            gpu_unary_cost,
            stk::cuda::Stream::null()
        );

        unary_cost = gpu_unary_cost.download();

        DisplacementField df(vectorfield, affine_transform);

        UnaryFunction unary_fn;
        unary_fn.add_function(
            make_unique<NCCFunction_sphere<float>>(fixed, moving, 2),
            1.0f
        );
        
        for (int3 p : dims) {
            float3 d1 = df.get(p);
            float3 d1d = df.get(p, delta,
                                update_rule == Settings::UpdateRule_Compositive);

            double f0 = unary_fn(p, d1);
            double f1 = unary_fn(p, d1d);

            double gpu_f0 = unary_cost(p).x;
            double gpu_f1 = unary_cost(p).y;

            REQUIRE(gpu_f0 == Approx(f0));
            REQUIRE(gpu_f1 == Approx(f1));
        }
        stk::VolumeFloat4 gpu_vectorfield_copy = gpu_vectorfield.download();
        stk::VolumeFloat gpu_fixed_copy = gpu_fixed.download();
        stk::VolumeFloat gpu_moving_copy = gpu_moving.download();
        for (int3 p : dims) {
            REQUIRE(gpu_vectorfield_copy(p).x == vectorfield(p).x);
            REQUIRE(gpu_vectorfield_copy(p).y == vectorfield(p).y);
            REQUIRE(gpu_vectorfield_copy(p).z == vectorfield(p).z);
            REQUIRE(gpu_fixed_copy(p) == fixed(p));
            REQUIRE(gpu_moving_copy(p) == moving(p));
        }
    } // for update_rule

}

#endif // DF_USE_CUDA
