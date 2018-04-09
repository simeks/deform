template<typename T>
VolumeHelper<T>::VolumeHelper()
{
}
template<typename T>
VolumeHelper<T>::VolumeHelper(const Volume& other) : Volume(other)
{
    assert(!Volume::valid() || ::voxel_type<T>::type_id == other.voxel_type());
}
template<typename T>
VolumeHelper<T>::VolumeHelper(const Dims& size) : Volume(size, ::voxel_type<T>::type_id)
{
}
template<typename T>
VolumeHelper<T>::VolumeHelper(const Dims& size, const T& value) : Volume(size, ::voxel_type<T>::type_id)
{
    fill(value);
}
template<typename T>
VolumeHelper<T>::~VolumeHelper()
{
}
template<typename T>
void VolumeHelper<T>::fill(const T& value)
{
    for (uint32_t z = 0; z < _size.depth; ++z)
    {
        for (uint32_t y = 0; y < _size.height; ++y)
        {
            T* begin = (T*)(((uint8_t*)_ptr) + (z * _stride * _size.height + y * _stride));
            T* end = begin + _size.width;
            std::fill(begin, end, value);
        }
    }
}
template<typename T>
T VolumeHelper<T>::at(int x, int y, int z, volume::BorderMode border_mode) const
{
    // PERF: 15% faster than ceilf/floorf
    #define FAST_CEIL(x_) ((int)x_ + (x_ > (int)x_))
    #define FAST_FLOOR(x_) ((int)x_ - (x_ < (int)x_))

    if (border_mode == volume::Border_Constant) 
    {
        if (x < 0 || FAST_CEIL(x) >= int(_size.width) ||
            y < 0 || FAST_CEIL(y) >= int(_size.height) ||
            z < 0 || FAST_CEIL(z) >= int(_size.depth)) 
        {
            return T{0};
        }
    }
    else if (border_mode == volume::Border_Replicate)
    {
        x = std::max(x, 0);
        x = std::min(x, int(_size.width - 1));
        y = std::max(y, 0);
        y = std::min(y, int(_size.height - 1));
        z = std::max(z, 0);
        z = std::min(z, int(_size.depth - 1));
    }

    return *((T const*)(((uint8_t*)_ptr) + offset(x, y, z)));

    #undef FAST_CEIL
    #undef FAST_FLOOR
}
template<typename T>
T VolumeHelper<T>::at(int3 p, volume::BorderMode border_mode) const
{
    return at(p.x, p.y, p.z, border_mode);
}
template<typename T>
inline T VolumeHelper<T>::linear_at(float x, float y, float z, volume::BorderMode border_mode) const
{
    // PERF: 15% faster than ceilf/floorf
    #define FAST_CEIL(x_) ((int)x_ + (x_ > (int)x_))

    // We expect all indices to be positive therefore a regular cast should suffice
    #define FAST_FLOOR(x_) int(x_)

    if (border_mode == volume::Border_Constant) 
    {
        if (x < 0 || FAST_CEIL(x) >= int(_size.width) ||
            y < 0 || FAST_CEIL(y) >= int(_size.height) ||
            z < 0 || FAST_CEIL(z) >= int(_size.depth)) 
        {
            return T{0};
        }
    }
    else if (border_mode == volume::Border_Replicate)
    {
        x = std::max(x, 0.0f);
        x = std::min(x, float(_size.width - 1));
        y = std::max(y, 0.0f);
        y = std::min(y, float(_size.height - 1));
        z = std::max(z, 0.0f);
        z = std::min(z, float(_size.depth - 1));
    }

    float xt = x - FAST_FLOOR(x);
    float yt = y - FAST_FLOOR(y);
    float zt = z - FAST_FLOOR(z);

    int x1 = int(FAST_FLOOR(x));
    int x2 = int(FAST_CEIL(x));
    int y1 = int(FAST_FLOOR(y));
    int y2 = int(FAST_CEIL(y));
    int z1 = int(FAST_FLOOR(z));
    int z2 = int(FAST_CEIL(z));

    return T(
        (1 - zt) *
            (
                (1 - yt) *
                (
                    (1 - xt) * operator()(x1, y1, z1) + // s1
                    (xt) * operator()(x2, y1, z1) // s2
                ) +

                (yt) * 
                (
                    (1 - xt) * operator()(x1, y2, z1) + // s3
                    (xt) * operator()(x2, y2, z1) // s4
                )
            ) +
        (zt) * 
            (
                (1 - yt)*
                (
                    (1 - xt)*operator()(x1, y1, z2) + // s5
                    (xt)*operator()(x2, y1, z2) // s6
                ) +
                
                (yt)*
                (
                    (1 - xt)*operator()(x1, y2, z2) + // s7
                    (xt)*operator()(x2, y2, z2) // s8
                )
            )
        );

    #undef FAST_CEIL
    #undef FAST_FLOOR
}

#ifdef DF_ENABLE_SSE_LINEAR_AT
template<>
inline float VolumeHelper<float>::linear_at(float x, float y, float z, volume::BorderMode border_mode) const
{
    // An attempt to speed-up linear_at which takes up a majority of the time when using NCC.
    // However, does not seem to make any difference for MSVC2017 as the compiler performs these optimizations

    // PERF: 15% faster than ceilf/floorf
    #define FAST_CEIL(x_) ((int)x_ + (x_ > (int)x_))
    #define FAST_FLOOR(x_) int(x_)

    if (border_mode == volume::Border_Constant) 
    {
        if (x < 0 || FAST_CEIL(x) >= int(_size.width) ||
            y < 0 || FAST_CEIL(y) >= int(_size.height) ||
            z < 0 || FAST_CEIL(z) >= int(_size.depth)) 
        {
            return 0;
        }
    }
    else if (border_mode == volume::Border_Replicate)
    {
        x = std::max(x, 0.0f);
        x = std::min(x, float(_size.width - 1));
        y = std::max(y, 0.0f);
        y = std::min(y, float(_size.height - 1));
        z = std::max(z, 0.0f);
        z = std::min(z, float(_size.depth - 1));
    }

    __m128 p = _mm_set_ps(1.0f, z, y, x);
    __m128 fp = _mm_floor_ps(p);
    __m128 t = _mm_sub_ps(p, fp);

    int x1 = int((x));
    int x2 = int(FAST_CEIL(x));
    int y1 = int((y));
    int y2 = int(FAST_CEIL(y));
    int z1 = int((z));
    int z2 = int(FAST_CEIL(z));

    __m128 ones = _mm_set1_ps(1.0f);
    __m128 it = _mm_sub_ps(ones, t);

    // Samples
    __m128 sa = _mm_set_ps(
        operator()(x2, y2, z1),  // s4
        operator()(x1, y2, z1), // s3
        operator()(x2, y1, z1), // s2
        operator()(x1, y1, z1) // s1
    );

    __m128 sb = _mm_set_ps(
        operator()(x2, y2, z2),  // s8
        operator()(x1, y2, z2), // s7
        operator()(x2, y1, z2), // s6
        operator()(x1, y1, z2) // s5
    );

    // Sum of
    //  iz * iy * ix * s1
    //  iz * iy *  x * s2
    //  iz *  y * ix * s3
    //  iz *  y *  x * s4
    //   z * iy * ix * s5
    //   z * iy *  x * s6
    //   z *  y * ix * s7
    //   z *  y *  x * s8

    //  iz * s1
    //  iz * s2
    //  iz * s3
    //  iz * s4
    //   z * s5
    //   z * s6
    //   z * s7
    //   z * s8

    __m128 za = _mm_shuffle_ps(it, it, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 zb = _mm_shuffle_ps(t, t, _MM_SHUFFLE(2, 2, 2, 2));

    __m128 z_sa = _mm_mul_ps(za, sa);
    __m128 z_sb = _mm_mul_ps(zb, sb);

    // iy * (iz * s1)
    // iy * (iz * s2)
    //  y * (iz * s3)
    //  y * (iz * s4)
    // iy * ( z * s5)
    // iy * ( z * s6)
    //  y * ( z * s7)
    //  y * ( z * s8)

    __m128 ya = _mm_shuffle_ps(it, t, _MM_SHUFFLE(1,1,1,1));
    __m128 y_z_sa = _mm_mul_ps(ya, z_sa);
    __m128 y_z_sb = _mm_mul_ps(ya, z_sb);

    // (s1, s2, s3, s4) => (s1, s3, s5, s7)
    // (s5, s6, s7, s8) => (s2, s4, s6, s8)
    __m128 y_z_sc = _mm_shuffle_ps(y_z_sa, y_z_sb, _MM_SHUFFLE(0, 2, 0, 2));
    __m128 y_z_sd = _mm_shuffle_ps(y_z_sa, y_z_sb, _MM_SHUFFLE(1, 3, 1, 3));

    // ix * (iy * (iz * s1))
    // ix * ( y * (iz * s3))
    // ix * (iy * ( z * s5))
    // ix * ( y * ( z * s7))

    //  x * (iy * (iz * s2))
    //  x * ( y * (iz * s4))
    //  x * (iy * ( z * s6))
    //  x * ( y * ( z * s8))

    __m128 xa = _mm_shuffle_ps(it, it, _MM_SHUFFLE(0,0,0,0));
    __m128 xb = _mm_shuffle_ps(t, t, _MM_SHUFFLE(0,0,0,0));

    __m128 x_y_z_sc = _mm_mul_ps(xa, y_z_sc);
    __m128 x_y_z_sd = _mm_mul_ps(xb, y_z_sd);

    __m128 sum_1 = _mm_dp_ps(x_y_z_sc, ones, 0xff);
    __m128 sum_2 = _mm_dp_ps(x_y_z_sd, ones, 0xff);

    __m128 sum = _mm_add_ps(sum_1, sum_2);

    alignas(16) float sum_f = 0.0f;
    _mm_store_ps1(&sum_f, sum);

    return sum_f;

    #undef FAST_CEIL
    #undef FAST_FLOOR
}
#endif // DF_ENABLE_SSE_LINEAR_AT

template<typename T>
T VolumeHelper<T>::linear_at(float3 p, volume::BorderMode border_mode) const
{
    return linear_at(p.x, p.y, p.z, border_mode);
}
template<typename T>
VolumeHelper<T>& VolumeHelper<T>::operator=(VolumeHelper& other)
{
    assert(::voxel_type<T>::type_id == other.voxel_type());
    Volume::operator=(other);
    return *this;
}
template<typename T>
VolumeHelper<T>& VolumeHelper<T>::operator=(Volume& other)
{
    assert(::voxel_type<T>::type_id == other.voxel_type());
    Volume::operator=(other);
    return *this;
}

template<typename T>
const T& VolumeHelper<T>::operator()(int x, int y, int z) const
{
    return *((T const*)(((uint8_t*)_ptr) + offset(x, y, z)));
}
template<typename T>
T& VolumeHelper<T>::operator()(int x, int y, int z)
{
    return *((T*)(((uint8_t*)_ptr) + offset(x, y, z)));
}
template<typename T>
const T& VolumeHelper<T>::operator()(const int3& p) const
{
    return operator()(p.x, p.y, p.z);
}
template<typename T>
T& VolumeHelper<T>::operator()(const int3& p)
{
    return operator()(p.x, p.y, p.z);
}
template<typename T>
inline size_t VolumeHelper<T>::offset(int x, int y, int z) const
{
    assert(x < int(_size.width));
    assert(y < int(_size.height));
    assert(z < int(_size.depth));
    return z * _stride * _size.height + y * _stride + x * sizeof(T);
}
template<typename T>
void VolumeHelper<T>::min_max(T& min, T& max) const
{
    min = FLT_MAX;
    max = -FLT_MAX;

    for (uint32_t z = 0; z < _size.depth; ++z)
    {
        for (uint32_t y = 0; y < _size.height; ++y)
        {
            for (uint32_t x = 0; x < _size.width; ++x)
            {
                min = std::min<T>(min, (*this)(x, y, z));
                max = std::max<T>(max, (*this)(x, y, z));
            }
        }
    }
}

