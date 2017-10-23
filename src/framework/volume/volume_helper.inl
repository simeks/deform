
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
T VolumeHelper<T>::linear_at(float x, float y, float z, volume::BorderMode border_mode) const
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

    return T((1 - zt)*((1 - yt)*((1 - xt)*operator()(x1, y1, z1) +
        (xt)*operator()(x2, y1, z1)) +
        (yt)*((1 - xt)*operator()(x1, y2, z1) +
        (xt)*operator()(x2, y2, z1))) +
        (zt)*((1 - yt)*((1 - xt)*operator()(x1, y1, z2) +
        (xt)*operator()(x2, y1, z2)) +
        (yt)*((1 - xt)*operator()(x1, y2, z2) +
        (xt)*operator()(x2, y2, z2))));

    #undef FAST_CEIL
    #undef FAST_FLOOR
}
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

