template<typename T>
VolumeHelper<T>::VolumeHelper(Volume& other) : Volume(other)
{
    assert(!Volume::valid() || ::voxel_type<T>::type_id == other.voxel_type());
}
template<typename T>
VolumeHelper<T>::VolumeHelper(const Dims& size) : Volume(size, ::voxel_type<T>::type_id)
{
}
template<typename T>
VolumeHelper<T>::VolumeHelper(const Dims& size, const T& value) : VolumeHelper(size)
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
VolumeHelper<T>& VolumeHelper<T>::operator=(VolumeHelper& other)
{
    assert(voxel_type<T>::type_id == other.voxel_type());
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
inline size_t VolumeHelper<T>::offset(int x, int y, int z) const
{
    return z * _stride * _size.height + y * _stride + x * sizeof(T);
}
