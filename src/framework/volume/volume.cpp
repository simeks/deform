#ifdef DF_ENABLE_CUDA
    #include <cuda_runtime.h>

    #include "gpu_volume.h"
    #include "helper_cuda.h"
#endif

#include "volume.h"
#include "voxel.h"

#include <assert.h>
#include <iostream>


namespace
{
    typedef void(*ConverterFn)(void*, void*, size_t num);

    template<typename TSrc, typename TDest>
    void convert_voxels(void* src, void* dest, size_t num)
    {
        // TODO:
        // Very rough conversion between voxel formats,
        // works fine for double <-> float, should suffice for now

        for (size_t i = 0; i < num; ++i)
        {
            ((TDest*)dest)[i] = TDest(((TSrc*)src)[i]);
        }
    }
}

VolumeData::VolumeData() : data(nullptr), size(0)
{
}
VolumeData::VolumeData(size_t size) : size(size)
{
    data = (uint8_t*)malloc(size);
}
VolumeData::~VolumeData()
{
    if (data)
        free(data);
}

Volume::Volume() : _ptr(nullptr), _stride(0)
{
    _origin = {0, 0, 0};
    _spacing = {1, 1, 1};
}
Volume::Volume(const Dims& size, uint8_t voxel_type, uint8_t* data) :
    _size(size),
    _voxel_type(voxel_type)
{
    _origin = {0, 0, 0};
    _spacing = {1, 1, 1};

    allocate(size, voxel_type);
    if (data)
    {
        size_t num_bytes = _size.width * _size.height *
            _size.depth * voxel::size(_voxel_type);

        memcpy(_ptr, data, num_bytes);
    }
}
Volume::~Volume()
{
}
Volume Volume::clone() const
{
    Volume copy(_size, _voxel_type);
    copy._origin = _origin;
    copy._spacing = _spacing;

    size_t num_bytes = _size.width * _size.height * 
        _size.depth * voxel::size(_voxel_type);
    
    memcpy(copy._ptr, _ptr, num_bytes);

    return copy;
}
Volume Volume::as_type(uint8_t type) const
{
    if (_voxel_type == type)
        return *this;

    assert(voxel::num_components(type) == voxel::num_components(_voxel_type));

    Volume dest(_size, type);
    
    uint8_t src_type = voxel::base_type(_voxel_type);
    uint8_t dest_type = voxel::base_type(type);

    size_t num = _size.width * _size.height * _size.depth * voxel::num_components(type);
    if (src_type == voxel::Type_Float && dest_type == voxel::Type_Double)
        convert_voxels<float, double>(_ptr, dest._ptr, num);
    if (src_type == voxel::Type_Double && dest_type == voxel::Type_Float)
        convert_voxels<double, float>(_ptr, dest._ptr, num);
    else
        assert(false);

    return dest;
}
bool Volume::valid() const
{
    return _ptr != nullptr;
}
void* Volume::ptr()
{
    assert(_ptr);
    assert(_data->data);
    assert(_data->size);
    return _ptr;
}
void const* Volume::ptr() const
{
    assert(_ptr);
    assert(_data->data);
    assert(_data->size);
    return _ptr;
}
uint8_t Volume::voxel_type() const
{
    return _voxel_type;
}
const Dims& Volume::size() const
{
    return _size;
}
void Volume::set_origin(const float3& origin)
{
    _origin = origin;
}
void Volume::set_spacing(const float3& spacing)
{
    _spacing = spacing;
}
const float3& Volume::origin() const
{
    return _origin;
}
const float3& Volume::spacing() const
{
    return _spacing;
}

Volume::Volume(const Volume& other) :
    _data(other._data),
    _ptr(other._ptr),
    _stride(other._stride),
    _size(other._size),
    _voxel_type(other._voxel_type),
    _origin(other._origin),
    _spacing(other._spacing)
{
}
Volume& Volume::operator=(const Volume& other)
{
    _data = other._data;
    _ptr = other._ptr;
    _stride = other._stride;
    _size = other._size;
    _voxel_type = other._voxel_type;
    _origin = other._origin;
    _spacing = other._spacing;

    return *this;
}
void Volume::allocate(const Dims& size, uint8_t voxel_type)
{
    assert(voxel_type != voxel::Type_Unknown);

    _size = size;
    _voxel_type = voxel_type;
    _origin = { 0, 0, 0 };
    _spacing = { 1, 1, 1 };

    size_t num_bytes = _size.width * _size.height *
        _size.depth * voxel::size(_voxel_type);

    _data = std::make_shared<VolumeData>(num_bytes);
    _ptr = _data->data;
    _stride = voxel::size(_voxel_type) * _size.width;
}
void Volume::release()
{
    _data = nullptr;
    _ptr = nullptr;
    _size = { 0, 0, 0 };
    _stride = 0;
    _origin = { 0, 0, 0 };
    _spacing = { 1, 1, 1 };
}

#ifdef DF_ENABLE_CUDA
Volume::Volume(const GpuVolume& gpu_volume)
{
    allocate(gpu_volume.size, gpu::voxel_type(gpu_volume));
    download(gpu_volume);
}
GpuVolume Volume::upload() const
{
    GpuVolume vol = gpu::allocate_volume(_voxel_type, _size);
    upload(vol);
    return vol;
}
void Volume::upload(const GpuVolume& gpu_volume) const
{
    assert(gpu_volume.ptr != nullptr); // Requires gpu memory to be allocated
    assert(valid()); // Requires cpu memory to be allocated as well

    // We also assume both volumes have same dimensions
    assert( gpu_volume.size.width == _size.width &&
            gpu_volume.size.height == _size.height &&
            gpu_volume.size.depth == _size.depth);

    // TODO: Validate format?

    cudaMemcpy3DParms params = { 0 };
    params.srcPtr = make_cudaPitchedPtr(_ptr, _size.width * voxel::size(_voxel_type), _size.width, _size.height);
    params.dstArray = gpu_volume.ptr;
    params.extent = { gpu_volume.size.width, gpu_volume.size.height, gpu_volume.size.depth };
    params.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&params));
}
void Volume::download(const GpuVolume& gpu_volume)
{
    assert(gpu_volume.ptr != nullptr); // Requires gpu memory to be allocated
    assert(valid()); // Requires cpu memory to be allocated as well

    // We also assume both volumes have same dimensions
    assert( gpu_volume.size.width == _size.width &&
            gpu_volume.size.height == _size.height &&
            gpu_volume.size.depth == _size.depth);

    // TODO: Validate format?

    cudaMemcpy3DParms params = { 0 };
    params.srcArray = gpu_volume.ptr;
    params.dstPtr = make_cudaPitchedPtr(_ptr, _size.width * voxel::size(_voxel_type), _size.width, _size.height);
    params.extent = { gpu_volume.size.width, gpu_volume.size.height, gpu_volume.size.depth };
    params.kind = cudaMemcpyDeviceToHost;
    checkCudaErrors(cudaMemcpy3D(&params));
}

#endif // DF_ENABLE_CUDA

