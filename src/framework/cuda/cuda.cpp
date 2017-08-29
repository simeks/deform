#include "cuda.h"

#ifdef DF_ENABLE_CUDA

#include <cuda_runtime.h>
#include <stdio.h>

#include "helper_cuda.h"

void cuda::init(int device_id)
{
    cudaSetDevice(device_id);
    
    checkCudaErrors(cudaGetDevice(&device_id));

    cudaDeviceProp device_prop;
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, device_id));

    printf("Device %d - name: %s\n", device_id, device_prop.name);
}
#endif 