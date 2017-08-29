#pragma once

#ifdef DF_ENABLE_CUDA
namespace cuda
{
    void init(int device_id);
}
#endif
