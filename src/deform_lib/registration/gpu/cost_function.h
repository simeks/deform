#pragma once

namespace stk {
    class GpuVolume;
}

namespace gpu {
    void run_ssd_kernel(
        const stk::GpuVolume& fixed,
        const stk::GpuVolume& moving,
        const stk::GpuVolume& df,
        stk::GpuVolume& cost_acc
    );
}
