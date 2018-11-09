#pragma once

#include "optimiser.h"

#if defined(__GNUC__) || defined(__clang__)
    #define GCC_VERSION (__GNUC__ * 10000 \
        + __GNUC_MINOR__ * 100 \
        + __GNUC_PATCHLEVEL__)

    #if GCC_VERSION > 40604 // 4.6.4
        #pragma GCC diagnostic push
    #endif

    #pragma GCC diagnostic ignored "-Wpedantic"
    #pragma GCC diagnostic ignored "-Wreorder"

    #include <QPBO.h>

    #if GCC_VERSION > 40604 // 4.6.4
        #pragma GCC diagnostic pop
    #endif
#endif

template<typename T>
class Qpbo : public Optimiser<T>
{
public:
    Qpbo(const int3& size);
    virtual ~Qpbo();

    virtual void add_term1(const int3& p, T e0, T e1);
    virtual void add_term1(int x, int y, int z, T e0, T e1);

    virtual void add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11);
    virtual void add_term2(int x1, int y1, int z1,
                   int x2, int y2, int z2,
                   T e00, T e01, T e10, T e11);

    virtual T minimize();

    virtual int get_var(const int3& p);
    virtual int get_var(int x, int y, int z);

private:
    QPBO<T> _q;
};

#include "qpbo.inl"

