#pragma once

#include <stk/math/math.h>

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
class QPBO
{
public:
    QPBO(const int3& size);
    ~QPBO();

    void add_term1(const int3& p, T e0, T e1);
    void add_term1(const int x, const int y, const int z, T e0, T e1);

    void add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11);
    void add_term2(const int x1, const int y1, const int z1,
                   const int x2, const int y2, const int z2,
                   T e00, T e01, T e10, T e11);

    template<int N>
    void add_term(const int3 p[N], const T e[1 << N]);

    T minimize();

    int get_var(const int3& p);
    int get_var(const int x, const int y, const int z);

private:
    inline int get_index(const int x, const int y, const int z) const {
        return x + y*_size.x + z*_size.x*_size.y;
    }

    inline int get_index(const int3& p) const {
        return p.x + p.y*_size.x + p.z*_size.x*_size.y;
    }

    const int3 _size;
    qpbo::QPBO<T> _q;
};

#include "qpbo.inl"

