#pragma once

/// Interface for using the GCO graph cut solver.

#if defined(__GNUC__) || defined(__clang__)
    #define GCC_VERSION (__GNUC__ * 10000 \
        + __GNUC_MINOR__ * 100 \
        + __GNUC_PATCHLEVEL__)

    // Older versions of GCC do not support push/pop
    #if GCC_VERSION > 40604 // 4.6.4
        #pragma GCC diagnostic push
    #endif

    #pragma GCC diagnostic ignored "-Wparentheses"
    #pragma GCC diagnostic ignored "-Woverflow"

    #if defined(__clang__)
        #pragma clang diagnostic ignored "-Wkeyword-macro"
    #endif
#else
    #pragma warning(push)
    #pragma warning(disable: 4512)
    #pragma warning(disable: 4100)
    #pragma warning(disable: 4189)
    #pragma warning(disable: 4701)
    #pragma warning(disable: 4706)
    #pragma warning(disable: 4463)
#endif

// Prevent breaking the build in C++17, where register was removed.
// The keyword is used within GCO, no idea why, since it is a few
// decades that it is "exactly as meaningful as whitespace" (cit).
#define register

#include <gco/energy.h>
#include <gco/graph.cpp>
#include <gco/maxflow.cpp>

#if defined(__GNUC__) || defined(__clang__)
    #if GCC_VERSION > 40604 // 4.6.4
        #pragma GCC diagnostic pop
    #endif
#else
    #pragma warning(pop)
#endif

template<typename T>
class GCOSolver
{
public:
    typedef T FlowType;

    GCOSolver(const int3& size);
    ~GCOSolver();

    void add_term1(int x, int y, int z, T e0, T e1);

    void add_term2(int x1, int y1, int z1,
                   int x2, int y2, int z2,
                   T e00, T e01, T e10, T e11);

    T minimize();

    int get_var(int x, int y, int z);

private:
    int get_index(int x, int y, int z) const;

    Energy<T, T, T> _e;
    int3 _size;
};

#include "gco_solver.inl"
