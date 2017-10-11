#pragma once

/// Interface for using the GCO graph cut solver.

#include <framework/platform/gcc.h>

#if defined(__GNUC__) || defined(__clang__)

    // Older versions of GCC do not support push/pop
    #if GCC_VERSION > 40604 // 4.6.4
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wdeprecated-register"
    #endif

    #pragma GCC diagnostic ignored "-Wparentheses"
#else
    #pragma warning(push)
    #pragma warning(disable: 4512)
    #pragma warning(disable: 4100)
    #pragma warning(disable: 4189)
    #pragma warning(disable: 4701)
    #pragma warning(disable: 4706)
    #pragma warning(disable: 4463)
#endif
namespace gco
{
    #include <gco/energy.h>
    #include <gco/graph.cpp>
    #include <gco/maxflow.cpp>
}
#if defined(__GNUC__) || defined(__clang__)
    #if GCC_VERSION > 40604 // 4.6.4
        #pragma GCC diagnostic pop
    #endif
#else
    #pragma warning(pop)
#endif

template<typename T>
class GraphCut
{
public:
    GraphCut(const int3& size);
    ~GraphCut();

    void add_term1(const int3& p, T e0, T e1);
    void add_term1(int x, int y, int z, T e0, T e1);

    void add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11);
    void add_term2(int x1, int y1, int z1,
                   int x2, int y2, int z2,
                   T e00, T e01, T e10, T e11);

    T minimize();

    int get_var(const int3& p);
    int get_var(int x, int y, int z);

private:
    int get_index(int x, int y, int z) const;
    int get_index(const int3& p) const;

    gco::Energy<T, T, T> _e;
    int3 _size;
};

#include "graph_cut.inl"
