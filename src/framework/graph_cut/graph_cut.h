#pragma once

/// Interface for using the GCO graph cut solver.

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wdeprecated-register"
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
#pragma GCC diagnostic pop
#else
#pragma warning(pop)
#endif

template<typename T>
class GraphCut
{
public:
    GraphCut(const Vec3i& size);
    ~GraphCut();

    void add_term1(const Vec3i& p, T e0, T e1);
    void add_term1(int x, int y, int z, T e0, T e1);

    void add_term2(const Vec3i& p1, const Vec3i& p2, T e00, T e01, T e10, T e11);
    void add_term2(int x1, int y1, int z1,
                   int x2, int y2, int z2,
                   T e00, T e01, T e10, T e11);

    T minimize();

    int get_var(const Vec3i& p);
    int get_var(int x, int y, int z);

private:
    int get_index(int x, int y, int z) const;
    int get_index(const Vec3i& p) const;

    gco::Energy<T, T, T> _e;
    Vec3i _size;
};

#include "graph_cut.inl"
