#pragma once

#include "solver.h"

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

// Include all headers used by gco here, before including gco.
// Otherwise, the symbols here defined will end up inside the gco
// namespace.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace gco
{
    // Prevent breaking the build in C++17, where register was removed.
    // The keyword is used within GCO, no idea why, since it is a few
    // decades that it is "exactly as meaningful as whitespace" (cit).
    #define register

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
class GraphCut : public Solver<T>
{
public:
    GraphCut(const int3& size);
    virtual ~GraphCut();

    virtual void add_term1(const int3& p, T e0, T e1);
    virtual void add_term1(const int x, const int y, const int z, T e0, T e1);

    virtual void add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11);
    virtual void add_term2(const int x1, const int y1, const int z1,
                           const int x2, const int y2, const int z2,
                           T e00, T e01, T e10, T e11);

	virtual void add_term3(const int3& p1, const int3& p2, const int3& p3,
	                       T e000, T e001,
	                       T e010, T e011,
	                       T e100, T e101,
	                       T e110, T e111);
	virtual void add_term3(const int x1, const int y1, const int z1,
                           const int x2, const int y2, const int z2,
                           const int x3, const int y3, const int z3,
	                       T e000, T e001,
	                       T e010, T e011,
	                       T e100, T e101,
	                       T e110, T e111);

    virtual T minimize();

    virtual int get_var(const int3& p);
    virtual int get_var(int x, int y, int z);

private:
    gco::Energy<T, T, T> _e;
};

#include "graph_cut.inl"
