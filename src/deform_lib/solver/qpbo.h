#pragma once

#include "solver.h"

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
class QPBO : public Solver<T>
{
public:
    QPBO(const int3& size);
    virtual ~QPBO();

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
    virtual int get_var(const int x, const int y, const int z);

private:
    qpbo::QPBO<T> _q;
};

#include "qpbo.inl"

