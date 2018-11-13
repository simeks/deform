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
    #pragma GCC diagnostic ignored "-Wsign-compare"

    #include <ELC/ELC.h>
    #include <QPBO.h>

    #if GCC_VERSION > 40604 // 4.6.4
        #pragma GCC diagnostic pop
    #endif
#endif

namespace elc = ELCReduce;

enum class ELCReductionMode {
    HOCR,
    ELC_HOCR,
    Approximate,
};

template<typename T, ELCReductionMode mode>
class ELC : public Solver<T>
{
public:
    ELC(const int3& size);
    virtual ~ELC();

    virtual void add_term1(const int3& p, T e0, T e1);
    virtual void add_term1(const int x, const int y, const int z, T e0, T e1);

    virtual void add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11);
    virtual void add_term2(const int x1, const int y1, const int z1,
                           const int x2, const int y2, const int z2,
                           T e00, T e01, T e10, T e11);

	virtual void add_term_n(const std::vector<int3>& p, const std::vector<T> e);

    virtual T minimize();

    virtual int get_var(const int3& p);
    virtual int get_var(const int x, const int y, const int z);

private:
    void convert();

    const int3 _size;
    qpbo::QPBO<T> _q;
    elc::PBF<T> _pbf;
};

#include "elc.inl"
