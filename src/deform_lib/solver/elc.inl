#pragma once

#include "elc.h"

template<typename T, ELCReductionMode mode>
ELC<T, mode>::ELC(const int3& size)
    : _size(size)
    , _q(size.x * size.y * size.z, size.x * size.y * size.z * 3)
    , _pbf(size.x * size.y * size.z)
{
    _q.AddNode(size.x * size.y * size.z);
}

template<typename T, ELCReductionMode mode>
ELC<T, mode>::~ELC()
{
}

template<typename T, ELCReductionMode mode>
void ELC<T, mode>::add_term1(const int3& p, T e0, T e1)
{
    _pbf.AddUnaryTerm(Solver<T>::get_index(p), e0, e1);
}
template<typename T, ELCReductionMode mode>
void ELC<T, mode>::add_term1(const int x, const int y, const int z, T e0, T e1)
{
    _pbf.AddUnaryTerm(Solver<T>::get_index(x, y, z), e0, e1);
}

template<typename T, ELCReductionMode mode>
void ELC<T, mode>::add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11)
{
    int i1 = Solver<T>::get_index(p1);
    int i2 = Solver<T>::get_index(p2);
    _pbf.AddPairwiseTerm(i1, i2, e00, e01, e10, e11);
}
template<typename T, ELCReductionMode mode>
void ELC<T, mode>::add_term2(
    const int x1, const int y1, const int z1,
    const int x2, const int y2, const int z2,
    T e00, T e01, T e10, T e11)
{
    int i1 = Solver<T>::get_index(x1, y1, z1);
    int i2 = Solver<T>::get_index(x2, y2, z2);
    _pbf.AddPairwiseTerm(i1, i2, e00, e01, e10, e11);
}

template<typename T, ELCReductionMode mode>
void ELC<T, mode>::add_term_n(const std::vector<int3>& p, const std::vector<T> e)
{
    ASSERT((1 << p.size()) == e.size() && "Mismatching number of variables and energy terms");
    std::vector<int> i;
    std::transform(p.begin(), p.end(), back_inserter(i), [this](auto& p) { return Solver<T>::get_index(p); });
    _pbf.AddHigherTerm(p.size(), i.data(), const_cast<T*>(e.data()));
}

template<typename T, ELCReductionMode mode>
T ELC<T, mode>::minimize()
{
    convert();
    _q.MergeParallelEdges();
    _q.Solve();
    _q.ComputeWeakPersistencies();
    return _q.ComputeTwiceEnergy() / 2.0;
}

template<typename T, ELCReductionMode mode>
int ELC<T, mode>::get_var(const int3& p)
{
    int index = Solver<T>::get_index(p.x, p.y, p.z);
    return _q.GetLabel(index);
}
template<typename T, ELCReductionMode mode>
int ELC<T, mode>::get_var(const int x, const int y, const int z)
{
    int index = Solver<T>::get_index(x, y, z);
    return _q.GetLabel(index);
}

template<typename T, ELCReductionMode mode>
void ELC<T, mode>::convert()
{
    elc::PBF<T> quadratic_pbf;

    // Convert to quadratic PBF
    if constexpr (mode == ELCReductionMode::ELC_HOCR) {
        _pbf.reduceHigher();
        _pbf.toQuadratic(quadratic_pbf, _pbf.maxID() + 1);
    }
    else if constexpr (mode == ELCReductionMode::HOCR) {
        _pbf.toQuadratic(quadratic_pbf, _pbf.maxID() + 1);
    }
    else if constexpr (mode == ELCReductionMode::Approximate) {
        quadratic_pbf = _pbf;
        quadratic_pbf.reduceHigherApprox();
        ASSERT(false && "This line should be unreachable");
    }

    // Convert to QPBO object
    quadratic_pbf.convert(_q, quadratic_pbf.maxID() + 1);
}

