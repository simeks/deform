#pragma once

#include "qpbo.h"

template<typename T>
Qpbo<T>::Qpbo(const int3& size)
    : Optimiser<T>(size)
    , _q(size.x * size.y * size.z, size.x * size.y * size.z * 3)
{
    _q.AddNode(size.x * size.y * size.z);
}

template<typename T>
Qpbo<T>::~Qpbo()
{
}

template<typename T>
void Qpbo<T>::add_term1(const int3& p, T e0, T e1)
{
    _q.AddUnaryTerm(Optimiser<T>::get_index(p), e0, e1);
}
template<typename T>
void Qpbo<T>::add_term1(int x, int y, int z, T e0, T e1)
{
    _q.AddUnaryTerm(Optimiser<T>::get_index(x, y, z), e0, e1);
}

template<typename T>
void Qpbo<T>::add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11)
{
    int i1 = Optimiser<T>::get_index(p1);
    int i2 = Optimiser<T>::get_index(p2);
    _q.AddPairwiseTerm(i1, i2, e00, e01, e10, e11);
}
template<typename T>
void Qpbo<T>::add_term2(
    int x1, int y1, int z1,
    int x2, int y2, int z2,
    T e00, T e01, T e10, T e11)
{
    int i1 = Optimiser<T>::get_index(x1, y1, z1);
    int i2 = Optimiser<T>::get_index(x2, y2, z2);
    _q.AddPairwiseTerm(i1, i2, e00, e01, e10, e11);
}

template<typename T>
T Qpbo<T>::minimize()
{
    _q.Solve();
    _q.ComputeWeakPersistencies();
    return _q.ComputeTwiceEnergy() / 2.0;
}
template<typename T>
int Qpbo<T>::get_var(const int3& p)
{
    int index = Optimiser<T>::get_index(p.x, p.y, p.z);
    return _q.GetLabel(index);
}
template<typename T>
int Qpbo<T>::get_var(int x, int y, int z)
{
    int index = Optimiser<T>::get_index(x, y, z);
    return _q.GetLabel(index);
}

