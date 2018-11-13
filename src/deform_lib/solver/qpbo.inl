#pragma once

template<typename T>
QPBO<T>::QPBO(const int3& size)
    : _size(size)
    , _q(size.x * size.y * size.z, size.x * size.y * size.z * 3)
{
    _q.AddNode(size.x * size.y * size.z);
}

template<typename T>
QPBO<T>::~QPBO()
{
}

template<typename T>
void QPBO<T>::add_term1(const int3& p, T e0, T e1)
{
    _q.AddUnaryTerm(get_index(p), e0, e1);
}
template<typename T>
void QPBO<T>::add_term1(const int x, const int y, const int z, T e0, T e1)
{
    _q.AddUnaryTerm(get_index(x, y, z), e0, e1);
}

template<typename T>
void QPBO<T>::add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11)
{
    int i1 = get_index(p1);
    int i2 = get_index(p2);
    _q.AddPairwiseTerm(i1, i2, e00, e01, e10, e11);
}
template<typename T>
void QPBO<T>::add_term2(
    const int x1, const int y1, const int z1,
    const int x2, const int y2, const int z2,
    T e00, T e01, T e10, T e11)
{
    int i1 = get_index(x1, y1, z1);
    int i2 = get_index(x2, y2, z2);
    _q.AddPairwiseTerm(i1, i2, e00, e01, e10, e11);
}

template<typename T>
template<int N>
void QPBO<T>::add_term(const int3 [N], const T [1 << N])
{
    throw std::runtime_error("not implemented");
}

template<typename T>
T QPBO<T>::minimize()
{
    _q.Solve();
    _q.ComputeWeakPersistencies();
    return _q.ComputeTwiceEnergy() / 2.0;
}
template<typename T>
int QPBO<T>::get_var(const int3& p)
{
    return get_var(p.x, p.y, p.z);
}
template<typename T>
int QPBO<T>::get_var(const int x, const int y, const int z)
{
    int index = get_index(x, y, z);
    return _q.GetLabel(index);
}

