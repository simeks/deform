#pragma once

template<typename T>
GraphCut<T>::GraphCut(const int3& size)
    : _size(size)
    , _e(size.x * size.y * size.z, size.x * size.y * size.z * 3)
{
    _e.add_variable(size.x * size.y * size.z);
}
template<typename T>
GraphCut<T>::~GraphCut()
{
}

template<typename T>
void GraphCut<T>::add_term1(const int3& p, T e0, T e1)
{
    int index = get_index(p);
    _e.add_term1(index, e0, e1);
}
template<typename T>
void GraphCut<T>::add_term1(const int x, const int y, const int z, T e0, T e1)
{
    int index = get_index(x, y, z);
    _e.add_term1(index, e0, e1);
}

template<typename T>
void GraphCut<T>::add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11)
{
    int i1 = get_index(p1);
    int i2 = get_index(p2);
    _e.add_term2(i1, i2, e00, e01, e10, e11);
}
template<typename T>
void GraphCut<T>::add_term2(
    const int x1, const int y1, const int z1,
    const int x2, const int y2, const int z2,
    T e00, T e01, T e10, T e11)
{
    int i1 = get_index(x1, y1, z1);
    int i2 = get_index(x2, y2, z2);
    _e.add_term2(i1, i2, e00, e01, e10, e11);
}

template<typename T>
template<int N>
void GraphCut<T>::add_term(const int3 p[N], const T e[1 << N])
{
    static_assert(N == 3, "Only ternary terms are supported");
    int i1 = get_index(p[0]);
    int i2 = get_index(p[1]);
    int i3 = get_index(p[2]);
    _e.add_term3(i1, i2, i3, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]);
}

template<typename T>
T GraphCut<T>::minimize()
{
    return _e.minimize();
}
template<typename T>
int GraphCut<T>::get_var(const int3& p)
{
    return get_var(p.x, p.y, p.z);
}
template<typename T>
int GraphCut<T>::get_var(const int x, const int y, const int z)
{
    int index = get_index(x, y, z);
    return _e.get_var(index);
}

