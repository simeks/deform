#pragma once

template<typename T>
GraphCut<T>::GraphCut(const int3& size)
    : Solver<T>(size)
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
    int index = Solver<T>::get_index(p);
    _e.add_term1(index, e0, e1);
}
template<typename T>
void GraphCut<T>::add_term1(const int x, const int y, const int z, T e0, T e1)
{
    int index = Solver<T>::get_index(x, y, z);
    _e.add_term1(index, e0, e1);
}

template<typename T>
void GraphCut<T>::add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11)
{
    int i1 = Solver<T>::get_index(p1);
    int i2 = Solver<T>::get_index(p2);
    _e.add_term2(i1, i2, e00, e01, e10, e11);
}
template<typename T>
void GraphCut<T>::add_term2(
    const int x1, const int y1, const int z1,
    const int x2, const int y2, const int z2,
    T e00, T e01, T e10, T e11)
{
    int i1 = Solver<T>::get_index(x1, y1, z1);
    int i2 = Solver<T>::get_index(x2, y2, z2);
    _e.add_term2(i1, i2, e00, e01, e10, e11);
}

template<typename T>
void GraphCut<T>::add_term3(const int3& p1, const int3& p2, const int3& p3,
                            T e000, T e001,
                            T e010, T e011,
                            T e100, T e101,
                            T e110, T e111)
{
    int i1 = Solver<T>::get_index(p1);
    int i2 = Solver<T>::get_index(p2);
    int i3 = Solver<T>::get_index(p3);
    _e.add_term3(i1, i2, i3, e000, e001, e010, e011, e100, e101, e110, e111);
}
template<typename T>
void GraphCut<T>::add_term3(const int x1, const int y1, const int z1,
                            const int x2, const int y2, const int z2,
                            const int x3, const int y3, const int z3,
                            T e000, T e001,
                            T e010, T e011,
                            T e100, T e101,
                            T e110, T e111)
{
    int i1 = Solver<T>::get_index(x1, y1, z1);
    int i2 = Solver<T>::get_index(x2, y2, z2);
    int i3 = Solver<T>::get_index(x3, y3, z3);
    _e.add_term3(i1, i2, i3, e000, e001, e010, e011, e100, e101, e110, e111);
}

template<typename T>
T GraphCut<T>::minimize()
{
    return _e.minimize();
}
template<typename T>
int GraphCut<T>::get_var(const int3& p)
{
    int index = Solver<T>::get_index(p.x, p.y, p.z);
    return _e.get_var(index);
}
template<typename T>
int GraphCut<T>::get_var(const int x, const int y, const int z)
{
    int index = Solver<T>::get_index(x, y, z);
    return _e.get_var(index);
}

