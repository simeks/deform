template<typename T>
GraphCut<T>::GraphCut(const Vec3i& size) :
    _e(size.x * size.y * size.z, size.x * size.y * size.z * 3),
    _size(size)
{
    _e.add_variable(size.x * size.y * size.z);
}
template<typename T>
GraphCut<T>::~GraphCut()
{

}

template<typename T>
void GraphCut<T>::add_term1(const Vec3i& p, T e0, T e1)
{
    int index = get_index(p);
    _e.add_term1(index, e0, e1);
}
template<typename T>
void GraphCut<T>::add_term1(int x, int y, int z, T e0, T e1)
{
    int index = get_index(x, y, z);
    _e.add_term1(index, e0, e1);
}

template<typename T>
void GraphCut<T>::add_term2(const Vec3i& p1, const Vec3i& p2, T e00, T e01, T e10, T e11)
{
    int i1 = get_index(p1);
    int i2 = get_index(p2);
    _e.add_term2(i1, i2, e00, e01, e10, e11);
}
template<typename T>
void GraphCut<T>::add_term2(
    int x1, int y1, int z1,
    int x2, int y2, int z2,
    T e00, T e01, T e10, T e11)
{
    int i1 = get_index(x1, y1, z1);
    int i2 = get_index(x2, y2, z2);
    _e.add_term2(i1, i2, e00, e01, e10, e11);
}

template<typename T>
T GraphCut<T>::minimize()
{
    return _e.minimize();
}
template<typename T>
int GraphCut<T>::get_var(const Vec3i& p)
{
    int index = get_index(p.x, p.y, p.z);
    return _e.get_var(index);
}
template<typename T>
int GraphCut<T>::get_var(int x, int y, int z)
{
    int index = get_index(x, y, z);
    return _e.get_var(index);
}

template<typename T>
int GraphCut<T>::get_index(int x, int y, int z) const
{
    return x + y*_size.x + z*_size.x*_size.y;
}
template<typename T>
int GraphCut<T>::get_index(const Vec3i& p) const
{
    return p.x + p.y*_size.x + p.z*_size.x*_size.y;
}
