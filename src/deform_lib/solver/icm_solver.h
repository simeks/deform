#pragma once

#include <stk/math/types.h>
#include <vector>

template<typename T>
class ICMSolver
{
public:
    typedef T FlowType;

    ICMSolver(const int3& size) : _size(size)
    {
        int n = size.x*size.y*size.z;
        _f0.insert(_f0.begin(), n, T{0});
        _f1.insert(_f1.begin(), n, T{0});
        _var.insert(_var.begin(), n, 0);
    }
    ~ICMSolver() {}

    void add_term1(int x, int y, int z, T e0, T e1)
    {
        _f0[index(x,y,z)] += e0;
        _f1[index(x,y,z)] += e1;
    }
    
    void add_term2(int x1, int y1, int z1, int /*x2*/, int /*y2*/, int /*z2*/,
                   T e00, T /*e01*/, T e10, T /*e11*/)
    {
        _f0[index(x1,y1,z1)] += e00;
        _f1[index(x1,y1,z1)] += e10;
    }

    T minimize()
    {
        T e = 0;
        for (int i = 0; i < _f0.size(); ++i) {
            if (_f1[i] < _f0[i]) {
                _var[i] = 1;
                e += _f1[i];
            } else {
                _var[i] = 0;
                e += _f0[i];
            }
        }
        return e;
    }

    int get_var(int x, int y, int z)
    {
        return _var[index(x,y,z)];
    }

private:
    int index(int x, int y, int z)
    {
        return x + y*_size.x + z*_size.x*_size.y;
    }

    int3 _size;
    std::vector<T> _f0;
    std::vector<T> _f1;
    std::vector<int> _var;

};