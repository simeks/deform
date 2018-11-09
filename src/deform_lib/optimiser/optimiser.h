#pragma once

#include <stk/math/math.h>

template<typename T>
class Optimiser
{
public:
    Optimiser(const int3& size) : _size(size) {}
    virtual ~Optimiser() {}

    virtual void add_term1(const int3& p, T e0, T e1) = 0;
    virtual void add_term1(int x, int y, int z, T e0, T e1) = 0;

    virtual void add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11) = 0;
    virtual void add_term2(int x1, int y1, int z1,
                   int x2, int y2, int z2,
                   T e00, T e01, T e10, T e11) = 0;

    virtual T minimize() = 0;

    virtual int get_var(const int3& p) = 0;
    virtual int get_var(int x, int y, int z) = 0;

protected:
    int get_index(int x, int y, int z) const {
        return x + y*_size.x + z*_size.x*_size.y;
    }

    int get_index(const int3& p) const {
        return p.x + p.y*_size.x + p.z*_size.x*_size.y;
    }

private:
    int3 _size;
};

