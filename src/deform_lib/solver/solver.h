#pragma once

#include <stk/math/math.h>

#include <vector>

template<typename T>
class Solver
{
public:
    Solver(const int3& size) : _size(size) {}
    virtual ~Solver() {}

    virtual void add_term1(const int3& p, T e0, T e1) = 0;
    virtual void add_term1(const int x, const int y, const int z, T e0, T e1) = 0;

    virtual void add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11) = 0;
    virtual void add_term2(const int x1, const int y1, const int z1,
                           const int x2, const int y2, const int z2,
                           T e00, T e01, T e10, T e11) = 0;

	virtual void add_term_n(const std::vector<int3>& p, const std::vector<T> e) = 0;

    virtual T minimize() = 0;

    virtual int get_var(const int3& p) = 0;
    virtual int get_var(const int x, const int y, const int z) = 0;

protected:
    int get_index(const int x, const int y, const int z) const {
        return x + y*_size.x + z*_size.x*_size.y;
    }

    int get_index(const int3& p) const {
        return p.x + p.y*_size.x + p.z*_size.x*_size.y;
    }

private:
    int3 _size;
};

