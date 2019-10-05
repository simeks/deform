#pragma once

#include <stk/math/types.h>
#include <vector>

#include <GridCut/GridGraph_3D_6C.h>

template<typename T>
class GridCutSolver
{
public:
    typedef T FlowType;

    GridCutSolver(const int3& size) :
        _size(size),
        _graph(size.x, size.y, size.z)
    {
        size_t n = _size.x * _size.y * _size.z;
        _tweights.insert(_tweights.begin(), 2*n, 0.0);
    }
    ~GridCutSolver() {}

    void add_term1(int x, int y, int z, T e0, T e1)
    {
        add_tweights(x, y, z, e1, e0);
    }
    
    void add_term2(int x1, int y1, int z1, int x2, int y2, int z2,
                   T e00, T e01, T e10, T e11)
    {
        add_tweights(x1, y1, z1, e11, e00);

        e01 -= e00; e10 -= e11;

        ASSERT(e01 + e10 >= 0); // check regularity
        if (e01 < 0) {
            add_tweights(x1, y1, z1, 0, e01);
            add_tweights(x2, y2, z2, 0, -e01);
            add_edge(x1, y1, z1, x2, y2, z2, 0, e01+e10);
        }
        else if (e10 < 0) {
            add_tweights(x1, y1, z1, 0, -e10);
            add_tweights(x2, y2, z2, 0, e10);
            add_edge(x1, y1, z1, x2, y2, z2, e01+e10, 0);
        }
        else {
            add_edge(x1, y1, z1, x2, y2, z2, e01, e10);
        }
    }

    T minimize()
    {
        for (int z = 0; z < _size.z; ++z) {
        for (int y = 0; y < _size.y; ++y) {
        for (int x = 0; x < _size.x; ++x) {
            T e0 = _tweights[index(x,y,z)];
            T e1 = _tweights[index(x,y,z)+1];
            _graph.set_terminal_cap(_graph.node_id(x,y,z), e0, e1);
        }}}

        _graph.compute_maxflow();
        return _graph.get_flow();
    }

    int get_var(int x, int y, int z)
    {
        return _graph.get_segment(_graph.node_id(x,y,z));
    }

private:
    // index in _tweights, not the graph itself
    size_t index(int x, int y, int z)
    {
        return 2*x + y * 2*_size.x + z * 2*_size.x
                * _size.y;
    }
    void add_tweights(int x, int y, int z, T e0, T e1)
    {
        _tweights[index(x,y,z)] += e0;
        _tweights[index(x,y,z)+1] += e1;
    }
    void add_edge(int x1, int y1, int z1, int x2, int y2, int z2, 
                  T cap, T rev_cap)
    {
        _graph.set_neighbor_cap(
            _graph.node_id(x1,y1,z1), x2-x1, y2-y1, z2-z1, cap
        );
        _graph.set_neighbor_cap(
            _graph.node_id(x2,y2,z2), x1-x2, y1-y2, z1-z2, rev_cap
        );
    }

    int3 _size;
    GridGraph_3D_6C<T, T, T> _graph;
    std::vector<T> _tweights;
};