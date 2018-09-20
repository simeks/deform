#pragma once

#include <stk/math/types.h>

#include <stdint.h>
#include <vector>

class BlockChangeFlags
{
public:
    BlockChangeFlags(const int3& block_count = {0,0,0});

    bool is_block_set(const int3& block_p, bool shift) const;
    void set_block(const int3& block_p, bool changed, bool shift);

private:
    uint8_t flag(const int3& subblock_p) const;
    void set(const int3& subblock_p, uint8_t flags);

    int3 _block_count;
    std::vector<uint8_t> _flags;
};
