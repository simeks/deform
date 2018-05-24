#include "block_change_flags.h"

#include <stk/math/int3.h>

BlockChangeFlags::BlockChangeFlags(const int3& block_count)
{
    // Number of sub blocks
    _block_count = {
        2 * (block_count.x + 1),
        2 * (block_count.y + 1),
        2 * (block_count.z + 1)
    };

    _flags.resize(_block_count.x * _block_count.y * _block_count.z);
    std::fill(_flags.begin(), _flags.end(), uint8_t(1));
}
bool BlockChangeFlags::is_block_set(const int3& block_p, bool shift) const
{
    int3 sub_block_offset = shift ? int3{0, 0, 0} : int3{1, 1, 1};

    for (int z = 0; z < 2; ++z) {
    for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 2; ++x) {
        int3 sb = 2 * block_p + sub_block_offset + int3{x, y, z};
        if (flag(sb) != 0)
            return true;
    }
    }
    }

    return false;
}
void BlockChangeFlags::set_block(const int3& block_p, bool changed, bool shift)
{
    int3 sub_block_offset = shift ? int3{0, 0, 0} : int3{1, 1, 1};

    for (int z = 0; z < 2; ++z) {
    for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 2; ++x) {
        int3 sb = 2 * block_p + sub_block_offset + int3{x, y, z};
        
        uint8_t f = flag(sb);
        if (!shift)
            f = (f & 0x2) | (changed ? 1 : 0);
        else
            f = (f & 0x1) | ((changed ? 1 : 0) << 1);

        set(sb, f);
    }
    }
    }
}
uint8_t BlockChangeFlags::flag(const int3& subblock_p) const
{
    int i = subblock_p.z * _block_count.x * _block_count.y + subblock_p.y * _block_count.x + subblock_p.x;
    return _flags[i];
}
void BlockChangeFlags::set(const int3& subblock_p, uint8_t flags)
{
    int i = subblock_p.z * _block_count.x * _block_count.y + subblock_p.y * _block_count.x + subblock_p.x;
    _flags[i] = flags;
}
