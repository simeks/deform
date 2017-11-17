#pragma once

//#define DF_ENABLE_STATS

/* 
    Statistics gathering tools
*/

#ifdef DF_ENABLE_STATS
#define STATS_ADD_VALUE(name, value) \
    stats::add_stat_value(name, value)
#define STATS_DUMP(name, filename) \
    stats::dump_stat(name, filename)
#define STATS_RESET(name) \
    stats::reset_stat(name)

#else
#define STATS_ADD_VALUE(name, value)
#define STATS_DUMP(name, filename)
#define STATS_RESET(name)
#endif

namespace stats
{
    void reset_stat(const char* name);
    void dump_stat(const char* name, const char* filename);
    
    // Not thread-safe
    void add_stat_value(const char* name, double value);
}

