#include "debug/assert.h"
#include "platform/timer.h"
#include "stats.h"

#include <fstream>
#include <string.h>
#include <vector>

namespace
{
    const int max_stat_count = 64;
    
    struct Samples
    {
        std::vector<double> _timestamps;
        std::vector<double> _values;

        Samples()
        {
        }
        ~Samples()
        {
        }

        void reset()
        {
            _timestamps.clear();
            _values.clear();
        }

        void push(double timestamp, double value)
        {
            // TODO: Thread safety
            
            _timestamps.push_back(timestamp);
            _values.push_back(value);
        }
    };

    struct Stat
    {
        Stat() {}
        ~Stat() {}

        const char* name;
        Samples samples;
        double start_time;
    };

    Stat stat_entries[max_stat_count];
    int num_stat_entries = 0;

    // Tries to find a stat with the given name. 
    //  If not stat was found: creates a new one
    Stat& find_stat(const char* name)
    {
        assert(name);
        for (int i = 0; i < num_stat_entries; ++i)
        {
            if (strcmp(stat_entries[i].name, name) == 0)
            {
                return stat_entries[i];
            }
        }
        assert(num_stat_entries < max_stat_count);
        auto& stat = stat_entries[num_stat_entries++];
        stat.name = name;
        stat.start_time = timer::seconds();
        return stat;
    }

}

namespace stats
{
    void reset_stat(const char* name)
    {
        assert(name);

        auto& stat = find_stat(name);
        stat.samples.reset();
        stat.start_time = timer::seconds();
    }
    void dump_stat(const char* name, const char* filename)
    {
        assert(name);

        auto& stat = find_stat(name);

        std::ofstream f;
        f.open(filename, std::ios::out);
        for (int i = 0; i < int(stat.samples._timestamps.size()); ++i)
        {
            f << stat.samples._timestamps[i] << ";" << stat.samples._values[i] << std::endl;
        }
        f.close();
    }
    void add_stat_value(const char* name, double value)
    {
        assert(name);

        auto& stat = find_stat(name);

        double ts = timer::seconds() - stat.start_time;
        stat.samples.push(ts, value);
    }
}
