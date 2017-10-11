#include "debug/assert.h"
#include "platform/timer.h"
#include "stats.h"

#include <fstream>
#include <string.h>

namespace
{
    const int max_stat_count = 64;
    
    struct Samples
    {
        double* _timestamps;
        double* _values;

        // Number of slots in data (in n of doubles)
        size_t _values_size; 

        // Current number of samples
        size_t _current_index;

        Samples() : 
            _timestamps(nullptr),
            _values(nullptr),
            _values_size(0),
            _current_index(0)
        {
        }
        ~Samples()
        {
            delete [] _timestamps;
            delete [] _values;
        }

        void reset()
        {
            _current_index = 0;
        }

        void push(double timestamp, double value)
        {
            // TODO: Thread safety

            if (!_values)
            {
                _values_size = 32;
                _timestamps = new double[_values_size];
                _values = new double[_values_size];
            }
            else if (_current_index >= _values_size)
            {
                // Exponential growth
                size_t new_size = _values_size * 2;
                double* new_timestamps = new double[new_size];
                double* new_values = new double[new_size];
                
                memcpy(new_timestamps, _timestamps, _values_size);
                memcpy(new_values, _values, _values_size);
                delete [] _timestamps;
                delete [] _values;
                
                _timestamps = new_timestamps;
                _values = new_values;
                _values_size = new_size;
            }
            
            _timestamps[_current_index] = timestamp;
            _values[_current_index] = value;
            ++_current_index;
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
        for (int i = 0; i < stat.samples._current_index; ++i)
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
