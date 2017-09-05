#include "config_file.h"
#include "registration/registration_engine.h"
#include "registration/volume_pyramid.h"

#include <framework/debug/assert.h>
#include <framework/debug/log.h>
#include <framework/filters/gaussian_filter.h>
#include <framework/platform/file_path.h>
#include <framework/volume/volume.h>
#include <framework/volume/volume_helper.h>
#include <framework/volume/stb.h>
#include <framework/volume/vtk.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

class ArgParser
{
public:
    ArgParser(int argc, char** argv)
    {
        _executable = argv[0];

        std::vector<std::string> tokens;
        for (int i = 1; i < argc; ++i)
            tokens.push_back(argv[i]);

        while (!tokens.empty())
        {
            const std::string& token = tokens.back();
            
            if (token[0] == '-')
            {
                int b = 1;
                if (token[1] == '-')
                {
                    b = 2;
                }

                std::string line = token.substr(b);
                size_t p = line.find('=');
                if (p != std::string::npos)
                {
                    std::string key = line.substr(0, p);
                    std::string value = line.substr(p + 1);
                    _values[key] = value;
                }
                else
                {
                    _values[line] = "";
                }
            }
            else
            {
                _tokens.push_back(token);
            }
            tokens.pop_back();
        }
    }

    bool is_set(const std::string& key) const
    {
        return _values.find(key) != _values.end();
    }
    const std::string& value(const std::string& key) const
    {
        assert(is_set(key));
        return _values.at(key);
    }
    const std::map<std::string, std::string>& values() const
    {
        return _values;
    }

    const std::string& token(int i) const
    {
        return _tokens[i];
    }
    int num_tokens() const
    {
        return (int)_tokens.size();
    }
    const std::string& executable() const
    {
        return _executable;
    }

private:
    std::string _executable;
    std::map<std::string, std::string> _values;
    std::vector<std::string> _tokens;

};

// Identifies and loads the given file
// file : Filename
// is_2d [out] : Indicates if the volume is a 2d volume
// Returns the loaded volume, if load failed the returned volume will be flagged as invalid 
Volume load_volume(const std::string& file)
{
    FilePath path(file);
    std::string ext = path.extension();
    
    // To lower case
    std::transform(ext.begin(), ext.end(), ext.begin(), [](char c){ return (char)::tolower(c); });

    if (ext == "vtk")
    {
        vtk::Reader reader;
        Volume vol = reader.execute(file.c_str());
        if (!vol.valid())        
        {
            LOG(Error, "Failed to read image: %s\n", reader.last_error());
        }
        return vol;
    }
    else if (ext == "png")
    {
        std::string err;
        Volume vol = stb::read_image(file.c_str());
        if (!vol.valid())
        {
            LOG(Error, "Failed to read image: %s\n", stb::last_read_error());
        }
        return vol;
    }
    else
    {
        LOG(Error, "Unsupported file extension: '%s'\n", ext.c_str());
    }
    // Returning an "invalid" volume
    return Volume();
}

void print_help()
{
    std::cout   << "Arguments:" << std::endl
                << "-p=<filename> : parameter file (obligatory)" << std::endl
                << "-f<i>=<filename> : Filename for the i:th fixed image" << std::endl
                << "-m<i>=<filename> : Filename for the i:th moving image" << std::endl
                << "-h, --help : Show this help section" << std::endl;

}



Volume downsample_volume_gaussian(const Volume& vol, float scale)
{
    scale;
    assert(scale <= 1.0f);
    //assert(vol.voxel_type() == voxel::Type_UChar);

    return vol.clone();
}

int main(int argc, char* argv[])
{
    ArgParser args(argc, argv);

    // if (args.is_set("help") || args.is_set("h"))
    // {
    //     print_help();
    //     return 1;
    // }

    // if (args.num_tokens() < 1)
    // {
    //     print_help();
    //     return 1;
    // }

    vtk::Reader reader;
    Volume vol = reader.execute("C:\\data\\test.vtk");//args.token(0).c_str());
    if (reader.failed())
    {
        std::cout << reader.last_error();
        return 1;
    }

    //VolumePyramid pyramid(5);
    //pyramid.build_from_base(vol, downsample_volume_gaussian);

    vtk::write_volume("C:\\data\\gauss_test_10.vtk",  filters::gaussian_filter_3d(vol, 10.0f));
    vtk::write_volume("C:\\data\\gauss_test_20.vtk",  filters::gaussian_filter_3d(vol, 20.0f));
    vtk::write_volume("C:\\data\\gauss_test_50.vtk",  filters::gaussian_filter_3d(vol, 50.0f));
    vtk::write_volume("C:\\data\\gauss_test_100.vtk",  filters::gaussian_filter_3d(vol, 100.0f));

    // std::string param_file;
    // if (args.is_set("p"))
    // {
    //     param_file = args.value("p");
    // }
    // else
    // {
    //     print_help();
    //     return 1;
    // }

    // ConfigFile cfg(param_file);
    // RegistrationEngine engine;

    // std::string fi, mi, fixed_file, moving_file;
    // for (int i = 0; ; ++i)
    // {
    //     std::stringstream ss;
    //     ss << "-f" << i;
    //     fi = ss.str();

    //     ss.str("");
    //     ss << "-m" << i;
    //     mi = ss.str();

    //     if (args.is_set(fi) && args.is_set(mi))
    //     {
    //         // engine.set_fixed_image(i, file);
    //         // engine.set_moving_image(i, file);
    //     }
    // }

    // Optimizer* optimizer = new BlockedGraphCut();
    // engine.set_optimizer(optimizer);


    // if (!engine.initialize(cfg))
    // {
    //     return 1;
    // }


    // engine.shutdown();

    return 0;
}