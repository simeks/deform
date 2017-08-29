#include "volume.h"
#include "vtk.h"

#include <assert.h>
#include <fstream>
#include <intrin.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    void byteswap_16(uint16_t* p, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            p[i] = _byteswap_ushort(p[i]);
    }
    void byteswap_32(uint32_t* p, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            p[i] = _byteswap_ulong(p[i]);
    }
    void byteswap_64(uint64_t* p, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            p[i] = _byteswap_uint64(p[i]);
    }

    void write_big_endian_16(uint16_t* p, size_t n, std::ofstream& f)
    {
        for (size_t i = 0; i < n; ++i)
        {
            uint16_t c = _byteswap_ushort(p[i]);
            f.write((const char*)&c, 2);
        }
    }
    void write_big_endian_32(uint32_t* p, size_t n, std::ofstream& f)
    {
        for (size_t i = 0; i < n; ++i)
        {
            uint32_t c = _byteswap_ulong(p[i]);
            f.write((const char*)&c, 4);
        }
    }
    void write_big_endian_64(uint64_t* p, size_t n, std::ofstream& f)
    {
        for (size_t i = 0; i < n; ++i)
        {
            uint64_t c = _byteswap_uint64(p[i]);
            f.write((const char*)&c, 8);
        }
    }
}

namespace vtk
{
    Volume read_volume(const char* file)
    {
        // Spec: http://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf

        //# vtk DataFile Version 3.0
        //<Title>
        //BINARY
        //DATASET STRUCTURED_POINTS
        //DIMENSIONS 256 256 740
        //ORIGIN -249.023 -249.023 21.0165
        //SPACING 1.9531 1.9531 2.6001
        //POINT_DATA 48496640
        //SCALARS image_data double
        //LOOKUP_TABLE default

        std::ifstream f(file, std::ios::binary);
        assert(f.is_open());

        std::string line;
        
        //# vtk DataFile Version 3.0
        std::getline(f, line);
        //<Title>
        std::getline(f, line);

        //BINARY
        std::getline(f, line);
        if (line != "BINARY")
        {
            std::cout << "Invalid format: " << line << std::endl;
            f.close();
            return Volume();
        }

        //DATASET STRUCTURED_POINTS
        std::getline(f, line);
        if (line != "DATASET STRUCTURED_POINTS")
        {
            std::cout << "Unexpected dataset: " << line << std::endl;
            f.close();
            return Volume();
        }

        Dims size = { 0, 0, 0 };
        // Hopefully we wont have files as large as 2^64
        size_t point_data = size_t(~0);
        uint8_t voxel_type = voxel::Type_Unknown;
        int num_comp = 1;

        std::string key;
        std::string value;
        while (std::getline(f, line))
        {
            std::stringstream ss(line);
            
            ss >> key;
            //DIMENSIONS 256 256 740
            if (key == "DIMENSIONS")
            {
                ss >> value;
                size.width = atoi(value.c_str());
                ss >> value;
                size.height = atoi(value.c_str());
                ss >> value;
                size.depth = atoi(value.c_str());
            }
            //ORIGIN - 249.023 - 249.023 21.0165
            else if (key == "ORIGIN")
            {
                // TODO:
            }
            //SPACING 1.9531 1.9531 2.6001
            else if (key == "SPACING")
            {
                // TODO:
            }
            //POINT_DATA 48496640
            else if (key == "POINT_DATA")
            {
                ss >> value;
                point_data = atoll(value.c_str());
            }
            //SCALARS image_data double
            else if (key == "SCALARS")
            {
                // SCALARS dataName dataType numComp
                // Here i think numComp is optional 

                ss >> value; // dataName, don't know what this is good for
                std::string data_type;
                ss >> data_type;
                
                if (!ss.eof())
                {
                    std::string num_comp_s;
                    ss >> num_comp_s;
                    num_comp = atoi(num_comp_s.c_str());
                }

                if (data_type == "double")
                {
                    if (num_comp == 1) voxel_type = voxel::Type_Double;
                    if (num_comp == 2) voxel_type = voxel::Type_Double2;
                    if (num_comp == 3) voxel_type = voxel::Type_Double3;
                    if (num_comp == 4) voxel_type = voxel::Type_Double4;
                }
                else if (data_type == "float")
                {
                    if (num_comp == 1) voxel_type = voxel::Type_Float;
                    if (num_comp == 2) voxel_type = voxel::Type_Float2;
                    if (num_comp == 3) voxel_type = voxel::Type_Float3;
                    if (num_comp == 4) voxel_type = voxel::Type_Float4;
                }
                
                if (voxel_type == voxel::Type_Unknown)
                {
                    std::cout << "Unsupported data type: " << data_type << " " << num_comp << std::endl;
                    f.close();
                    return Volume();
                }
            }
            //LOOKUP_TABLE default
            else if (key == "LOOKUP_TABLE")
            {
                ss >> value;
                if (value != "default")
                {
                    std::cout << "Invalid parameter to LOOKUP_TABLE: " << value << std::endl;
                    f.close();
                    return Volume();
                }
                // Assume that blob comes after this line
                break;
            }
        }

        if (size.width == 0 || size.height == 0 || size.depth == 0)
        {
            std::cout << "Invalid volume size: " <<
                size.width << ", " <<
                size.height << ", " <<
                size.depth << std::endl;
            f.close();
            return Volume();
        }

        if (voxel_type == voxel::Type_Unknown)
        {
            std::cout << "Invalid voxel type" << std::endl;
            f.close();
            return Volume();
        }

        if (point_data == size_t(~0))
        {
            std::cout << "Invalid point_data" << std::endl;
        }


        // Allocate volume
        Volume vol(size, voxel_type);

        size_t num_bytes = size.width * size.height * size.depth * voxel::size(voxel_type);
        f.read((char*)vol.ptr(), num_bytes);
        f.close();

        // Switch to little endian
        size_t bytes_per_elem = voxel::size(voxel_type) / num_comp;
        size_t num_values = size.width * size.height * size.depth *  num_comp;
        if (bytes_per_elem == 8) // double, uint64_t
            byteswap_64((uint64_t*)vol.ptr(), num_values);
        if (bytes_per_elem == 4) // float, uint32_t, ...
            byteswap_32((uint32_t*)vol.ptr(), num_values);
        if (bytes_per_elem == 2) // short
            byteswap_16((uint16_t*)vol.ptr(), num_values);
        
        // We don't need to do anything for 1 byte elements

        return vol;
    }

    void write_volume(const char* file, const Volume& vol)
    {
        assert(vol.valid());
        assert(vol.voxel_type() != voxel::Type_Unknown);

        //# vtk DataFile Version 3.0
        //<Title>
        //BINARY
        //DATASET STRUCTURED_POINTS
        //DIMENSIONS 256 256 740
        //ORIGIN -249.023 -249.023 21.0165
        //SPACING 1.9531 1.9531 2.6001
        //POINT_DATA 48496640
        //SCALARS image_data double
        //LOOKUP_TABLE default

        std::ofstream f(file, std::ios::binary);
        f << "# vtk DataFile Version 3.0\n";
        f << "Written by cortado (vtk.cpp)\n";
        f << "BINARY\n";
        f << "DATASET STRUCTURED_POINTS\n";

        auto size = vol.size();
        f << "DIMENSIONS " << size.width << " " << size.height << " " << size.depth << "\n";
        f << "ORIGIN 0 0 0\n"; // TODO:
        f << "SPACING 1 1 1\n"; // TODO:
        f << "POINT_DATA " << size.width * size.height * size.depth << "\n";

        std::string data_type;
        int num_comp = 1;
        switch (vol.voxel_type())
        {
        case voxel::Type_Float:
            data_type = "float";
            num_comp = 1;
            break;
        case voxel::Type_Float2:
            data_type = "float";
            num_comp = 2;
            break;
        case voxel::Type_Float3:
            data_type = "float";
            num_comp = 3;
            break;
        case voxel::Type_Float4:
            data_type = "float";
            num_comp = 4;
            break;
        case voxel::Type_Double:
            data_type = "double";
            num_comp = 1;
            break;
        case voxel::Type_Double2:
            data_type = "double";
            num_comp = 2;
            break;
        case voxel::Type_Double3:
            data_type = "double";
            num_comp = 3;
            break;
        case voxel::Type_Double4:
            data_type = "double";
            num_comp = 4;
            break;
        default:
            std::cout << "Unsupported format" << std::endl;
            return;
        };

        f << "SCALARS image_data " << data_type << " " << num_comp << "\n";
        f << "LOOKUP_TABLE default\n";

        // Switch to big endian
        size_t bytes_per_elem = voxel::size(vol.voxel_type()) / num_comp;
        size_t num_values = size.width * size.height * size.depth * num_comp;

        if (bytes_per_elem == 8) // double, uint64_t
            write_big_endian_64((uint64_t*)vol.ptr(), num_values, f);
        else if (bytes_per_elem == 4) // float, uint32_t, ...
            write_big_endian_32((uint32_t*)vol.ptr(), num_values, f);
        else if (bytes_per_elem == 2) // short
            write_big_endian_16((uint16_t*)vol.ptr(), num_values, f);
        else
            f.write((const char*)vol.ptr(), num_values);

        f.close();
    }
}