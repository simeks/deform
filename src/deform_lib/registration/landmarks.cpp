#include <deform_lib/registration/landmarks.h>

#include <fstream>

namespace {
    std::string trim_string(const std::string& str)
    {
        size_t i = 0, j = str.size() - 1; 

        while (i < j && isspace(str[i])) i++;
        while (j > i && isspace(str[j])) j--;

        return std::string(str, i, j-2 );
    }
}

std::vector<float3> parse_landmarks_file(const std::string& filename)
{
    std::vector<float3> landmarks;
    std::string line;
    long n;

    std::ifstream f {filename, std::ifstream::in};
    if (!f.is_open()) {
        throw ValidationError("Failed to open file '" + filename + "'");
    }

    // Read first line, must be "point"
    std::getline(f, line);
    if ("point" != line) {
        throw ValidationError("Invalid point file format, expected \"point\" at line 1");
    }

    // Read second line, must be the number of points (but tolerate stray whitespace)
    std::getline(f, line);

    line = trim_string(line);
    if (line.find_first_of(" \t") == std::string::npos) {
        // No whitespace signals that there are no other numbers in string
        n = std::stol(line);
    }
    else {
        throw ValidationError("Invalid point file, expected number of points at line 2");
    }

    // Read one line for each point
    landmarks.resize(n);
    for (long i = 0; i < n; ++i) {
        std::getline(f, line);
        float x, y, z;
        if(3 != std::sscanf(line.c_str(), "%f %f %f", &x, &y, &z)) {
            throw ValidationError("Wrong number of components for point " +
                                  std::to_string(i + 1));
        }
        landmarks[i] = {x, y, z};
    }

    return landmarks;
}
