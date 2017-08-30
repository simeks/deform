#pragma once

#include <string>

/// Very simple implementation of reading/writing of VTK files
/// This will in no way cover the whole VTK specs and have only 
/// been tested on a small subset of different volumes.

class Volume;

namespace vtk
{
    class Reader
    {
    public:
        Reader();
        ~Reader();

        /// Executes the reader
        /// If read fails a invalid Volume is returned, see last_error() for error mesage.
        /// file : Path to the image file
        Volume execute(const char* file);

        /// Returns true if the read failed
        /// See last_error for further information
        bool failed() const;
        
        /// Returns the last error, empty string if no error has occured
        const char* last_error() const;

    private:
        std::string _error;
    };

    /// Reads a VTK-file and returns a volume with the read data
    /// Returns the read volume or an invalid volume if the read failed.
    Volume read_volume(const char* file);

    /// Writes a given volume to the given file in the VTK format
    void write_volume(const char* file, const Volume& vol);
}
