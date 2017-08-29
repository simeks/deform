#pragma once


/// Very simple implementation of reading/writing of VTK files
/// This will in no way cover the whole VTK specs and have only 
/// been tested on a small subset of different volumes.

class Volume;

namespace vtk
{
    /// Reads a VTK-file and returns a volume with the read data
    Volume read_volume(const char* file);

    /// Writes a given volume to the given file in the VTK format
    void write_volume(const char* file, const Volume& vol);
}
