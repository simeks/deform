#pragma once

class Volume;

/// Cortado-friendly interface towards stb_image and stb_image_write
namespace stb
{
    /// Reads a number of common 2D image formats, such as png, gif, jpeg.
    /// See stb_image.h for more information
    Volume read_image(const char* file);
    
    /// Only supports 2D images, meaning volume has to be of size 1 in z-direction
    /// Supported formats: .png, .bmp, .tga 
    void write_image(const char* file, const Volume& volume);
}
