
template<typename T>
VolumeHelper<T> filters::normalize(const VolumeHelper<T>& in, T min, T max)
{
    T in_min, in_max;
    in.min_max(in_min, in_max);

    Dims dims = in.size();
    VolumeHelper<T> out(dims);
    out.set_origin(in.origin());
    out.set_spacing(in.spacing());

    double range = double(max - min);
    double in_range = double(in_max - in_min);
    
    #pragma omp parallel for
    for (int z = 0; z < int(dims.depth); ++z)
    {
        for (int y = 0; y < int(dims.height); ++y)
        {
            for (int x = 0; x < int(dims.width); ++x)
            {
                out(x,y,z) = T(range * (in(x, y, z) - in_min) / in_range + min);
            }
        }
    }
    return out;
}
