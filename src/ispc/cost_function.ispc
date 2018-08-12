#define AT(data, s, _x, _y, _z) (data)[(_z)*(s.y)*(s.x) + (_y)*(s.x) + (_x)]


struct int3 { int x, y, z; };
struct float3 { float x, y, z; };


/*!
 * \brief Trilinear interpolation of the intensity for a tuple of
 *        non-integer indices.
 *
 * The function assumes a zero-padding of thickness 1.
 *
 * @param data Data array.
 * @param s Size of the volume (including the padding).
 * @param x Non-integer coordinate along the x axis. 
 * @param y Non-integer coordinate along the y ayis. 
 * @param z Non-integer coordinate along the z azis. 
 * @returns Trilinear interpolation of the intensity at (x, y, z).
 */
static inline float linear_at(
        const uniform float data[],
        const uniform int3 s,
        const float x,
        const float y,
        const float z
        )
{
    // Equivalent to a constant border of zero
    const float x_ = clamp(x, (const varying float) 0.0f, (const varying float) s.x - 1.0f);
    const float y_ = clamp(y, (const varying float) 0.0f, (const varying float) s.y - 1.0f);
    const float z_ = clamp(z, (const varying float) 0.0f, (const varying float) s.z - 1.0f);

    const float xt = x_ - floor(x_);
    const float yt = y_ - floor(y_);
    const float zt = z_ - floor(z_);

    const int x1 = (int) floor(x_);
    const int x2 = (int) ceil(x_);
    const int y1 = (int) floor(y_);
    const int y2 = (int) ceil(y_);
    const int z1 = (int) floor(z_);
    const int z2 = (int) ceil(z_);

    return
        (1.0f - zt) *
            (
                (1.0f - yt) *
                    (
                        (1.0f - xt) * AT(data, s, x1, y1, z1) +
                        (xt)        * AT(data, s, x2, y1, z1)
                    ) +

                (yt) *
                    (
                        (1.0f - xt) * AT(data, s, x1, y2, z1) +
                        (xt)        * AT(data, s, x2, y2, z1)
                    )
            ) +
        (zt) *
            (
                (1.0f - yt) *
                    (
                        (1.0f - xt) * AT(data, s, x1, y1, z2) +
                        (xt)        * AT(data, s, x2, y1, z2)
                    ) +

                (yt) *
                    (
                        (1.0f - xt) * AT(data, s, x1, y2, z2) +
                        (xt)        * AT(data, s, x2, y2, z2)
                    )
            );
}


/*!
 * \brief Compute NCC on one couple of fixed and moving voxels.
 *
 * This function implements a heavily vectorised kernel to compute
 * normalised cross correlation.  The fixed image is assumed to have a
 * zero padding with thickness equal to the window radius, so that no
 * bound check is required, while the moving image is assumed to have a
 * zero padding with thickness 1, so that clamping to the edge is
 * equivalent to having a constant border mode.
 *
 * @param r        NCC window radius.
 * @param fixed_p  Indices of the point in the fixed image.
 * @param moving_p Indices of the point in the moving image.
 * @param fixed    Data array for the fixed image.
 * @param moving   Data array for the moving image.
 * @param fs       Size of the fixed image.
 * @param ms       Size of the moving image.
 * @return Value of the NCC for the given couple of corresponding
 *         voxels. 
 */
export uniform float ncc(
        const uniform int r,
        const uniform int3& fixed_p,
        const uniform float3& moving_p,
        const uniform float fixed[],
        const uniform float moving[],
        const uniform int3& fs,
        const uniform int3& ms
        )
{
    float _sff = 0.0f;
    float _smm = 0.0f;
    float _sfm = 0.0f;
    float _sf = 0.0f;
    float _sm = 0.0f;

    foreach_tiled (dz = -r ... r, dy = -r ... r, dx = -r ... r) {

        int fx = fixed_p.x + dx;
        int fy = fixed_p.y + dy;
        int fz = fixed_p.z + dz;

        float mx = moving_p.x + dx;
        float my = moving_p.y + dy;
        float mz = moving_p.z + dz;

        float fixed_v = AT(fixed, fs, fx, fy, fz);
        float moving_v = linear_at(moving, ms, mx, my, mz);

        _sff += fixed_v * fixed_v;
        _smm += moving_v * moving_v;
        _sfm += fixed_v*moving_v;
        _sm += moving_v;
        _sf += fixed_v;

    }

    uniform float sff = reduce_add(_sff);
    uniform float smm = reduce_add(_smm);
    uniform float sfm = reduce_add(_sfm);
    uniform const float sf = reduce_add(_sf);
    uniform const float sm = reduce_add(_sm);
    uniform const int n = (2*r + 1) * (2*r + 1) * (2*r + 1);

    // Subtract mean
    sff -= (sf * sf / n);
    smm -= (sm * sm / n);
    sfm -= (sf * sm / n);
    
    uniform float d = sqrt(sff*smm);

    return d > 1e-14 ? 0.5f * (1.0f - sfm / d) : 0.0f;
}


#undef AT

