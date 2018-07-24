#pragma once

#include <deform_lib/registration/settings.h>
#include <stk/math/float3.h>

#include <string>
#include <vector>

/*!
 * \brief Parse a landmark file, passed as a string.
 *
 * \note The landmarks are in image space coordinates.
 *
 * Use the Elastix format for landmark files (only point, index is
 * currently unsupported).  The first line of the file should be the
 * word "point", the second line should be the number `n` of points, and
 * each of the following `n` lines, one for each landmark point, should
 * contain three space-separated floating point coordinates in image
 * space.
 *
 * @param filename String containing the landmark filename.
 * @return A vector of points.
 *
 * @throws ValidationError
 */
std::vector<float3> parse_landmarks_file(const std::string& filename);

