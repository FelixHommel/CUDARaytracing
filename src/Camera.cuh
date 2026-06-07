#ifndef CRT_SRC_CAMERA_CUH
#define CRT_SRC_CAMERA_CUH

#include "src/Ray.cuh"

#include <cuda_runtime_api.h>

namespace
{

constexpr auto DEF_ORIGIN{ Vec3(0.0, 0.0, 0.0) };
constexpr auto DEF_LOWER_LEFT_CORNER{ Vec3(-2.0, -1.0, -1.0) };
constexpr auto DEF_HORIZONTAL{ Vec3(2.0 + 2.0, 0.0, 0.0) };
constexpr auto DEF_VERTICAL{ Vec3(0.0, 2.0, 0.0) };

} // namespace

/// \brief The Camera is where the \ref Ray are cast from.
///
/// \author Felix Hommel
/// \date 6/6/2026
struct Camera // NOLINT
{
    __device__ Camera() {}  // NOLINT
    __device__ ~Camera() {} // NOLINT

    /// \brief Get a Ray that is cast towards a specific point (u, v)
    ///
    /// \param u Position on the X-Axis
    /// \param v Position on the Y-Axis
    ///
    /// \returns \ref Ray from the camera center to (u, v)
    __device__ Ray getRay(float u, float v) const noexcept
    {
        return { origin, lowerLeftCorner + (u * horizontal) + (v * vertical) };
    }

    Vec3 origin{ ::DEF_ORIGIN };
    Vec3 lowerLeftCorner{ ::DEF_LOWER_LEFT_CORNER };
    Vec3 horizontal{ ::DEF_HORIZONTAL };
    Vec3 vertical{ ::DEF_VERTICAL };
};

#endif // !CRT_SRC_CAMERA_CUH
