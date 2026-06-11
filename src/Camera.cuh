#ifndef CRT_SRC_CAMERA_CUH
#define CRT_SRC_CAMERA_CUH

#include "src/Ray.cuh"
#include "src/Vec3.cuh"

#include <cmath>
#include <cuda_runtime_api.h>

/// \brief The Camera is where the \ref Ray are cast from.
///
/// \author Felix Hommel
/// \date 6/6/2026
struct Camera // NOLINT
{
    /// \brief Construct a new \ref Camrea with a custom position
    ///
    /// \param lookFrom Where the \ref Camera is looking from
    /// \param lookAt Where the \ref Camera is looking at
    /// \param up Which axis is pointing up
    /// \param verticalFov The vertical field of view of the \ref Camera
    /// \param aspectRatio The aspect ratio of the output image
    __device__ Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 up, float verticalFov, float aspectRatio) : origin{ lookFrom }
    {
        const float theta{ verticalFov * 3.14159265358979323846f / 180.f };
        const float halfHeight{ std::tan(theta / 2.f) };
        const float halfWidth{ aspectRatio * halfHeight };

        const Vec3 w{ unitVector(lookFrom - lookAt) };
        const Vec3 u{ unitVector(cross(up, w)) };
        const Vec3 v{ cross(w, u) };

        lowerLeftCorner = origin - halfWidth * u - halfHeight * v - w;
        horizontal = 2 * halfWidth * u;
        vertical = 2 * halfHeight * v;
    }
    __device__ ~Camera(){}; // NOLINT(modernize-use-equals-default): nvcc warning: non-virtual __device__ = default methods

    /// \brief Get a Ray that is cast towards a specific point (u, v)
    ///
    /// \param u Position on the X-Axis
    /// \param v Position on the Y-Axis
    ///
    /// \returns \ref Ray from the camera center to (u, v)
    __device__ Ray getRay(float u, float v) const noexcept
    {
        return { origin, lowerLeftCorner + (u * horizontal) + (v * vertical) - origin };
    }

    Vec3 origin;
    Vec3 lowerLeftCorner;
    Vec3 horizontal;
    Vec3 vertical;
};

#endif // !CRT_SRC_CAMERA_CUH
