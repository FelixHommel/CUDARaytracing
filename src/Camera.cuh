#ifndef CRT_SRC_CAMERA_CUH
#define CRT_SRC_CAMERA_CUH

#include "src/Ray.cuh"
#include "src/Utility.cuh"
#include "src/Vec3.cuh"

#include <cmath>
#include <cuda_runtime_api.h>

/// \brief The Camera is where the \ref Ray are cast from.
///
/// \author Felix Hommel
/// \date 6/6/2026
class Camera // NOLINT
{
public:
    /// \brief Construct a new \ref Camera with a custom position
    ///
    /// \param lookFrom Where the \ref Camera is looking from
    /// \param lookAt Where the \ref Camera is looking at
    /// \param up Which axis is pointing up
    /// \param verticalFov The vertical field of view of the \ref Camera
    /// \param aspectRatio The aspect ratio of the output image
    /// \param aperture The size of the opening in the lens of the \ref Camera
    /// \param focusDistance The distance over which the \ref Camera keeps focus
    __device__ Camera(
        Vec3 lookFrom, Vec3 lookAt, Vec3 up, float verticalFov, float aspectRatio, float aperture, float focusDistance
    )
        : m_origin{ lookFrom }
        , m_w{ unitVector(lookFrom - lookAt) }
        , m_u{ unitVector(cross(up, m_w)) }
        , m_v{ cross(m_w, m_u) }
        , m_lensRadius{ aperture / 2.f }
    {
        const float theta{ verticalFov * 3.14159265358979323846f / 180.f };
        const float halfHeight{ std::tan(theta / 2.f) };
        const float halfWidth{ aspectRatio * halfHeight };

        m_lowerLeftCorner
            = m_origin - halfWidth * focusDistance * m_u - halfHeight * focusDistance * m_v - focusDistance * m_w;
        m_horizontal = 2 * halfWidth * focusDistance * m_u;
        m_vertical = 2 * halfHeight * focusDistance * m_v;
    }
    __device__ ~Camera(){}; // NOLINT(modernize-use-equals-default): nvcc warning: non-virtual __device__ = default methods

    /// \brief Get a Ray that is cast towards a specific point (s, t)
    ///
    /// \param s Position on the X-Axis
    /// \param t Position on the Y-Axis
    /// \param randState The \ref curandState to access the thread local random state
    ///
    /// \returns \ref Ray from the camera center to (u, v)
    __device__ Ray getRay(float s, float t, curandStatePhilox4_32_10_t* randState) const noexcept
    {
        const Vec3 rd{ m_lensRadius * device::randomInUnitDisk(randState) };
        const Vec3 offset{ m_u * rd.x() + m_v * rd.y() };

        return { m_origin + offset, m_lowerLeftCorner + (s * m_horizontal) + (t * m_vertical) - m_origin - offset };
    }

private:
    Vec3 m_origin;
    Vec3 m_lowerLeftCorner;
    Vec3 m_horizontal;
    Vec3 m_vertical;
    Vec3 m_w;
    Vec3 m_u;
    Vec3 m_v;
    float m_lensRadius;
};

#endif // !CRT_SRC_CAMERA_CUH
