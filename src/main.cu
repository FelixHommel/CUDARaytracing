#include "src/Ray.cuh"
#include "src/Vec3.cuh"

#include <cuda_runtime.h>
#include <fmt/base.h>

#include <cstdlib>
#include <ctime>
#include <span>

__host__ __device__ unsigned int calculatePixelIndex(unsigned int x, unsigned int y, unsigned int width);

#define CHECK_CUDA_ERROR(val) ::check_cuda((val), #val, __FILE__, __LINE__)

namespace
{

constexpr auto ERROR_EXIT_CODE{ 99 };

constexpr auto NX{ 1200u }; ///< Pixels on the X-Axis (width)
constexpr auto NY{ 600u };  ///< Pixels on the Y-Axis (height)
constexpr auto NUM_PIXELS{ NX * NY };
constexpr auto FRAMEBUFFER_SIZE{ NUM_PIXELS * sizeof(Vec3) }; ///< Size in bytes

constexpr auto TX{ 8 }; ///< Threads on the X-Axis
constexpr auto TY{ 8 }; ///< Threads on the Y-Axis

constexpr auto LOWER_LEFT_CORNER{ Vec3(-2.0, -1.0, -1.0) };
constexpr auto HORIZONTAL{ Vec3(2.0 + 2.0, 0.0, 0.0) };
constexpr auto VERTICAL{ Vec3(0.0, 2.0, 0.0) };
constexpr auto ORIGIN{ Vec3(0.0, 0.0, 0.0) };

constexpr auto COLOR_MAX{ 255.99 };

/// \brief Check if a CUDA call produced an error.
///
/// \param result The return value of a CUDA call
/// \param func Name of the function that was called
/// \param file Name of the file where the call was made
/// \param line Linenumber in which the call was made
void check_cuda(cudaError_t result, char const* func, char const* file, int line)
{
    if(result != cudaSuccess)
    {
        fmt::println("CUDA error = {} at {}: {} '{}'", static_cast<unsigned int>(result), file, line, func);

        cudaDeviceReset();
        std::exit(::ERROR_EXIT_CODE);
    }
}

/// \brief Export framebuffer as PPM image.
///
/// \param framebuffer span of floats that represent the color values
void exportImage(std::span<const Vec3> framebuffer)
{
    fmt::println("P3\n{} {}\n255", NX, NY);

    for(int j{ NY - 1 }; j >= 0; j--)
    {
        for(int i{ 0 }; i < NX; i++)
        {
            const std::size_t pixelIndex{ calculatePixelIndex(i, j, NX) };

            fmt::println(
                "{} {} {}",
                static_cast<int>(::COLOR_MAX * framebuffer[pixelIndex].x()),
                static_cast<int>(::COLOR_MAX * framebuffer[pixelIndex].y()),
                static_cast<int>(::COLOR_MAX * framebuffer[pixelIndex].z())
            );
        }
    }
}

} // namespace

/// \brief Convert a (x, y) position to the index of a pixel.
///
/// \param x Component on the X-Axis
/// \param y Component on the Y-Axis
/// \param width The width of the framebuffer
///
/// \note Available on \p host and \p device
__host__ __device__ unsigned int calculatePixelIndex(unsigned int x, unsigned int y, unsigned int width)
{
    return (y * width) + x;
}

// NOLINTBEGIN: CUDA kernels follow C-style syntax, which the clang-tidy settings do not like

__device__ constexpr auto DEV_COLOR_FACTOR{ Vec3(0.5f, 0.7f, 1.f) };

/// \brief Generate the background of the rendered image.
///
/// \param r The ray that points to a position
///
/// \returns Vec3 Color at that position
__device__ Vec3 color(const Ray& r)
{
    const Vec3 dir{ unitVector(r.direction) };
    const float t{ 0.5f * (dir.y() + 1.f) };

    return (1.f - t) * Vec3(1.f) + t * DEV_COLOR_FACTOR;
}

/// \brief Kernel that produces the framebuffer.
///
/// \param pFramebuffer The framebuffer array
/// \param width The width of the framebuffer
/// \param height The height of the framebuffer
/// \param lowerLeftCorner Specify what the bottom left corner of the "screen" is
/// \param horizontal Size of the "screen" on the X-Axis
/// \param vertical Size of the "screen" on the Y-Axis
/// \param origin The center of the "screen"
__global__ void render(
    Vec3* pFramebuffer, int width, int height, Vec3 lowerLeftCorner, Vec3 horizontal, Vec3 vertical, Vec3 origin
)
{
    const auto i{ threadIdx.x + (blockIdx.x * blockDim.x) };
    const auto j{ threadIdx.y + (blockIdx.y * blockDim.y) };

    if(i >= width || j >= height)
        return;

    const auto pixelIndex{ calculatePixelIndex(i, j, width) };

    const auto u{ static_cast<float>(i) / static_cast<float>(width) };
    const auto v{ static_cast<float>(j) / static_cast<float>(height) };
    const Ray r{ origin, lowerLeftCorner + (u * horizontal) + (v * vertical) };

    pFramebuffer[pixelIndex] = color(r);
}

// NOLINTEND

int main()
{
    Vec3* framebuffer{ nullptr };
    CHECK_CUDA_ERROR(cudaMallocManaged(&framebuffer, ::FRAMEBUFFER_SIZE));
    CHECK_CUDA_ERROR(cudaGetLastError());

    constexpr dim3 blocks((::NX / ::TX) + 1, (::NY / ::TY) + 1);
    constexpr dim3 threads(::TX, ::TY);

    render<<<blocks, threads>>>(framebuffer, ::NX, ::NY, ::LOWER_LEFT_CORNER, ::HORIZONTAL, ::VERTICAL, ::ORIGIN);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    ::exportImage({ framebuffer, ::FRAMEBUFFER_SIZE });

    CHECK_CUDA_ERROR(cudaFree(framebuffer));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return 0;
}
