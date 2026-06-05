#include "Vec3.cuh"

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

__device__ constexpr auto DEV_FB_BLUE{ 0.2f }; ///< Value for the framebuffers blue component

/// \brief Kernel that produces the framebuffer.
///
/// \param pFramebuffer The framebuffer array
/// \param width The width of the framebuffer
/// \param height The height of the framebuffer
__global__ void render(Vec3* pFramebuffer, int width, int height)
{
    const auto i{ threadIdx.x + (blockIdx.x * blockDim.x) };
    const auto j{ threadIdx.y + (blockIdx.y * blockDim.y) };

    if(i >= width || j >= height)
        return;

    const auto pixelIndex{ calculatePixelIndex(i, j, width) };

    pFramebuffer[pixelIndex] = Vec3{
        static_cast<float>(i) / static_cast<float>(width),
        static_cast<float>(j) / static_cast<float>(height),
        DEV_FB_BLUE,
    };
}

// NOLINTEND

int main()
{
    Vec3* framebuffer{ nullptr };
    CHECK_CUDA_ERROR(cudaMallocManaged(&framebuffer, FRAMEBUFFER_SIZE));
    CHECK_CUDA_ERROR(cudaGetLastError());

    constexpr dim3 blocks((NX / TX) + 1, (NY / TY) + 1);
    constexpr dim3 threads(TX, TY);

    render<<<blocks, threads>>>(framebuffer, NX, NY);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    ::exportImage({ framebuffer, FRAMEBUFFER_SIZE });

    CHECK_CUDA_ERROR(cudaFree(framebuffer));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return 0;
}
