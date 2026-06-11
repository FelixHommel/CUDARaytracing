#include "src/Camera.cuh"
#include "src/HitableList.cuh"
#include "src/IHitable.cuh"
#include "src/Ray.cuh"
#include "src/Sphere.cuh"
#include "src/Vec3.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fmt/base.h>

#include <cfloat>
#include <cstdlib>
#include <ctime>
#include <span>
#include <vector>

__host__ __device__ unsigned int calculatePixelIndex(unsigned int x, unsigned int y, unsigned int width);

#define CHECK_CUDA_ERROR(val) ::check_cuda((val), #val, __FILE__, __LINE__)

namespace
{

constexpr auto ERROR_EXIT_CODE{ 99 };

constexpr auto NX{ 1200u };                                   ///< Pixels on the X-Axis (width)
constexpr auto NY{ 600u };                                    ///< Pixels on the Y-Axis (height)
constexpr auto NUM_PIXELS{ NX * NY };                         ///< The amount of pixels in the framebuffer
constexpr auto FRAMEBUFFER_SIZE{ NUM_PIXELS * sizeof(Vec3) }; ///< Framebuffer size in bytes

constexpr auto TX{ 8 };                                        ///< Threads on the X-Axis
constexpr auto TY{ 8 };                                        ///< Threads on the Y-Axis
constexpr dim3 BLOCKS{ (::NX / ::TX) + 1, (::NY / ::TY) + 1 }; ///< The layout of the blocks for CUDA Kernels
constexpr dim3 THREADS{ ::TX, ::TY };                          ///< The layout of the threads for CUDA Kernels

constexpr auto SAMPLE_COUNT{ 100u };

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

__device__ constexpr auto MAX_COLOR_ITERATIONS{ 50u };
__device__ constexpr auto DEV_COLOR_FACTOR{ Vec3(0.5f, 0.7f, 1.f) };

/// \brief Calculate the color at a specific \ref Ray position.
///
/// \param r The ray that points to a position
/// \param world List of all objects in the world
/// \param localRandState The thread local \ref curandState
///
/// \returns Vec3 Color at that position
///
/// \note Only callable from a CUDA Kernel or other device functions.
__device__ Vec3 color(const Ray& r, IHitable** world, curandState* localRandState)
{
    Ray curRay{ r };
    Vec3 curAttenuation{ 1.f };

    for(int i{ 0 }; i < MAX_COLOR_ITERATIONS; ++i)
    {
        HitRecord rec;
        if((*world)->hit(curRay, 0.001f, FLT_MAX, rec))
        {
            Ray scattered;
            Vec3 attenuation;

            if(rec.pMaterial->scatter(curRay, rec, attenuation, scattered, localRandState))
            {
                curAttenuation *= attenuation;
                curRay = scattered;
            }
            else
                return Vec3{ 0.f };
        }
        else
        {
            const Vec3 dir{ unitVector(r.direction) };
            const float t{ 0.5f * (dir.y() + 1.f) };
            const Vec3 c{ (1.f - t) * Vec3(1.f) + t * DEV_COLOR_FACTOR };

            return curAttenuation * c;
        }
    }

    return Vec3{ 0.f };
}

/// \brief Kernel that produces the framebuffer.
///
/// \param pFramebuffer The framebuffer array
/// \param width The width of the framebuffer
/// \param height The height of the framebuffer
/// \param sampleCount The amount of samples taken per pixel
/// \param camera The \ref Camera that observes the scene
/// \param world List of all objects in the world
/// \param randState The \ref curandState to access the thread local random state
///
/// \note CUDA Kernel
__global__ void render(
    Vec3* pFramebuffer,
    int width,
    int height,
    unsigned int sampleCount,
    Camera** camera,
    IHitable** world,
    curandState* randState
)
{
    const auto i{ threadIdx.x + (blockIdx.x * blockDim.x) };
    const auto j{ threadIdx.y + (blockIdx.y * blockDim.y) };

    if(i >= width || j >= height)
        return;

    const auto pixelIndex{ calculatePixelIndex(i, j, width) };
    auto localRandState{ randState[pixelIndex] };

    auto c{ Vec3{ 0.f } };
    for(int s{ 0 }; s < sampleCount; ++s)
    {
        const auto u{ (static_cast<float>(i) + curand_uniform(&localRandState)) / static_cast<float>(width) };
        const auto v{ (static_cast<float>(j) + curand_uniform(&localRandState)) / static_cast<float>(height) };

        c += color((*camera)->getRay(u, v), world, &localRandState);
    }

    randState[pixelIndex] = localRandState;

    // NOTE: Simplified color correction
    c /= static_cast<float>(sampleCount);
    c[0] = std::sqrt(c[0]);
    c[1] = std::sqrt(c[1]);
    c[2] = std::sqrt(c[2]);

    pFramebuffer[pixelIndex] = c;
}

/// \brief Kernel to prepare the rendering state on the GPU.
///
/// \param width The width of the framebuffer
/// \param height The height of the framebuffer
/// \param randState The \ref curandState to access the thread local random state
///
/// \note CUDA Kernel
__global__ void renderInit(int width, int height, curandState* randState)
{
    const auto i{ threadIdx.x + (blockIdx.x * blockDim.x) };
    const auto j{ threadIdx.y + (blockIdx.y * blockDim.y) };

    if(i >= width || j >= height)
        return;

    const auto pixelIndex{ calculatePixelIndex(i, j, width) };

    curand_init(0xC0FFEE, pixelIndex, 0, &randState[pixelIndex]);
}

/// \brief Generate a random number.
///
/// \param randState The \ref curandState to access the thread local random state
///
/// \returns the generated random number
__device__ inline float randNum(curandState* randState)
{
    return curand_uniform(randState);
}

/// \brief Generate two random numbers and multiply them.
///
/// \param randState The \ref curandState to access the thread local random state
///
/// \returns the generated random number
__device__ inline float randNumSq(curandState* randState)
{
    return randNum(randState) * randNum(randState);
}

constexpr auto OBJECTS_IN_SCENE{ (22u * 22u) + 1u + 3u };
__device__ constexpr auto DEV_OBJECTS_IN_SCENE{ OBJECTS_IN_SCENE };

/// \brief Kernel to create the objects that are in the world on the GPU.
///
/// \param list The objects that are in the world
/// \param world The container that contains the objects in \p list
/// \param camera The \ref Camera that observes the scene
/// \param width The width of the framebuffer
/// \param height The height of the framebuffer
/// \param randState The \ref curandState to access the thread local random state
///
/// \note CUDA Kernel
__global__ void createWorld(
    IHitable** list, IHitable** world, Camera** camera, int width, int height, curandState* randState
)
{
    if(!(threadIdx.x == 0 && blockIdx.x == 0))
        return;

    auto localRandState{ *randState };

    list[0] = new Sphere{
        Vec3{ 0.f, -1000.f, -1.f },
        1000.f, new Lambertian{ Vec3{ 0.5f } }
    };

    int i{ 1 };
    for(int a{ -11 }; a < 11; ++a)
    {
        for(int b{ -11 }; b < 11; ++b)
        {
            const float chooseMat{ randNum(&localRandState) };
            const Vec3 center{ a + randNum(&localRandState), 0.2f, b + randNum(&localRandState) };

            if(chooseMat < 0.8f)
                list[i++] = new Sphere{ center,
                                        0.2f,
                                        new Lambertian{ Vec3{ randNumSq(&localRandState),
                                                              randNumSq(&localRandState),
                                                              randNumSq(&localRandState) } } };
            else if(chooseMat < 0.95f)
            {
                const auto metalColorFn{ [&localRandState] { return 0.5f * (1.f + randNum(&localRandState)); } };

                list[i++] = new Sphere{
                    center,
                    0.2f,
                    new Metal{ Vec3{ metalColorFn(), metalColorFn(), metalColorFn() }, 0.5f * randNum(&localRandState) }
                };
            }
            else
                list[i++] = new Sphere{ center, 0.2f, new Dielectric{ 1.5f } };
        }
    }

    list[i++] = new Sphere{
        Vec3{ 0.f, 1.f, 0.f },
        1.f, new Dielectric{ 1.5f }
    };
    list[i++] = new Sphere{
        Vec3{ -4.f, 1.f, 0.f },
        1.f, new Lambertian{ Vec3{ 0.4f, 0.2f, 0.1f } }
    };
    list[i++] = new Sphere{
        Vec3{ 4.f, 1.f, 0.f },
        1.f, new Metal{ Vec3{ 0.7f, 0.6f, 0.5f }, 0.f }
    };

    *randState = localRandState;

    *world = new HitableList(list, DEV_OBJECTS_IN_SCENE);

    *camera = new Camera(
        Vec3(13.f, 2.f, 3.f),
        Vec3(0.f, 0.f, 0.f),
        Vec3(0.f, 1.f, 0.f),
        30.f,
        (static_cast<float>(width) / static_cast<float>(height))
    );
}

/// \brief Kernel to destroy the objects that are in the world on the GPU.
///
/// \param list The objects that are in the world
/// \param world The container that contains the objects in \p list
/// \param camera The \ref Camera that observes the scene
///
/// \note CUDA Kernel
__global__ void freeWorld(IHitable** list, IHitable** world, Camera** camera)
{
    for(int i{ 0 }; i < DEV_OBJECTS_IN_SCENE; ++i)
    {
        delete static_cast<Sphere*>(list[i])->pMaterial;
        delete list[i];
    }
    delete *world;
    delete *camera;
}

// NOLINTEND

int main()
{
    Vec3* d_framebuffer{ nullptr };
    CHECK_CUDA_ERROR(cudaMalloc(&d_framebuffer, ::FRAMEBUFFER_SIZE));
    CHECK_CUDA_ERROR(cudaGetLastError());

    curandState* d_randState{ nullptr };
    CHECK_CUDA_ERROR(cudaMalloc(&d_randState, NUM_PIXELS * sizeof(curandState)));

    renderInit<<<BLOCKS, THREADS>>>(::NX, ::NY, d_randState);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    IHitable** d_list{ nullptr };
    CHECK_CUDA_ERROR(cudaMalloc(&d_list, OBJECTS_IN_SCENE * sizeof(IHitable*)));
    IHitable** d_world{ nullptr };
    CHECK_CUDA_ERROR(cudaMalloc(&d_world, sizeof(IHitable*)));
    Camera** d_camera{ nullptr };
    CHECK_CUDA_ERROR(cudaMalloc(&d_camera, sizeof(Camera*)));
    createWorld<<<1, 1>>>(d_list, d_world, d_camera, ::NX, ::NY, d_randState);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    render<<<BLOCKS, THREADS>>>(d_framebuffer, ::NX, ::NY, ::SAMPLE_COUNT, d_camera, d_world, d_randState);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<Vec3> framebuffer(::NUM_PIXELS);
    CHECK_CUDA_ERROR(cudaMemcpy(framebuffer.data(), d_framebuffer, ::FRAMEBUFFER_SIZE, cudaMemcpyDeviceToHost));
    ::exportImage({ framebuffer.data(), ::NUM_PIXELS });

    freeWorld<<<1, 1>>>(d_list, d_world, d_camera);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void*>(d_camera)));
    CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void*>(d_world)));
    CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void*>(d_list)));
    CHECK_CUDA_ERROR(cudaFree(d_randState));
    CHECK_CUDA_ERROR(cudaFree(d_framebuffer));

    cudaDeviceReset();

    return 0;
}
