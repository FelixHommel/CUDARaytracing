#ifndef CRT_SRC_ERROR_CUH
#define CRT_SRC_ERROR_CUH

#include <fmt/base.h>

namespace
{

constexpr auto ERROR_EXIT_CODE{ 1 };

} // namespace

namespace errorChecking
{

/// \brief Check if a CUDA call produced an error.
///
/// \param result The return value of a CUDA call
/// \param func Name of the function that was called
/// \param file Name of the file where the call was made
/// \param line Linenumber in which the call was made
void checkCuda(cudaError_t result, char const* func, char const* file, int line)
{
    if(result != cudaSuccess)
    {
        fmt::println("CUDA error = {} at {}: {} '{}'", static_cast<unsigned int>(result), file, line, func);

        cudaDeviceReset();
        std::exit(::ERROR_EXIT_CODE);
    }
}

} // namespace errorChecking

#define CHECK_CUDA_ERROR(val) errorChecking::checkCuda((val), #val, __FILE__, __LINE__)

#endif // !CRT_SRC_ERROR_CUH
