set(NAME "CRTBenchmark")

add_executable(${NAME} ./main.cu)

target_compile_features(${NAME} PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(${NAME} PRIVATE fmt::fmt CRTLib)
