cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(lab LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find the required packages
# find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)

# Set the output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Set the include directories
include_directories(include)

# set the complie flags for 03 optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")

# # Add the library target
# add_library(lapackage SHARED src/lapackage.cpp)
# target_link_libraries(lapackage 
#     PUBLIC 
#         Kokkos::kokkos 
#         Kokkos::kokkoskernels 
#         LAPACK)

# Add the main executable
add_executable(run 
    # src/lapackage.cpp
    # src/lobpcg.cpp
    src/main.cpp
    )

# Link the required libraries
target_link_libraries(run 
    PUBLIC 
        MPI::MPI_CXX 
        OpenMP::OpenMP_CXX 
        Kokkos::kokkos 
        Kokkos::kokkoskernels
        # lapackage
        # LAPACK
    )
# target_link_libraries(run PUBLIC LAPACK::LAPACK)

# ---- Different Modes ----

# Debug mode
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(run PRIVATE DEBUG_MODE)
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_definitions(run PRIVATE RELEASE_MODE)
endif()

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set warning flags
add_compile_options(
  -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare
)

# ---- Google Test Integration ----

# Download and configure Google Test if in test mode
if(CMAKE_BUILD_TYPE MATCHES Test)
    include(FetchContent)

    # Fetch and configure Google Test
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.12.0
    )
    FetchContent_MakeAvailable(googletest)
    
    # Add the test executable and link with Google Test
    add_executable(runUnitTest 
        # test/test_lapackage.cpp
        test/test_lobpcg.cpp
        )

    target_link_libraries(runUnitTest gtest_main MPI::MPI_CXX OpenMP::OpenMP_CXX Kokkos::kokkos Kokkos::kokkoskernels LAPACK)
    
    # Optionally, you can define a custom test target to run the tests easily
    add_custom_target(run-tests
        COMMAND runUnitTest
        DEPENDS runUnitTest
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()