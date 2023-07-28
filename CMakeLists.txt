cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(lab LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find the required packages
# find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(LAPACK REQUIRED)

# Set the output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Set the include directories
include_directories(include)

# Set compiler-specific optimization flags
add_compile_options(
    -O3
    -march=native
    -Wall
    -Wextra
    -Wno-unused-variable
    -Wno-unused-parameter
    -Wno-sign-compare
)

# Add the main executable if not in Test mode
if(NOT CMAKE_BUILD_TYPE MATCHES Test)
    add_executable(run 
        # src/lapackage.cpp
        # src/lobpcg.cpp
        src/main.cpp
        )

    # Link the required libraries
    target_link_libraries(run 
        PUBLIC 
            # Kokkos::kokkos 
            Kokkos::kokkoskernels
            LAPACK::LAPACK
        )
    
    # Set Debug/Release mode definitions
    target_compile_definitions(run 
        PRIVATE 
            "$<$<CONFIG:Debug>:DEBUG_MODE>"
            "$<$<CONFIG:Release>:RELEASE_MODE>"
    )
endif()

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

    target_link_libraries(runUnitTest 
        gtest_main 
        # Kokkos::kokkos 
        Kokkos::kokkoskernels 
        LAPACK::LAPACK
    )
    
    # Optionally, you can define a custom test target to run the tests easily
    add_custom_target(run-tests
        COMMAND runUnitTest
        DEPENDS runUnitTest
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# cmake .. -DCMAKE_BUILD_TYPE=Release
# cmake .. -DCMAKE_BUILD_TYPE=Debug
# cmake .. -DCMAKE_BUILD_TYPE=Test