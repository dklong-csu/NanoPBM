cmake_minimum_required(VERSION 3.14)

project(
    NanoPBM
    LANGUAGES CXX C
)

message(STATUS "Installing NanoPBM" )

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ---- Prevent in-source builds ----
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()


# ---- Install YAML Input File Parser ----
include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG master # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_MakeAvailable(yaml-cpp)




# ---- Include header files from this library ----
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- Header only library ----
add_library(libNanoPBM INTERFACE)
target_include_directories(libNanoPBM 
INTERFACE
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/nanopbm>
$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}
)

find_package(SUNDIALS
NAMES SUNDIALS
REQUIRED
)

find_package(Ginkgo REQUIRED)

find_package(OpenMP REQUIRED COMPONENTS C CXX)

find_package(HDF5 REQUIRED COMPONENTS C CXX)
if (OpenMP_CXX_FOUND)
    message(STATUS Found OpenMP)
    target_link_libraries(libNanoPBM
    INTERFACE
    OpenMP::OpenMP_CXX 
    SUNDIALS::core
    SUNDIALS::cvode
    SUNDIALS::nvecserial
    SUNDIALS::nvecopenmp
    SUNDIALS::sunmatrixdense
    SUNDIALS::sunmatrixsparse
    SUNDIALS::sunlinsoldense
    SUNDIALS::sunlinsollapackdense
    HDF5::HDF5 
    yaml-cpp::yaml-cpp
    Ginkgo::ginkgo)
else ()
target_link_libraries(libNanoPBM
    INTERFACE
    SUNDIALS::core
    SUNDIALS::cvode
    SUNDIALS::nvecserial
    SUNDIALS::sunmatrixdense
    SUNDIALS::sunmatrixsparse
    SUNDIALS::sunlinsoldense
    SUNDIALS::sunlinsollapackdense
    HDF5::HDF5 
    yaml-cpp::yaml-cpp
    Ginkgo::ginkgo)
endif()

add_subdirectory(applications)

# ---- Test suite ----
enable_testing()
add_subdirectory(tests)

# ---- Documentation ----
# add_subdirectory(documentation)

# ---- Make compile commands file for clangd ----
message(STATUS "Creating compile_commands.json for the clangd language server.")
execute_process(
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_SOURCE_DIR}/compile_commands.json)