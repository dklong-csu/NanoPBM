cmake_minimum_required(VERSION 3.14)

project(
    NanoPBM
    LANGUAGES CXX
)

message(STATUS "Installing NanoPBM" )

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ---- Prevent in-source builds ----
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()


# ---- Install SUNDIALS ----



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

find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message(STATUS Found OpenMP)
    target_link_libraries(libNanoPBM
    INTERFACE
    OpenMP::OpenMP_CXX 
    SUNDIALS::core
    SUNDIALS::cvode
    SUNDIALS::nvecserial
    # SUNDIALS::nvecopenmp
    SUNDIALS::sunmatrixdense
    SUNDIALS::sunmatrixsparse
    SUNDIALS::sunlinsoldense
    SUNDIALS::sunlinsollapackdense)
else ()
target_link_libraries(libNanoPBM
    INTERFACE
    SUNDIALS::core
    SUNDIALS::cvode
    SUNDIALS::nvecserial
    SUNDIALS::sunmatrixdense
    SUNDIALS::sunmatrixsparse
    SUNDIALS::sunlinsoldense
    SUNDIALS::sunlinsollapackdense)
endif()

add_subdirectory(applications)

# ---- Make compile commands file for clangd ----
message(STATUS "Creating compile_commands.json for the clangd language server.")
execute_process(
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_SOURCE_DIR}/compile_commands.json)