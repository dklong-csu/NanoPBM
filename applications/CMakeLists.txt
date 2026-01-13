cmake_minimum_required(VERSION 3.14)

message(STATUS "Configuring applications")

file(GLOB _apps "*.cpp")
foreach(_app ${_apps})
    string(REPLACE ".cpp" "" _appname ${_app})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" _appname ${_appname})
    message(STATUS "  ${_appname}")

    add_executable(
        ${_appname}
        ${_app}
    )
    target_compile_options(${_appname} PUBLIC -march=native)
    target_link_libraries(
        ${_appname}
        PRIVATE
        libNanoPBM
    )
endforeach()