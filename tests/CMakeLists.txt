cmake_minimum_required(VERSION 3.14)

message(STATUS "Configuring tests...")

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/ut/include")
# include_directories("${PROJECT_SOURCE_DIR}/include")

file(GLOB _testfiles "*.cpp")
foreach(_testfile ${_testfiles})
  # ---- Name of the test ----
  string(REPLACE ".cpp" "" _testname ${_testfile})
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" _testname ${_testname})
  message(STATUS "  ${_testname}")

  # ---- Create the test executable ----
  add_executable(${_testname} ${_testfile})
  target_link_libraries(${_testname} PRIVATE libNanoPBM)

  # ---- Tell cmake how to run a test ----
  add_test(NAME ${_testname} COMMAND ${_testname} )

endforeach()