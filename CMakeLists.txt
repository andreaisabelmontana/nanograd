cmake_minimum_required(VERSION 3.16)
project(nanograd CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(nanograd src/main.cpp)
target_include_directories(nanograd PRIVATE include)

enable_testing()
add_executable(nanograd_tests tests/test_nanograd.cpp)
target_include_directories(nanograd_tests PRIVATE include)
add_test(NAME nanograd_tests COMMAND nanograd_tests)
