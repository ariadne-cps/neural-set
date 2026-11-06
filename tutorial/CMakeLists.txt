cmake_minimum_required(VERSION 3.15)
project(TutorialNeuralSet VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(AriadneNeuralSet REQUIRED)

add_executable(tutorial_neural_set tutorial_neural_set.cpp)
target_link_libraries(tutorial_neural_set PRIVATE
    AriadneNeuralSet::ariadne-neural-set)
