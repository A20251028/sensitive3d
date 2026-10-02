cmake_minimum_required(VERSION 3.12)
project(osgb_bridge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenSceneGraph 3.4 REQUIRED COMPONENTS osgDB)

add_executable(osgb_bridge osgb_bridge.cpp)
target_include_directories(osgb_bridge PRIVATE ${OPENSCENEGRAPH_INCLUDE_DIRS})
target_link_libraries(osgb_bridge PRIVATE ${OPENSCENEGRAPH_LIBRARIES})
install(TARGETS osgb_bridge RUNTIME DESTINATION bin)
