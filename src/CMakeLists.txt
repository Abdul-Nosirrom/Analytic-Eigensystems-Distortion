cmake_minimum_required(VERSION 3.16)
project(Deformation)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
option(LIBIGL_GLFW "Build target igl::glfw" ON)
option(LIBIGL_IMGUI "Build target igl::imgui" ON)
option(LIBIGL_OPENGL_GLFW_IMGUI "Build target igl::imgui" ON)
option(LIBIGL_OPENGL "Build target igl::opengl" ON)
include(libigl)


# Add your project files
file(GLOB SRC_FILES *.cpp *.h 
	Utilities/*.h Utilities/*.cpp
	SimulatorCore/*.h SimulatorCore/*.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw)
target_link_libraries(${PROJECT_NAME} PUBLIC igl::opengl)
target_link_libraries(${PROJECT_NAME} PUBLIC igl::imgui)

