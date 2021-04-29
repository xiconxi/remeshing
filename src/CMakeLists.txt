file(GLOB SOURCES ./*.cpp)
file(GLOB HEADERS ./*.h)


include_directories(${PROJECT_SOURCE_DIR}/external/triangle/)
include_directories(${PROJECT_SOURCE_DIR}/external/)
include_directories(${PROJECT_SOURCE_DIR}/external/voronoi/src)


add_executable(remeshing ${SOURCES} ${HEADERS}
        ${PROJECT_SOURCE_DIR}/external/triangulation.cpp )
target_link_libraries(remeshing pmp_vis triangle)

