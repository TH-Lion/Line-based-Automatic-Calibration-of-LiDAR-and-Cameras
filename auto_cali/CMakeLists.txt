# http://ros.org/doc/groovy/api/catkin/html/user_guide/supposed.html
cmake_minimum_required(VERSION 2.8.3)
project(auto_calibration)
set(CMAKE_BUILD_TYPE Release)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# Load catkin and all dependencies required for this package
find_package(catkin REQUIRED COMPONENTS roscpp camera_info_manager dynamic_reconfigure image_transport nodelet sensor_msgs)
find_package(PCL REQUIRED) #配置PCL库
find_package( OpenCV REQUIRED)
set(CMAKE_C_STANDARD 99)
 
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(PkgConfig)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp>=0.5)

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules ) 

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
set(THIRD_PARTY_LIBS
${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
    
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension




        ${CSPARSE_LIBRARY}
        )

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  tf
  jsk_recognition_msgs
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    camera_info_manager
    dynamic_reconfigure
    image_transport
    nodelet
    sensor_msgs
    LIBRARIES auto_calibration_lib auto_calibration_lib
    jsk_recognition_msgs
  )

#message(STATUS "OpenCV library status:")
#message(STATUS "    version: ${OpenCV_VERSION}")
#message(STATUS "    libraries: ${OpenCV_LIBS}")
#message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(
  include 
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR} 
  ${catkin_INCLUDE_DIRS} 
  ${YAMLCPP_INCLUDE_DIRS}
  "/usr/include/eigen3/"
  #/usr/include/opencv2
 
  )
  
link_directories(${catkin_LINK_DIRS})

find_package(Boost REQUIRED COMPONENTS thread)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(auto_calib_node src/auto_calib.cpp src/auto_calib_node.cpp)
#add_executable(lsd_node src/LSD.cpp)

target_link_libraries(auto_calib_node ${YAMLCPP_LIBRARIES} ${PCL_LIBRARIES}  ${OpenCV_LIBRARIES}
                      ${Boost_LIBRARIES} ${catkin_LIBRARIES}  g2o_core g2o_stuff Sophus::Sophus
                      ${OpenCV_LIBS} /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so
                      )
#target_link_libraries(lsd_node ${YAMLCPP_LIBRARIES} ${OpenCV_LIBRARIES}
#                      ${Boost_LIBRARIES} ${catkin_LIBRARIES}
#                      ${OpenCV_LIBS} ${PCL_LIBRARIES} /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so
#                      )


install(TARGETS auto_calib_node 
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
#install(TARGETS lsd_node 
#  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#  )
