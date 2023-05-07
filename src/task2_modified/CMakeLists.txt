cmake_minimum_required(VERSION 3.0.2)
project(task2_modified)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  std_msgs
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})
