cmake_minimum_required(VERSION 3.8)
project(multi_ugv_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# ── Static Library ─────────────────────────────────
file(GLOB_RECURSE CONTOURING_ACADOS_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/acados/generated/contouring_solver/*.c"
)
list(FILTER CONTOURING_ACADOS_SOURCES EXCLUDE REGEX ".*/main(_sim)?(_ocp)?_contouring_unicycle[^/]*\\.c$")

add_library(${PROJECT_NAME} SHARED
  src/mpc_solver.cpp
  src/acados_contouring_solver.cpp
  src/constraint_builder.cpp
  ${CONTOURING_ACADOS_SOURCES}
  src/controller.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/acados/generated>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/acados/generated/contouring_solver>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/acados/generated/contouring_solver/contouring_unicycle_model>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/acados/generated/contouring_solver/contouring_unicycle_constraints>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(${PROJECT_NAME} PRIVATE
  /home/lu/.local/share/acados/include
  /home/lu/.local/share/acados/include/acados
  /home/lu/.local/share/acados/include/blasfeo/include
  /home/lu/.local/share/acados/include/hpipm/include
)
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
)
target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  /home/lu/.local/share/acados/lib/libacados.so
  /home/lu/.local/share/acados/lib/libhpipm.so
  /home/lu/.local/share/acados/lib/libblasfeo.so
)

# ── Executable (Node) ──────────────────────────────
add_executable(ugv_planner_node_exe src/ugv_planner_node.cpp)
target_compile_features(ugv_planner_node_exe PUBLIC cxx_std_17)
target_include_directories(ugv_planner_node_exe PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(ugv_planner_node_exe
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
)
target_link_libraries(ugv_planner_node_exe
  ${PROJECT_NAME}
  Eigen3::Eigen
)

install(TARGETS ${PROJECT_NAME} ugv_planner_node_exe
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  multi_ugv_planner/ugv_vprm_node.py
  multi_ugv_planner/vprm.py
  multi_ugv_planner/ugv_interfaces.py
  multi_ugv_planner/fake_odom.py
  multi_ugv_planner/fake_scan.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
