cmake_minimum_required(VERSION 3.8)
project(bipedal_robot_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Only require ament_cmake for resource-only package
find_package(ament_cmake REQUIRED)

# Install directories
install(DIRECTORY
  launch
  urdf
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install Python executables
install(PROGRAMS
  scripts/robot_controller.py/robot_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
