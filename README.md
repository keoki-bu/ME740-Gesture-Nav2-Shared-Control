# ME740 Gesture-Driven Shared-Control Navigation

This project implements a ROS 2 framework for gesture-driven mobile robot navigation using TurtleBot3 in Gazebo simulation.

## Overview

The project extends a prior gesture-recognition pipeline into a ROS 2 mobile robot navigation system. It supports three staged modes:

1. Manual gesture teleoperation
2. Auto-only gesture-triggered Nav2 goal navigation
3. Auto + Shared control with command fusion

In the final Auto + Shared mode, Nav2 provides the nominal navigation command, while gesture input provides correction, speed limiting, and emergency stop behaviors.

## Main Features

- Real-time gesture recognition interface based on MediaPipe, Rules + SVM, voting, and hysteresis
- Manual gesture teleoperation baseline
- Auto-only gesture-triggered Nav2 goal navigation
- Auto + Shared control with command fusion
- Gesture-based correction, speed limiting, and emergency stop
- TurtleBot3 Gazebo simulation with Nav2 and saved map
- CSV logs for Auto-only and Auto + Shared experiments

## Custom ROS 2 Packages

### gesture_tb3_teleop

Gesture recognition node. It publishes:

- /gesture/stable
- /gesture/mode
- /gesture/estop
- /cmd_vel_gesture when remapped from /cmd_vel

### gesture_nav2_shared

Shared-control package. It contains:

- gesture_auto_only: gesture-triggered Nav2 goal selection
- cmd_vel_fusion: Auto + Shared velocity fusion node
- nav2_cmd_vel_nav.launch.py: Nav2 launch wrapper that remaps Nav2 velocity output to /cmd_vel_nav

## Repository Structure

ME740_Gesture_Nav2_Shared_Control/
- README.md
- .gitignore
- src/
  - gesture_tb3_teleop/
  - gesture_nav2_shared/
- maps/
  - tb3_map.yaml
  - tb3_map.pgm or tb3_map.png
- logs/
  - auto_only_goal_log_final.csv
  - shared_fusion_log_final.csv
- figures/
  - system_architecture.png
  - fusion_logic.png
  - topic_graph.png
  - shared_events.png
  - auto_shared_demo.png

## Build Instructions

Copy the custom packages into a ROS 2 workspace and build:

    mkdir -p ~/turtlebot3_ws/src
    cp -r src/gesture_tb3_teleop ~/turtlebot3_ws/src/
    cp -r src/gesture_nav2_shared ~/turtlebot3_ws/src/
    cd ~/turtlebot3_ws
    colcon build --symlink-install
    source install/setup.bash

## Example Auto + Shared Run Sequence

Use four terminals.

### Terminal A: Start Gazebo world

    source ~/.bashrc
    ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

### Terminal B: Start Nav2 with velocity output remapped to /cmd_vel_nav

    source ~/.bashrc
    ros2 launch gesture_nav2_shared nav2_cmd_vel_nav.launch.py map:=$HOME/maps/tb3_map.yaml use_sim_time:=True

### Terminal C: Start gesture recognition node

    source ~/.bashrc
    ros2 run gesture_tb3_teleop gesture_cmd_vel --ros-args -r /cmd_vel:=/cmd_vel_gesture

### Terminal D: Start Auto + Shared fusion node

    source ~/.bashrc
    ros2 run gesture_nav2_shared cmd_vel_fusion

## Key Topic Design

In Auto + Shared mode:

- Nav2 output is remapped to /cmd_vel_nav
- The gesture node publishes /gesture/stable, /gesture/mode, and /gesture/estop
- The fusion node subscribes to /cmd_vel_nav and the gesture-state topics
- The fusion node is the only final publisher of /cmd_vel
- TurtleBot3 receives the final command from /cmd_vel

This design avoids command-source conflicts and makes the Fist emergency stop reliable.

## Gesture Behaviors in Auto + Shared Mode

- None / OpenPalm: pass Nav2 command
- PointLeft: add left angular correction bias
- PointRight: add right angular correction bias
- ThumbUp / TwoFingers: apply speed limit
- Fist: emergency stop, final /cmd_vel = 0

## Maps and Logs

The maps folder contains the saved TurtleBot3 world map used by Nav2.

The logs folder contains:

- auto_only_goal_log_final.csv
- shared_fusion_log_final.csv

These CSV files record gesture-triggered goal events and shared-control fusion events.

## Notes

This project was validated in TurtleBot3 Gazebo simulation. It was not deployed on a physical robot.
