import math
import time
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


@dataclass
class Goal2D:
    name: str
    x: float
    y: float
    yaw: float  # rad


def yaw_to_quat(yaw: float):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return 0.0, 0.0, float(qz), float(qw)


class GestureAutoOnly(Node):
    """
    Auto-only mode:
      - PointRight: select next goal
      - PointLeft : select previous goal
      - OpenPalm  : send selected goal, only when mode == NAV
      - Fist/None : estop=True, cancel current Nav2 task
    """

    def __init__(self):
        super().__init__('gesture_auto_only')

        # ---------- Parameters ----------
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('cooldown_sec', 0.8)
        self.declare_parameter('status_print_period_sec', 1.0)

        # ---------- Demo goals ----------
        # These are the three goals you selected in turtlebot3_world.
        # z from /clicked_point is ignored. yaw is set to 0 for a simple demo.
        self.goals = [
            Goal2D('G0_Start', -0.013969013467431068, -0.016796452924609184, 0.0),
            Goal2D('G1_Upper',  1.1549410820007324,    2.2911908626556396,   0.0),
            Goal2D('G2_Lower',  3.17816424369812,     -1.2948600053787231,   0.0),
        ]
        self.goal_idx = 0

        # ---------- Gesture state ----------
        self.stable = "None"
        self.prev_stable = "None"
        self.mode = "IDLE"
        self.estop = True

        # ---------- Debounce / status ----------
        self.last_action_time = 0.0
        self.last_status_print_time = 0.0

        # ---------- Task state ----------
        self.task_sent = False

        # ---------- CSV log ----------
        self.log_path = Path.home() / "me740_logs" / "auto_only_goal_log.csv"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        new_file = (not self.log_path.exists()) or (self.log_path.stat().st_size == 0)
        self.csv_file = open(self.log_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        if new_file:
            self.csv_writer.writerow([
                "wall_time",
                "ros_time_sec",
                "event",
                "goal_name",
                "x",
                "y",
                "yaw",
                "mode",
                "stable_gesture",
                "estop"
            ])
            self.csv_file.flush()

        # ---------- Nav2 ----------
        self.nav = BasicNavigator()
        self.get_logger().info("Waiting for Nav2 to become active...")
        self.nav.waitUntilNav2Active()
        self.get_logger().info("Nav2 is active. Auto-only ready.")

        # ---------- Subscribers ----------
        self.create_subscription(String, '/gesture/stable', self.cb_stable, 10)
        self.create_subscription(String, '/gesture/mode', self.cb_mode, 10)
        self.create_subscription(Bool,   '/gesture/estop', self.cb_estop, 10)

        self.get_logger().info(
            "Auto-only mapping:\n"
            "  PointLeft / PointRight: select previous / next goal\n"
            "  OpenPalm: send selected goal when mode == NAV\n"
            "  Fist or None: estop and cancel navigation\n"
            f"Initial goal = {self.current_goal().name}\n"
            f"CSV log = {self.log_path}"
        )

        self.timer = self.create_timer(0.2, self.on_timer)

    # ---------- Callbacks ----------
    def cb_stable(self, msg: String):
        self.stable = msg.data

    def cb_mode(self, msg: String):
        self.mode = msg.data

    def cb_estop(self, msg: Bool):
        self.estop = bool(msg.data)

    # ---------- Helpers ----------
    def current_goal(self) -> Goal2D:
        return self.goals[self.goal_idx]

    def can_act(self) -> bool:
        cooldown = float(self.get_parameter('cooldown_sec').value)
        return (time.time() - self.last_action_time) >= cooldown

    def mark_act(self):
        self.last_action_time = time.time()

    def current_goal_pose(self) -> PoseStamped:
        g = self.current_goal()

        pose = PoseStamped()
        pose.header.frame_id = str(self.get_parameter('frame_id').value)
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(g.x)
        pose.pose.position.y = float(g.y)

        _, _, qz, qw = yaw_to_quat(g.yaw)
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        return pose

    def task_active(self) -> bool:
        """
        Compatible way to check if Nav2 currently has a running task.
        Some Humble versions do not provide isTaskActive().
        """
        fb = self.nav.getFeedback()
        return (fb is not None) and (not self.nav.isTaskComplete())

    def print_status(self):
        period = float(self.get_parameter('status_print_period_sec').value)
        now = time.time()

        if now - self.last_status_print_time >= period:
            g = self.current_goal()
            self.get_logger().info(
                f"STATUS | mode={self.mode} | stable={self.stable} | "
                f"current_goal={g.name} | estop={int(self.estop)}"
            )
            self.last_status_print_time = now

    def log_goal_send(self, goal: Goal2D):
        wall_time = datetime.now().isoformat(timespec="seconds")
        ros_time = self.get_clock().now().nanoseconds / 1e9

        self.csv_writer.writerow([
            wall_time,
            f"{ros_time:.3f}",
            "send_goal",
            goal.name,
            f"{goal.x:.6f}",
            f"{goal.y:.6f}",
            f"{goal.yaw:.3f}",
            self.mode,
            self.stable,
            int(self.estop)
        ])
        self.csv_file.flush()

    # ---------- Main loop ----------
    def on_timer(self):
        self.print_status()

        gesture_changed = (self.stable != self.prev_stable)

        # 1) ESTOP cancels current task
        if self.estop:
            if self.task_active():
                self.get_logger().warn("ESTOP: cancel Nav2 task.")
                self.nav.cancelTask()
                self.task_sent = False

            self.prev_stable = self.stable
            return

        # 2) Goal selection: only trigger on gesture change + cooldown
        if gesture_changed and self.stable == "PointRight" and self.can_act():
            self.goal_idx = (self.goal_idx + 1) % len(self.goals)
            self.get_logger().info(f"Select goal: {self.current_goal().name}")
            self.mark_act()

        elif gesture_changed and self.stable == "PointLeft" and self.can_act():
            self.goal_idx = (self.goal_idx - 1) % len(self.goals)
            self.get_logger().info(f"Select goal: {self.current_goal().name}")
            self.mark_act()

        # 3) Send goal: only trigger on OpenPalm edge, mode==NAV, and cooldown
        elif gesture_changed and self.stable == "OpenPalm" and self.mode == "NAV" and self.can_act():
            g = self.current_goal()
            pose = self.current_goal_pose()

            self.get_logger().info(f"Send goal: {g.name}")
            self.nav.goToPose(pose)

            self.log_goal_send(g)

            self.task_sent = True
            self.mark_act()

        # 4) Report result once when a sent task completes
        if self.task_sent and self.nav.isTaskComplete():
            result = self.nav.getResult()

            if result == TaskResult.SUCCEEDED:
                self.get_logger().info("Nav2: goal reached.")
            elif result == TaskResult.CANCELED:
                self.get_logger().warn("Nav2: goal canceled.")
            else:
                self.get_logger().error("Nav2: goal failed.")

            self.task_sent = False

        self.prev_stable = self.stable

    def close_csv(self):
        try:
            self.csv_file.close()
        except Exception:
            pass


def main():
    rclpy.init()
    node = GestureAutoOnly()

    try:
        rclpy.spin(node)
    finally:
        node.close_csv()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
