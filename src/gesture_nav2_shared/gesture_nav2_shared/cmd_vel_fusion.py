import time
import csv
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class CmdVelFusion(Node):
    """
    Auto + Shared control:
      - Nav2 publishes /cmd_vel_nav
      - Gesture node publishes /gesture/stable and /gesture/mode
      - This node publishes final /cmd_vel

    Gesture meanings:
      - Fist       : emergency stop
      - PointLeft  : add left-turn correction
      - PointRight : add right-turn correction
      - ThumbUp / TwoFingers : slow mode
      - OpenPalm / None : normal Nav2 tracking
    """

    def __init__(self):
        super().__init__('cmd_vel_fusion')

        # ---------- Parameters ----------
        self.declare_parameter('nav_timeout_sec', 0.5)
        self.declare_parameter('max_lin', 0.22)
        self.declare_parameter('max_ang', 1.20)
        self.declare_parameter('turn_bias', 0.35)
        self.declare_parameter('slow_scale', 0.45)
        self.declare_parameter('require_nav_mode', False)
        self.declare_parameter('status_print_period_sec', 1.0)

        # CSV log
        self.declare_parameter(
            'log_path',
            str(Path.home() / 'me740_logs' / 'shared_fusion_log.csv')
        )

        # ---------- State ----------
        self.last_nav = Twist()
        self.last_nav_time = 0.0

        self.stable = "None"
        self.mode = "IDLE"

        self.last_status_print_time = 0.0
        self.last_event = None

        # ---------- CSV ----------
        self.log_path = Path(str(self.get_parameter('log_path').value)).expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        new_file = (not self.log_path.exists()) or (self.log_path.stat().st_size == 0)
        self.csv_file = open(self.log_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        if new_file:
            self.csv_writer.writerow([
                "wall_time",
                "ros_time_sec",
                "event",
                "mode",
                "stable_gesture",
                "nav_lin_x",
                "nav_ang_z",
                "out_lin_x",
                "out_ang_z"
            ])
            self.csv_file.flush()

        # ---------- ROS I/O ----------
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(Twist, '/cmd_vel_nav', self.cb_nav, 10)
        self.create_subscription(String, '/gesture/stable', self.cb_stable, 10)
        self.create_subscription(String, '/gesture/mode', self.cb_mode, 10)

        self.timer = self.create_timer(0.05, self.on_timer)  # 20 Hz

        self.get_logger().info(
            "CmdVelFusion started.\n"
            "Inputs : /cmd_vel_nav, /gesture/stable, /gesture/mode\n"
            "Output : /cmd_vel\n"
            f"CSV log: {self.log_path}"
        )

    # ---------- callbacks ----------
    def cb_nav(self, msg: Twist):
        self.last_nav = msg
        self.last_nav_time = time.time()

    def cb_stable(self, msg: String):
        self.stable = msg.data

    def cb_mode(self, msg: String):
        self.mode = msg.data

    # ---------- helpers ----------
    def nav_is_fresh(self):
        timeout = float(self.get_parameter('nav_timeout_sec').value)
        return (time.time() - self.last_nav_time) <= timeout

    def log_event(self, event, nav, out):
        wall_time = datetime.now().isoformat(timespec="seconds")
        ros_time = self.get_clock().now().nanoseconds / 1e9

        self.csv_writer.writerow([
            wall_time,
            f"{ros_time:.3f}",
            event,
            self.mode,
            self.stable,
            f"{nav.linear.x:.4f}",
            f"{nav.angular.z:.4f}",
            f"{out.linear.x:.4f}",
            f"{out.angular.z:.4f}",
        ])
        self.csv_file.flush()

    def print_status(self, event, out, nav_ok):
        period = float(self.get_parameter('status_print_period_sec').value)
        now = time.time()

        if now - self.last_status_print_time >= period:
            self.get_logger().info(
                f"STATUS | mode={self.mode} | stable={self.stable} | "
                f"event={event} | nav_ok={int(nav_ok)} | "
                f"out_v={out.linear.x:.3f} | out_w={out.angular.z:.3f}"
            )
            self.last_status_print_time = now

    # ---------- main fusion ----------
    def on_timer(self):
        nav_ok = self.nav_is_fresh()

        nav = self.last_nav if nav_ok else Twist()
        out = Twist()

        # Start from Nav2 command
        out.linear.x = nav.linear.x
        out.angular.z = nav.angular.z

        event = "pass_nav"

        require_nav_mode = bool(self.get_parameter('require_nav_mode').value)

        # Optional strict gate
        if require_nav_mode and self.mode != "NAV":
            out = Twist()
            event = "hold_idle"

        # Emergency stop
        elif self.stable == "Fist":
            out = Twist()
            event = "estop_fist"

        # Slow mode
        elif self.stable in ("ThumbUp", "TwoFingers"):
            slow_scale = float(self.get_parameter('slow_scale').value)
            out.linear.x *= slow_scale
            out.angular.z *= slow_scale
            event = "speed_limit"

        # Left correction
        elif self.stable == "PointLeft":
            turn_bias = float(self.get_parameter('turn_bias').value)
            out.angular.z += turn_bias
            event = "correct_left"

        # Right correction
        elif self.stable == "PointRight":
            turn_bias = float(self.get_parameter('turn_bias').value)
            out.angular.z -= turn_bias
            event = "correct_right"

        # OpenPalm / None / others: keep Nav2 command
        else:
            event = "pass_nav"

        # Clamp final command
        max_lin = float(self.get_parameter('max_lin').value)
        max_ang = float(self.get_parameter('max_ang').value)

        out.linear.x = clamp(out.linear.x, -max_lin, max_lin)
        out.angular.z = clamp(out.angular.z, -max_ang, max_ang)

        self.pub.publish(out)

        # Print every second
        self.print_status(event, out, nav_ok)

        # Log only when event changes
        if event != self.last_event:
            self.log_event(event, nav, out)
            self.last_event = event

    def close_csv(self):
        try:
            self.csv_file.close()
        except Exception:
            pass


def main():
    rclpy.init()
    node = CmdVelFusion()

    try:
        rclpy.spin(node)
    finally:
        node.close_csv()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
