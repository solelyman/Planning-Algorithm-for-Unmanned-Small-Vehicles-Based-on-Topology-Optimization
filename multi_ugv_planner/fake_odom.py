#!/usr/bin/env python3
"""假 odom: 订阅 /cmd_vel 按 unicycle 积分, 发布 /odom (BEST_EFFORT), 模拟 UGV 闭环"""
import math, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class FakeOdom(Node):
    def __init__(self):
        super().__init__('fake_odom')
        q = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                 durability=rclpy.qos.DurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(Odometry, '/odom', q)
        self.sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, q)
        self.x, self.y, self.psi, self.v, self.w = 0.0, 0.0, 0.0, 0.0, 0.0
        self.last = self.get_clock().now()
        self.timer = self.create_timer(0.02, self.step)  # 50Hz 积分

    def cmd_cb(self, m):
        self.v = m.linear.x
        self.w = m.angular.z

    def step(self):
        now = self.get_clock().now()
        dt = min((now - self.last).nanoseconds / 1e9, 0.1)
        self.last = now
        self.psi += self.w * dt
        self.x += self.v * math.cos(self.psi) * dt
        self.y += self.v * math.sin(self.psi) * dt
        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        q = msg.pose.pose.orientation
        q.z, q.w = math.sin(self.psi/2), math.cos(self.psi/2)
        msg.twist.twist.linear.x = self.v
        msg.twist.twist.angular.z = self.w
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(FakeOdom())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
