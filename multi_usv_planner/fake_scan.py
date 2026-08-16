#!/usr/bin/env python3
"""假雷达: 模拟世界固定障碍 (1.0, 0.0) 半径0.15m, 订阅 /odom 计算相对扫描
真实雷达的行为: 障碍固定在墙上/地面, 车移动时障碍相对车的 range/angle 变化"""
import math, rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

N = 360
ANGLE_INC = 2 * math.pi / N

class FakeScan(Node):
    def __init__(self):
        super().__init__('fake_scan')
        self.pub = self.create_publisher(LaserScan, '/scan', 10)
        self.timer = self.create_timer(0.05, self.pub_scan)  # 20Hz
        self.psi = 0.0
        self.x = 0.0
        self.y = 0.0
        q = rclpy.qos.QoSProfile(depth=5, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(Odometry, '/odom', self.odom_cb, q)

    def odom_cb(self, m):
        q = m.pose.pose.orientation
        self.psi = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.x = m.pose.pose.position.x
        self.y = m.pose.pose.position.y

    def pub_scan(self):
        m = LaserScan()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'laser_link'
        m.angle_min = 0.0
        m.angle_max = ANGLE_INC * (N - 1)
        m.angle_increment = ANGLE_INC
        m.range_min = 0.05
        m.range_max = 6.0
        m.ranges = [6.0] * N
        # 世界固定障碍 (1.0, 0.0) 半径 0.15
        ox, oy = 1.0, 0.0
        dx, dy = ox - self.x, oy - self.y
        dist = math.hypot(dx, dy)
        if 0.05 < dist < 6.0:
            ang = math.atan2(dy, dx) - self.psi   # 障碍相对车头角度
            span = math.atan2(0.15, dist)          # 障碍张角
            for i in range(N):
                a = ANGLE_INC * i
                # 归一化角度差
                da = abs(((a - ang + math.pi) % (2*math.pi)) - math.pi)
                if da < span:
                    m.ranges[i] = dist
        self.pub.publish(m)

def main():
    rclpy.init()
    rclpy.spin(FakeScan())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
