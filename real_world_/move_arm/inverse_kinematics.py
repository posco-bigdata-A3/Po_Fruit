#!/usr/bin/env python
# coding: utf-8

# In[64]:


import sympy
import time
from Arm_Lib import Arm_Device
import math
from sympy import symbols, cos, sin, pi, sqrt, atan2, simplify, Matrix, N

duration = 2000
fruit_size = 145  # 이 값을 제거합니다.

class DofbotController:
    
    def __init__(self):
        self.Arm = Arm_Device()
        self.Arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 4000)
        self.d1 = 0.6605
        self.d2 = 0.4145
        self.a2 = 0.8285  
        self.a3 = 0.8285  
        self.a4 = 0.7385
        self.l = 59.85  # 이전에 계산한 값을 사용합니다.
    
    def pose(self, theta, alpha, a, d):
        T = Matrix([
            [cos(theta), -sin(theta), 0, a],
            [sin(theta) * cos(alpha), cos(theta) * cos(alpha), -sin(alpha), -d * sin(alpha)],
            [sin(theta) * sin(alpha), cos(theta) * sin(alpha), cos(alpha), d * cos(alpha)],
            [0, 0, 0, 1]
        ])
        T = simplify(T)
        return T

    def forward_kinematics(self, q1, q2, q3, q4):
        d90 = pi / 2
        T01 = self.pose(q1, 0, 0, self.d1)
        T12 = self.pose(q2 - d90, d90, 0, self.d2)
        T23 = self.pose(q3, 0, self.a2, 0)
        T34 = self.pose(q4, 0, self.a3, 0)
        T45 = self.pose(0, 0, self.a4, 0)
        
        T0g = T01 * T12 * T23 * T34 * T45
        return T0g    
    
    def inverse_kinematics(self, x, y, z, width):
        # Wrist center calculation
        xc = x
        yc = y
        zc = z - self.d1 - self.d2

        # Angle calculations
        q1 = atan2(yc, xc)
        r = sqrt(xc ** 2 + yc ** 2)
        s = zc

        D = (r ** 2 + s ** 2 - self.a2 ** 2 - self.a3 ** 2) / (2 * self.a2 * self.a3)
        if D < -1 or D > 1:
            raise ValueError("The point is out of the robot's reach")

        q3 = atan2(sqrt(1 - D ** 2), D)
        q2 = atan2(s, r) - atan2(self.a3 * sin(q3), self.a2 + self.a3 * cos(q3))

        # Ensure q2, q3 are positive by wrapping them correctly
        q2 = q2 % (2 * pi)
        q3 = q3 % (2 * pi)

        if q2 < 0:
            q2 += 2 * pi
        if q3 < 0:
            q3 += 2 * pi

        # q4 calculation (rotation angle at the target gripper position)
        T0g = self.forward_kinematics(q1, q2, q3, 0)
        T0g_inv = T0g.inv()
        gripper_vector = Matrix([x, y, z, 1])
        target_vector = T0g_inv * gripper_vector
        q4 = atan2(target_vector[1], target_vector[0])
        
        print(q1, q2, q3, q4)
        
        # Convert angles to degrees and ensure they are within range
        q1 = math.degrees(N(q1))
        q2 = math.degrees(N(q2))
        q3 = math.degrees(N(q3))
        q4 = math.degrees(N(q4))

        # Wrap angles into 0-180 range
        q1 = q1 % 360
        q2 = q2 % 360
        q3 = q3 % 360
        q4 = q4 % 360

        if q1 > 180:
            q1 -= 360
        if q2 > 180:
            q2 -= 360
        if q3 > 180:
            q3 -= 360
        if q4 > 180:
            q4 -= 360

        print(q1, q2, q3, q4)

        # q6 계산
        cos_q6 = (2 * self.l ** 2 - width ** 2) / (2 * self.l ** 2)
        theta = math.degrees(math.acos(cos_q6))
        q6 = 180 - theta

        return (q1, q2, q3, q4, q6)

    def move_arm(self, x, y, z, q5, width, duration=3000):
        q1, q2, q3, q4, q6 = self.inverse_kinematics(x, y, z, width)
        
        # Limit angles to their ranges
        q1 = max(0, min(180, q1))
        q2 = max(0, min(180, q2))
        q3 = max(0, min(180, q3))
        q4 = max(0, min(180, q4))
        q5 = max(0, min(270, q5))
        q6 = max(0, min(180, q6))
        
        self.Arm.Arm_serial_servo_write6(q1, q2, q3, q4, q5, q6, duration)
        time.sleep(3)

    def fruit_check(self, width):
        q6 = self.calculate_q6(width)
        self.Arm.Arm_serial_servo_write6(90, 80, 70, 60, 90, q6, duration)
        
    def calculate_q6(self, width):
        cos_q6 = (2 * self.l ** 2 - width ** 2) / (2 * self.l ** 2)
        theta = math.degrees(math.acos(cos_q6))
        q6 = 180 - theta
        return q6
        
    def pick_fruit(self, x, y, z, width):
        self.move_arm(x, y, z, 90, 90, 2000)  # q5 is 90 degrees (vertical), q6 is calculated
        time.sleep(3)
        self.move_arm(x, y, z, 90, width, duration)
        time.sleep(2)
        self.move_arm(x, y, z, 0, width, duration)
        time.sleep(2)
        self.move_arm(x, y, z, 270, width, duration)
        time.sleep(2)
        self.move_arm(x, y, z, 0, width, duration)
        time.sleep(2)
        self.move_arm(x, y, z, 270, width, duration)
        time.sleep(2)
        self.fruit_check(width)

    def input_BOX(self, width):
        q6 = self.calculate_q6(width)
        time.sleep(2)
        self.Arm.Arm_serial_servo_write6(180, 90, 90, 90, 90, q6, duration)
        time.sleep(3)
        self.Arm.Arm_serial_servo_write6(180, 120, 30, 30, 90, q6, duration)
        time.sleep(2)
        self.Arm.Arm_serial_servo_write6(180, 120, 30, 30, 90, 5, duration)
        time.sleep(1)

    def base_mode(self):
        self.Arm.Arm_serial_servo_write6(180, 90, 90, 90, 90, 0, duration)
        time.sleep(3)
        self.Arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 4000)
        time.sleep(3)


# In[ ]:




