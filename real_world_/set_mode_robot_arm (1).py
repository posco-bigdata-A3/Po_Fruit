#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

from time import sleep
import rospy
import sys
from moveit_commander.move_group import MoveGroupCommander
from moveit_commander import PlanningSceneInterface
import moveit_commander
from geometry_msgs.msg import PoseStamped

if __name__ == '__main__':
	# ROS 시스템 초기화
	moveit_commander.roscpp_initialize(sys.argv)
    # ROS 노드를 'Set_Scene'이라는 이름으로 초기화
	rospy.init_node('Set_Scene')
	# MoveIt!에서 제공하는 PlanningSceneInterface 클래스를 사용해 'scene' 객체를 초기화
	scene = PlanningSceneInterface()
    # 'dofbot'이라는 그룹에 대한 MoveGroupCommander 객체를 생성
	dofbot = MoveGroupCommander("dofbot")
    # 움직임 계획이 실패하면 새로운 계획을 세울 수 있도록 설정
	dofbot.allow_replanning(True)
	# 로봇이 움직임 계획을 세우는 데 최대 5초를 사용
	dofbot.set_planning_time(5)
    # 로봇이 움직임 계획을 세우는 데 10번의 시도
	dofbot.set_num_planning_attempts(10)
    # 각각 움직임의 목표 위치, 방향, 그리고 일반적인 허용 오차를 설정
	dofbot.set_goal_position_tolerance(0.01)
	dofbot.set_goal_orientation_tolerance(0.01)
	dofbot.set_goal_tolerance(0.01)
    # 최대 속도 및 가속도의 스케일링 요인을 설정
	dofbot.set_max_velocity_scaling_factor(1.0)
	dofbot.set_max_acceleration_scaling_factor(1.0)
	
	dofbot.set_named_target("up")
	dofbot.go()
	sleep(0.5)
	####
	
	# 테이블의 높이를 설정합니다. 여기서는 로봇의 기준점('base_link')으로부터 0.2 미터 높이에 테이블이 있다고 설정
    #table_ground = 0.2
    # 테이블의 3차원 크기를 설정합니다. 이 배열은 [길이, 너비, 높이]
    #table_size = [0.7, 0.1, 0.02]
    # 테이블의 위치와 방향을 저장하기 위한 PoseStamped 객체를 생성합니다. PoseStamped는 3차원 공간에서의 위치(x, y, z)와 방향(쿼터니언, quaternion) 정보를 가지는 메시지 타입
    #table_pose = PoseStamped()
	# PoseStamped 메시지의 좌표계를 로봇의 기준점인 'base_link'로 설정
    #table_pose.header.frame_id = 'base_link'
	# 테이블의 중심 위치를 설정합니다. z축의 위치는 테이블 높이와 테이블의 절반 높이를 합한 값으로, 테이블의 중심을 의미
    #table_pose.pose.position.x = 0
    #table_pose.pose.position.y = 0.15
    #table_pose.pose.position.z = table_ground + table_size[2] / 2.0
	# 테이블의 방향을 설정합니다. 여기서는 테이블이 로봇의 기준 좌표계에 따라서 평행하게 배치되어 있다고 가정하고 있습니다. 쿼터니언 형태의 회전에서 (0,0,0,1)은 회전이 없음
    #table_pose.pose.orientation.w = 1.0
	# 생성한 테이블을 '씬(Scene)'에 추가
    #scene.add_box('table', table_pose, table_size)
	# 'Scene'에 테이블이 추가되는 것을 기다리는 시간
    #rospy.sleep(2)
	
	
	# 도구의 크기를 설정
	tool_size = [0.03, 0.03, 0.03]
	# 로봇 암의 끝 단, 즉 end effector의 링크 이름을 가져옵
	end_effector_link = dofbot.get_end_effector_link()
    # PoseStamped 객체를 생성합니다. 이 객체는 3D 공간에서의 위치와 방향 정보를 포함
	p = PoseStamped()
	# 헤더의 frame_id를 end effector의 링크 이름으로 설정
	p.header.frame_id = end_effector_link
	# 도구의 위치를 설정
	p.pose.position.x = 0
	p.pose.position.y = 0
	p.pose.position.z = 0.10
	# 도구의 방향을 설정
	p.pose.orientation.x = 0
	p.pose.orientation.y = 0
	p.pose.orientation.z = 0
	p.pose.orientation.w = 1
    
    
    #-------------------------------------------------------------------------------
    #get out code 

	# 로봇의 목표 위치를 "pick"이라는 사전 정의된 위치로 설정
	dofbot.set_named_target("pick")
	# 로봇이 설정된 목표 위치로 움직이도록 
	dofbot.go()
	rospy.sleep(1)

	# Attach the tool to the robot
	scene.attach_box(end_effector_link, "tool", p, tool_size)
	rospy.sleep(1)
	
	# Move the robot arm to the "and" position
	dofbot.set_named_target("and")
	dofbot.go()
	rospy.sleep(1)

	# Move the robot arm to the "and1" position
	dofbot.set_named_target("and1")
	dofbot.go()
	rospy.sleep(1)

	# Move the robot arm to the "place" position and drop the tool
	dofbot.set_named_target("place")
	dofbot.go()
	rospy.sleep(1)

	# Detach and remove the tool
	scene.remove_attached_object(end_effector_link, "tool")
	rospy.sleep(1)
    
    # real my code
    #-----------------------------------------------------------------------------------
    #  과일이 여러개 박스 개수만큼 while 문 돌려서 코드 실행 
    # 가장 가까운 거리 인식 
    # 좌표로 이동 
    # 
    #-----------------------------------------------------------------------------------
    
    

	moveit_commander.roscpp_shutdown()
	moveit_commander.os._exit(0)


# 파일위치 : /home/dofbot/dofbot_ws/src/dofbot_moveit/scripts
# 실행방법 : (terminal) python /home/dofbot/dofbot_ws/src/dofbot_moveit/scripts/01_set_move_copy1.py
# pick, and, and1, place는 setup시 지정해줌


# In[ ]:




