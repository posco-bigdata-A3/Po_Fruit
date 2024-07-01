import numpy as np
import sys
import dofbot_collab.realdofbot_utils as realdof_utils
import rospy
import moveit_commander
from std_srvs.srv import Empty
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander

class DOFBOTMoveTimeoutException(Exception):
    def __init__(self):
        super().__init__('DOFBOT Move Timeout')

class DOFBOTROS:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('dofbot_ros_interface', anonymous=True)

        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander("manipulator")
        
        rospy.wait_for_service('/clear_octomap')
        self.clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)
        
        # DOFBOT의 초기 위치 설정 (예시 값)
        self.home_joint = [0, -1.57, 1.57, -1.57, 0, 0]
    
    def home(self):
        self.group.go(self.home_joint, wait=True)
        self.group.stop()
    
    def movej(self, q):
        self.group.go(q, wait=True)
        self.group.stop()
    
    def movel(self, pose):
        self.group.set_pose_target(pose)
        plan = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
    
    def open_gripper(self):
        # 그리퍼 열기 로직 구현 (필요한 경우)
        pass
    
    def close_gripper(self):
        # 그리퍼 닫기 로직 구현 (필요한 경우)
        pass
    
    def check_grasp(self):
        # 그리퍼 상태 확인 로직 구현 (필요한 경우)
        pass
    
    # DOFBOT의 작업 공간 내에 있는지 확인하는 함수 (예시 값)
    def check_pose_reachable(self, pose):
        x, y, z = pose
        if x < 0.1 or x > 0.5 or y < -0.5 or y > 0.5 or z < 0.0 or z > 0.5:
            return False
        else:
            return True 

if __name__ == "__main__":
    try:
        dofbot = DOFBOTROS()
        dofbot.home()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass