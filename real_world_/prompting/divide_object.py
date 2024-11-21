import sys
sys.path.append('..')  # 상위 디렉토리로 경로 설정


from prompting.gpt_planner import GPTPlanner
from real_env import RealEnv, RoCo
from real_client import RealSenseClient

class Divide:
    def __init__(self):
        # RealSense 설정
        realsense_client = RealSenseClient("192.168.0.6", 1024, fielt_bg=False)
        self.env = RealEnv(
            bin_cam=realsense_client,
            task='Thinning green fruits',
            all_objects=["round fruit_6", "round fruit_5","round fruit_4", "round fruit_3", "round fruit_2", "round fruit_1", "Branch", "fresh biggest fruit"],
            task_objects=["round fruit_6", "round fruit_5","round fruit_4", "round fruit_3", "round fruit_2", "round fruit_1"]
        )
        
        # 객체 감지
        self.roco = RoCo(self.env)
        self.detected_objects = self.roco.execute()

        # 로봇 위치 설정 (예시)
        self.robot1_position = (0.2, 0.0, 0.0)  # 좌측
        self.robot2_position = (-0.2, 0.0, 0.0)  # 우측

    def allocate_objects(self):
        # GPT-4를 사용하여 객체 분배 계획
        planner = GPTPlanner()
        robot1_allocations, robot2_allocations = planner.plan_and_allocate(self.robot1_position, self.robot2_position, self.detected_objects)
        print("Gangguk allocations:", robot1_allocations)
        print()
        print("Eunyong allocations:", robot2_allocations)
        # 결과 반환
        return robot1_allocations, robot2_allocations


if __name__ == "__main__":
    divider = Divide()
    robot1_allocations, robot2_allocations = divider.allocate_objects()
    print("Gangguk allocations:", robot1_allocations)
    print("Eunyong allocations:", robot2_allocations)

