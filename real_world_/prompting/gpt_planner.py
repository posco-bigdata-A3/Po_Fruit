import openai
import os
import numpy as np
import json
from typing import Dict, Tuple

class GPTPlanner:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 읽기

    def plan_and_allocate(self, robot1_position: Tuple[float, float, float], 
                          robot2_position: Tuple[float, float, float],
                          objects: Dict[str, np.ndarray]) -> Tuple[Dict[str, list], Dict[str, list]]:
        prompt = self.create_prompt(robot1_position, robot2_position, objects)
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",  # LLM 모델 선택택
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000  # 대화와 할당을 포함할 수 있도록 토큰 수를 증가
        )
        message_content = response.choices[0]['message']['content']
        conversation, allocation_str = self.split_conversation_and_allocation(message_content)

        # 대화 내용을 파일로 저장
        with open("conversation.txt", "w") as file:
            file.write(conversation)

        print("Conversation between Gangguk and Eunyong:\n", conversation)  # 대화를 출력
        allocation = self.parse_response(allocation_str)
        return allocation

    def create_prompt(self, robot1_position: Tuple[float, float, float], 
                      robot2_position: Tuple[float, float, float],
                      objects: Dict[str, np.ndarray]) -> str:
        prompt = (
            "Two robots are collaborating to pick objects in a space. The camera is at (0, 0, 0). "
            f"Gangguk is at {robot1_position} and Eunyong is at {robot2_position}. The objects detected are:\n"
        )

        for obj_name, obj_attr in objects.items():
            x, y, z, width = obj_attr  # numpy 배열에서 값을 추출
            prompt += f"{obj_name}: position({x}, {y}, {z}), width {width}\n"

        prompt += (
            "Please divide the objects between Gangguk and Eunyong based on their positions. "
            "First, simulate a conversation between Gangguk and Eunyong discussing which objects to pick. "
            "Use the following format for the conversation:\n"
            "Gangguk: \"There are [number] items in front of us. I'll choose first. I'll take the [object].\"\n"
            "Eunyong: \"Alright. Then I'll take the [object] from position (x, y, z).\"\n"
            "Gangguk: \"Okay, I'll take the [object] from position (x, y, z).\"\n"
            "Eunyong: \"Got it. I'll take the [object] from position (x, y, z).\"\n"
            "Gangguk: \"Finally, I'll take the [object] from position (x, y, z).\"\n"
            "Eunyong: \"Great. Now we've divided all the items.\"\n"
            "Then, provide the final allocation in the following JSON format without any additional text or explanation:\n"
            "{\n"
            '  "Gangguk allocations": {\n'
            '    "object_name": {"x": value, "y": value, "z": value, "width": value},\n'
            '    ...\n'
            '  },\n'
            '  "Eunyong allocations": {\n'
            '    "object_name": {"x": value, "y": value, "z": value, "width": value},\n'
            '    ...\n'
            '  }\n'
            "}\n"
            "Make sure the JSON is valid."
        )

        return prompt

    def split_conversation_and_allocation(self, message_content: str) -> Tuple[str, str]:
        # 대화와 할당 결과를 분리하는 메서드
        try:
            conversation, allocation_str = message_content.split("{", 1)
            allocation_str = "{" + allocation_str.rsplit("}", 1)[0] + "}"
            return conversation.strip(), allocation_str.strip()
        except ValueError:
            # 기본 값으로 반환하여 오류를 처리
            return message_content.strip(), ""

    def parse_response(self, response: str) -> Tuple[Dict[str, list], Dict[str, list]]:
        try:
            allocations = json.loads(response)
        except (json.JSONDecodeError, SyntaxError) as e:
            raise ValueError(f"Invalid response format: {response}") from e

        robot1_allocations = {k: list(v.values()) for k, v in allocations.get("Gangguk allocations", {}).items()}
        robot2_allocations = {k: list(v.values()) for k, v in allocations.get("Eunyong allocations", {}).items()}
        return robot1_allocations, robot2_allocations
