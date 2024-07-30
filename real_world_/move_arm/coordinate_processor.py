#!/usr/bin/env python
# coding: utf-8


import json
import subprocess
from typing import List

class ObjectData:
    def __init__(self, name: str, x: float, y: float, z: float, width: float):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.width = width

    def to_dict(self):
        return {self.name: {"x": self.x, "y": self.y, "z": self.z, "width": self.width}}

    def __repr__(self):
        return f"ObjectData(name={self.name}, x={self.x}, y={self.y}, z={self.z}, width={self.width})"

class DataProcessor:
    def __init__(self):
        self.python_executable = self.get_python_executable()

    def get_python_executable(self):
        # 가상환경의 Python 경로를 설정하세요
        # 예: return os.path.join("venv", "bin", "python")
        return "python"

    def save_to_file(self, object_data: ObjectData):
        data = object_data.to_dict()
        with open("object_info.json", "w") as f:
            json.dump(data, f)

    def run_inverse_kinematics(self, object_data: ObjectData):
        self.save_to_file(object_data)
        try:
            process = subprocess.Popen(
                [self.python_executable, 'run_inverse.py', 'object_info.json'],  # 정확한 파일명을 지정합니다.
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"처리 결과 ({object_data.name}):", stdout)
            else:
                print(f"에러 발생 ({object_data.name}):", stderr)
        except Exception as e:
            print(f"예외 발생 ({object_data.name}): {e}")

    def process_object(self, object_name: str, object_info: List[float]) -> ObjectData:
        x, y, z, width = object_info
        object_data = ObjectData(object_name, x, y, z, width)
        self.run_inverse_kinematics(object_data)
        return object_data





