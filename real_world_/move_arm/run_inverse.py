#!/usr/bin/env python
# coding: utf-8


import sys
import json
from inverse_kinematics import DofbotController

class ObjectData:
    def __init__(self, name, x, y, z, width):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.width = width

    @staticmethod
    def from_dict(name, data):
        return ObjectData(name, data['x'], data['y'], data['z'], data['width'])

    def __repr__(self):
        return "ObjectData(name={}, x={}, y={}, z={}, width={})".format(self.name, self.x, self.y, self.z, self.width)

def inverse_kinematics_and_movement(object_data):
    print("처리 중 ({}): {}, {}, {}, {}".format(object_data.name, object_data.x, object_data.y, object_data.z, object_data.width))
    
    controller = DofbotController()
    controller.pick_fruit(object_data.x, object_data.y, object_data.z, object_data.width)
    controller.input_BOX(object_data.width)
    controller.base_mode()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python run_inverse.py <object_info.json>")
        sys.exit(1)
    
    object_info_file = sys.argv[1]
    
    with open(object_info_file, "r") as f:
        object_data_dict = json.load(f)
    
    for obj_name, obj_info in object_data_dict.items():
        object_data = ObjectData.from_dict(obj_name, obj_info)
        inverse_kinematics_and_movement(object_data)




