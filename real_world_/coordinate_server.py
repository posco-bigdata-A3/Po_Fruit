#!/usr/bin/env python
# coding: utf-8

# In[2]:


import socket
import json
import struct
import zlib
import time
from data_processor import DataProcessor, ObjectData

def receive_object_info_server(port: int):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", port))
            s.listen(1)
            print(f"서버가 포트 {port}에서 리스닝 중입니다.")
            
            while True:
                client_socket, addr = s.accept()
                print(f"{addr}로부터 연결을 수락했습니다.")
                
                with client_socket:
                    data_len_raw = client_socket.recv(4)
                    if not data_len_raw:
                        break
                    data_len = struct.unpack("!I", data_len_raw)[0]
                    
                    compressed_data = b""
                    while len(compressed_data) < data_len:
                        packet = client_socket.recv(data_len - len(compressed_data))
                        if not packet:
                            break
                        compressed_data += packet
                    
                    json_data = zlib.decompress(compressed_data)
                    object_info_dict = json.loads(json_data)
                    
                    print("수신된 데이터:")
                    processor = DataProcessor()
                    objects = []
                    for obj_name, obj_info in object_info_dict.items():
                        print(f"{obj_name}: {obj_info}")
                        obj_data = processor.process_object(obj_name, obj_info)
                        objects.append(obj_data)
                        time.sleep(2)  # 다음 객체를 처리하기 전에 2초 대기
                    
                    print("모든 객체가 처리되었습니다.")
                    for obj in objects:
                        print(obj)
                    
                    client_socket.sendall("데이터를 성공적으로 수신했습니다.".encode('utf-8'))
    except Exception as e:
        print(f"데이터 수신 중 오류 발생: {e}")

if __name__ == "__main__":
    port = 1025  # 원하는 포트 번호로 변경하세요
    receive_object_info_server(port)


# In[ ]:




