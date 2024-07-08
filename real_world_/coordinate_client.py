#!/usr/bin/env python
# coding: utf-8

# In[2]:


import socket
import json
import struct
import zlib
from typing import Dict, List

MAX_DATA_SIZE = 1024 * 1024  # 최대 1MB로 제한
HOST = "141.223.140.15"
PORT = 1025

def send_object_info(object_info_dict: Dict[str, List[float]]):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10)  # 타임아웃 설정을 10초로 변경
            s.connect((HOST, PORT))
            print(f"Connected to server at {HOST}:{PORT}")

            # JSON 문자열로 변환, 인코딩 후 압축
            json_data = json.dumps(object_info_dict).encode('utf-8')
            compressed_data = zlib.compress(json_data)

            # 데이터 크기 확인
            data_len = len(compressed_data)
            if data_len > MAX_DATA_SIZE:
                raise ValueError(f"데이터 크기가 최대 허용 크기({MAX_DATA_SIZE}바이트)를 초과합니다.")

            # 데이터 길이를 패킹
            data_len_packed = struct.pack("!I", data_len)

            # 데이터 길이 전송
            s.sendall(data_len_packed)

            # 실제 데이터 전송
            s.sendall(compressed_data)

            print(f"데이터 전송 성공 (압축 후 크기: {len(compressed_data)} 바이트)")
    except socket.timeout as e:
        print(f"소켓 타임아웃 발생: {e}")
    except socket.error as e:
        print(f"소켓 에러 발생: {e}")
    except ValueError as e:
        print(f"데이터 크기 초과 오류: {e}")
    except Exception as e:
        print(f"기타 오류 발생: {e}")

if __name__ == "__main__":
    object_info_dict = {
        "object1": [-8.0, 1.0, -2.0, 6.0],
        "object2": [-1.0, 5.0, -4.0, 8.0],
        "object3": [3.0, 4.0, 1.0, 2.0]
    }
    print(f"원본 데이터 크기: {len(json.dumps(object_info_dict))} 바이트")
    send_object_info(object_info_dict)


# In[ ]:





# In[ ]:




