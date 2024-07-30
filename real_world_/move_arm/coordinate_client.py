import socket
import json
import struct
import zlib
from typing import Dict, List
from robot import Robot

MAX_DATA_SIZE = 1024 * 1024  # 최대 1MB로 제한

def send_object_info(host: str, port: int, object_info_dict: Dict):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10)  # 타임아웃 설정을 10초로 변경
            s.connect((host, port))
            print(f"서버 {host}:{port}에 연결되었습니다.")

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

            # 서버로부터 응답 수신
            response = s.recv(1024)
            print(f"서버 응답: {response.decode('utf-8')}")
    except socket.timeout as e:
        print(f"소켓 타임아웃 발생: {e}")
    except socket.error as e:
        print(f"소켓 에러 발생: {e}")
    except ValueError as e:
        print(f"데이터 크기 초과 오류: {e}")
    except Exception as e:
        print(f"기타 오류 발생: {e}")

if __name__ == "__main__":
    host = "192.168.0.6" # 해당 로봇의 컴퓨터 ip 주소
    
    data1 = {
        "object1": [-8.0, 1.0, -2.0, 6.0],
        "object2": [-1.0, 5.0, -4.0, 8.0],
        "object3": [3.0, 4.0, 1.0, 2.0]
    }
    data2 = {
        "object4": [7.0, -3.0, 2.0, 5.0],
        "object5": [0.0, 2.0, -1.0, 4.0],
        "object6": [-5.0, 3.0, 4.0, 6.0]
    }

    robot = Robot()
    robot1, robot2 = robot.main(data1, data2)

    send_object_info(host, 1025, robot1)
    send_object_info(host, 1026, robot2)
