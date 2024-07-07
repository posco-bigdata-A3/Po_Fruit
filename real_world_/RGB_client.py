import cv2
import socket
import struct
import pickle
import time

def send_video(server_ip, port):
    print(f"Starting video capture on port {port}...")
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)  # 카메라 인덱스 1과 V4L2 백엔드 사용
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("Video device opened successfully.")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((server_ip, port))
        print(f"Connected to server on port {port}.")
    except Exception as e:
        print(f"Error connecting to server on port {port}: {e}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        try:
            # 프레임을 직렬화하여 전송
            data = pickle.dumps(frame)
            # 패킷 헤더에 데이터 길이를 추가
            message_size = struct.pack("L", len(data))
            # 헤더와 프레임 데이터 전송
            client_socket.sendall(message_size + data)
            print("Frame sent.")
        except Exception as e:
            print(f"Error sending frame: {e}")
            break

        time.sleep(0.1)  # CPU 사용을 줄이기 위해 지연 시간 추가

    cap.release()
    client_socket.close()
    print(f"Video capture stopped on port {port}.")

# 첫 번째 Jetson Nano에서 실행
send_video('YOUR_SERVER_IP', 8000) # YOUR_SERVER_IP에 서버 IP & 포트번호

