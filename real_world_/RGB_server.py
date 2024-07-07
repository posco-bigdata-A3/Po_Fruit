import cv2
import socket
import struct
import pickle
import threading
import queue
import tkinter as tk
from PIL import Image, ImageTk

# 각 비디오 스트림을 위한 큐 생성
frame_queue_1 = queue.Queue(maxsize=10)
frame_queue_2 = queue.Queue(maxsize=10)

def receive_video(port, frame_queue):
    print(f"Starting server on port {port}...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(5)
    print(f"Server listening on port {port}")

    conn, addr = server_socket.accept()
    print(f"Connection accepted from {addr}")
    data = b""
    payload_size = struct.calcsize("L")

    while True:
        try:
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    print("Connection closed by client.")
                    return
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            while len(data) < msg_size:
                packet = conn.recv(4096)
                if not packet:
                    print("Connection closed by client.")
                    return
                data += packet

            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = pickle.loads(frame_data)
            print(f"Received frame on port {port}: {frame.shape}, {frame.dtype}")  # 프레임 정보 출력
            if frame_queue.full():
                frame_queue.get()  # 큐가 가득 차면 오래된 프레임을 버림
            frame_queue.put(frame)
            print(f"Frame received and buffered on port {port}.")
            
        except Exception as e:
            print(f"Error on port {port}: {e}")
            break

    conn.close()
    server_socket.close()
    print(f"Server stopped on port {port}.")

def update_frame(label, frame_queue):
    if not frame_queue.empty():
        frame = frame_queue.get()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    label.after(10, update_frame, label, frame_queue)

# GUI 설정
root = tk.Tk()
root.title("Multi-Camera Stream")

# 각 카메라 스트림을 위한 레이블 생성
label_1 = tk.Label(root)
label_1.pack(side="left")
label_2 = tk.Label(root)
label_2.pack(side="right")

# 비디오 수신을 위한 스레드 시작(포트번호 입력 8000, 8001)
receive_thread_1 = threading.Thread(target=receive_video, args=(8000, frame_queue_1))
receive_thread_2 = threading.Thread(target=receive_video, args=(8001, frame_queue_2))
receive_thread_1.start()
receive_thread_2.start()

# 프레임 업데이트
update_frame(label_1, frame_queue_1)
update_frame(label_2, frame_queue_2)

# GUI 루프 시작
root.mainloop()

# 스레드 종료 대기
receive_thread_1.join()
receive_thread_2.join()