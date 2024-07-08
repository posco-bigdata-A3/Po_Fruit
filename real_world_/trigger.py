import socket

class TriggerClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print(f"Connected to trigger listener at {self.host}:{self.port}")

    def send_trigger(self):
        self.socket.sendall(b"TRIGGER")
        print("Trigger sent")

if __name__ == '__main__':
    ip = "141.223.140.23"  # 클라이언트 내부 네트워크 IP 주소 사용
    port = 1025  # 트리거를 위한 포트 번호

    trigger_client = TriggerClient(ip, port)

    while True:
        input("Press Enter to send trigger...")
        trigger_client.send_trigger()
