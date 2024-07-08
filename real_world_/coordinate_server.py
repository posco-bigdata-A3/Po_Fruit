#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import socket
import json
import struct
import zlib
from typing import Dict, Any

def receive_object_info_server(port: int):
    try:
        # Create a socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", port))  # Bind to all available interfaces
            s.listen(1)
            print(f"Server listening on port {port}")

            while True:
                # Accept incoming connection
                client_socket, addr = s.accept()
                print(f"Accepted connection from {addr}")

                with client_socket:
                    # Receive data length
                    data_len_raw = client_socket.recv(4)
                    if not data_len_raw:
                        break
                    data_len = struct.unpack("!I", data_len_raw)[0]

                    # Receive compressed data
                    compressed_data = b""
                    while len(compressed_data) < data_len:
                        packet = client_socket.recv(data_len - len(compressed_data))
                        if not packet:
                            break
                        compressed_data += packet

                    # Decompress data
                    json_data = zlib.decompress(compressed_data)

                    # Deserialize JSON data
                    object_info_dict = json.loads(json_data)
                    print("Received data:")
                    for obj_name, obj_info in object_info_dict.items():
                        print(f"{obj_name}: {obj_info}")

                    # Respond to client that data was received
                    client_socket.sendall(b"Data received successfully")

    except Exception as e:
        print(f"Error receiving data: {e}")

if __name__ == "__main__":
    port = 1025  # Replace with yoursired port number
    receive_object_info_server(port)


# In[ ]:





# In[ ]:




