# Raspberry Pi (Sender)
import socket
import time

server_address = ('<PC_IP_ADDRESS>', 12345)  # Replace with PC's IP and port
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    message = "Hello from Raspberry Pi!"
    client_socket.sendto(message.encode(), server_address)
    time.sleep(2)