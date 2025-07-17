import socket

server_address = ('172.20.10.2', 12345)  # Same IP and port as sender
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(server_address)


def retrieve_data():    
    while True:
        data, addr = server_socket.recvfrom(1024)
        print(f"Received: {data.decode()} from {addr}")



#send cm needed to travel
# def main(): 
#     cmd = ''
#     num = (input("Enter a cmd (Options are 2, S, I, M, D): "))
#     num = num[0]
#     write_read(num) 
#         # Wait until Arduino sends a non-empty message back
#     match num:
#         case 'S': #spin to angle
#             heading = int(input("Enter a heading in degrees: "))# Taking input from user
#             write_read(heading)
#         case 'I': #spin infinitely
#             inp = input("Press Enter to stop spinning and continue...")
#             write_read(inp)
#         case 'M': #move forward infinitely
#             inp = input("Press Enter to stop moving and continue...")
#             write_read(inp)
#         case '2': #flight path 2
#             cm = input("Enter cm to travel forward: ")
#             write_read(cm)
#             cm = input("Enter cm to travel down")
#             write_read(cm)
#         #case w just gyro data
#         #case 'D' does not exist because the FlyInPlace() method does not require extra parameters
#     main()

#main()

retrieve_data()



