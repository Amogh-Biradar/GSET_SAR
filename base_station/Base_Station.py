# Required packages:
# pip install matplotlib numpy

#To run:
#source wifi_env/bin/activate
#python wifi_reciever.py
import socket
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import re
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

class DroneMonitorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAR Drone Monitor")
        self.root.geometry("700x1000")
        self.root.configure(bg="#f0f0f0")
        
        # Data storage
        self.position_x = 0.0
        self.position_y = 0.0
        self.target_distance = 0.0
        self.target_heading = 0.0
        self.distance_remaining = 0.0
        self.heading_needed = 0.0
        self.straight_line_distance = 0.0
        
        # History for plotting
        self.position_history_x = [0.0]
        self.position_history_y = [0.0]
        self.time_history = [0.0]
        self.start_time = time.time()
        
        # Alert status
        self.alert_active = False
        self.last_alert_time = 0
        self.last_update_time = time.time()
        
        self.target_locked = False
        self.locked_target_distance = 0.0
        self.locked_target_heading = 0.0
        self.locked_distance_remaining = 0.0
        self.locked_heading_needed = 0.0
        self.locked_straight_line_distance = 0.0
        
        self.setup_ui()
        self.setup_network()
        
        # Start receiver thread
        self.running = True
        self.receiver_thread = threading.Thread(target=self.receive_data)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
        
        # Update UI periodically
        self.root.after(100, self.update_ui)

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Style configuration
        style = ttk.Style()
        style.configure("Alert.TLabel", foreground="red", font=("Arial", 16, "bold"))
        style.configure("TLabel", font=("Arial", 14))
        style.configure("Header.TLabel", font=("Arial", 18, "bold"))
        style.configure("Value.TLabel", font=("Arial", 16))
        
        # Configure tab style
        style.configure("TNotebook.Tab", font=("Arial", 14, "bold"))
        style.configure("TNotebook", font=("Arial", 14))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.main_tab = ttk.Frame(self.notebook)
        self.graph_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.main_tab, text="Main Dashboard")
        self.notebook.add(self.graph_tab, text="Trajectory Graph")
        
        # Setup main tab content
        self.setup_main_tab()
        
        # Setup graph tab content
        self.setup_graph_tab()

    def setup_main_tab(self):
        # Top section - Status and Position
        top_frame = ttk.Frame(self.main_tab)
        top_frame.pack(fill=tk.X, pady=5)
        
        # Status indicator
        self.status_frame = ttk.Frame(top_frame, relief=tk.RAISED, borderwidth=2)
        self.status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(self.status_frame, text="DRONE STATUS", style="Header.TLabel").pack(pady=5)
        
        status_inner_frame = ttk.Frame(self.status_frame)
        status_inner_frame.pack(padx=10, pady=5)
        
        ttk.Label(status_inner_frame, text="Connection:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.connection_status = ttk.Label(status_inner_frame, text="CONNECTED", foreground="green", font=("Arial", 14, "bold"))
        self.connection_status.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(status_inner_frame, text="Last Update:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.last_update = ttk.Label(status_inner_frame, text="--", font=("Arial", 14))
        self.last_update.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Position display
        position_frame = ttk.Frame(self.main_tab, relief=tk.RAISED, borderwidth=2)
        position_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(position_frame, text="CURRENT POSITION", style="Header.TLabel").pack(pady=5)
        
        position_inner_frame = ttk.Frame(position_frame)
        position_inner_frame.pack(padx=10, pady=5)
        
        ttk.Label(position_inner_frame, text="X Coordinate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.x_coord_label = ttk.Label(position_inner_frame, text="0.00 m", style="Value.TLabel")
        self.x_coord_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(position_inner_frame, text="Y Coordinate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.y_coord_label = ttk.Label(position_inner_frame, text="0.00 m", style="Value.TLabel")
        self.y_coord_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Middle section - Navigation data
        middle_frame = ttk.Frame(self.main_tab)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Navigation data
        nav_frame = ttk.Frame(middle_frame, relief=tk.RAISED, borderwidth=2)
        nav_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        ttk.Label(nav_frame, text="NAVIGATION DATA", style="Header.TLabel").pack(pady=5)
        
        nav_inner_frame = ttk.Frame(nav_frame)
        nav_inner_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ttk.Label(nav_inner_frame, text="Target Distance:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.target_dist_label = ttk.Label(nav_inner_frame, text="0.00 m", style="Value.TLabel")
        self.target_dist_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(nav_inner_frame, text="Target Heading:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.target_head_label = ttk.Label(nav_inner_frame, text="0.0°", style="Value.TLabel")
        self.target_head_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(nav_inner_frame, text="Distance Remaining:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.dist_remain_label = ttk.Label(nav_inner_frame, text="0.00 m", style="Value.TLabel")
        self.dist_remain_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(nav_inner_frame, text="Heading Needed:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.head_needed_label = ttk.Label(nav_inner_frame, text="0.0°", style="Value.TLabel")
        self.head_needed_label.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(nav_inner_frame, text="Straight-Line Distance:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.straight_dist_label = ttk.Label(nav_inner_frame, text="0.00 m", style="Value.TLabel")
        self.straight_dist_label.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Bottom section - Alert log
        bottom_frame = ttk.Frame(self.main_tab, relief=tk.RAISED, borderwidth=2)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(bottom_frame, text="ALERT LOG", style="Header.TLabel").pack(pady=5)
        
        # Alert indicator
        self.alert_indicator = ttk.Label(bottom_frame, text="NO ALERTS", foreground="green", font=("Arial", 16, "bold"))
        self.alert_indicator.pack(pady=5)
        
        # Log display
        self.log_display = scrolledtext.ScrolledText(bottom_frame, height=8, font=("Arial", 14))
        self.log_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_display.config(state=tk.DISABLED)

    def setup_graph_tab(self):
        # Map/Plot frame
        map_frame = ttk.Frame(self.graph_tab, relief=tk.RAISED, borderwidth=2)
        map_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(map_frame, text="DRONE TRAJECTORY", style="Header.TLabel").pack(pady=5)
        
        self.fig = Figure(figsize=(6, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('X Position (m)', fontsize=14)
        self.ax.set_ylabel('Y Position (m)', fontsize=14)
        self.ax.tick_params(axis='both', which='major', labelsize=12)
        self.ax.grid(True)
        self.ax.set_title('Drone Path', fontsize=16)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=map_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def setup_network(self):
        self.server_address = ('172.20.10.2', 12345)  # Same as in pyscript.py
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.bind(self.server_address)
        self.server_socket.settimeout(1.0)  # 1 second timeout for non-blocking

    def receive_data(self):
        while self.running:
            try:
                data, addr = self.server_socket.recvfrom(1024)
                message = data.decode()
                self.process_message(message)
            except socket.timeout:
                # Check if connection is lost (no messages for 5 seconds)
                if time.time() - self.last_update_time > 5:
                    self.connection_status.config(text="DISCONNECTED", foreground="red")
            except Exception as e:
                self.log_message(f"Error: {str(e)}")

    def process_message(self, message=None):
        try:
            # If message is not provided, try to receive one
            if message is None:
                try:
                    data, addr = self.server_socket.recvfrom(1024)
                    message = data.decode()
                except socket.timeout:
                    return
                except Exception as e:
                    self.log_message(f"Error receiving message: {str(e)}")
                    return
            
            # Update last update time
            self.last_update_time = time.time()
            self.last_update.config(text=time.strftime("%H:%M:%S"))
            self.connection_status.config(text="CONNECTED", foreground="green")
            
            # Position message
            if message.startswith("Position:"):
                match = re.search(r"X=(-?\d+\.\d+)m, Y=(-?\d+\.\d+)m", message)
                if match:
                    self.position_x = float(match.group(1))
                    self.position_y = float(match.group(2))
                    
                    # Add to history for plotting
                    self.position_history_x.append(self.position_x)
                    self.position_history_y.append(self.position_y)
                    self.time_history.append(time.time() - self.start_time)
                    
                    # Keep history to a reasonable size
                    if len(self.position_history_x) > 100:
                        self.position_history_x.pop(0)
                        self.position_history_y.pop(0)
                        self.time_history.pop(0)
            
            # Target message
            elif message.startswith("Target:"):
                match = re.search(r"Target: (\d+\.\d+)m at (-?\d+\.\d+)° \| Distance remaining: (\d+\.\d+)m \| Heading needed: (-?\d+\.\d+)°", message)
                if match and not self.target_locked:
                    self.target_distance = float(match.group(1))
                    self.target_heading = float(match.group(2))
                    self.distance_remaining = float(match.group(3))
                    self.heading_needed = float(match.group(4))
                elif match and self.target_locked:
                    self.target_distance = self.locked_target_distance
                    self.target_heading = self.locked_target_heading
                    self.distance_remaining = self.locked_distance_remaining
                    self.heading_needed = self.locked_heading_needed
            
            # Straight-line distance message
            elif message.startswith("Straight-line distance"):
                match = re.search(r"Straight-line distance to target: (\d+\.\d+)m", message)
                if match and not self.target_locked:
                    self.straight_line_distance = float(match.group(1))
                elif match and self.target_locked:
                    self.straight_line_distance = self.locked_straight_line_distance
            
            # Alert messages
            elif "IMPORTANT" in message or "scream detected" in message.lower() or "person found" in message.lower():
                self.log_message(f"ALERT: {message}", is_alert=True)
                self.alert_active = True
                self.last_alert_time = time.time()
                if not self.target_locked:
                    self.locked_target_distance = self.target_distance
                    self.locked_target_heading = self.target_heading
                    self.locked_distance_remaining = self.distance_remaining
                    self.locked_heading_needed = self.heading_needed
                    self.locked_straight_line_distance = self.straight_line_distance
                    self.target_locked = True
            
            # Other messages
            else:
                self.log_message(message)
                
        except Exception as e:
            self.log_message(f"Error processing message: {str(e)}")

    def log_message(self, message, is_alert=False):
        timestamp = time.strftime("%H:%M:%S")
        
        self.log_display.config(state=tk.NORMAL)
        if is_alert:
            self.log_display.insert(tk.END, f"[{timestamp}] {message}\n", "alert")
            self.log_display.tag_config("alert", foreground="red", font=("Arial", 14, "bold"))
        else:
            self.log_display.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_display.see(tk.END)
        self.log_display.config(state=tk.DISABLED)

    def update_ui(self):
        if not self.running:
            return
            
        # Update position display
        self.x_coord_label.config(text=f"{self.position_x:.2f} m")
        self.y_coord_label.config(text=f"{self.position_y:.2f} m")
        
        # Update navigation data
        self.target_dist_label.config(text=f"{self.target_distance:.2f} m")
        self.target_head_label.config(text=f"{self.target_heading:.1f}°")
        self.dist_remain_label.config(text=f"{self.distance_remaining:.2f} m")
        self.head_needed_label.config(text=f"{self.heading_needed:.1f}°")
        self.straight_dist_label.config(text=f"{self.straight_line_distance:.2f} m")
        
        # Update plot
        if self.position_history_x and self.position_history_y:
            self.ax.clear()
            # Plot the trajectory path (excluding the first point)
            if len(self.position_history_x) > 1:
                self.ax.plot(self.position_history_x, self.position_history_y, 'b-', marker='o')
            # Plot the starting point (0,0) in green
            self.ax.plot(self.position_history_x[0], self.position_history_y[0], 'go', markersize=10, label='Start (0,0)')
            # Plot the current position in red
            self.ax.plot(self.position_history_x[-1], self.position_history_y[-1], 'ro', markersize=8, label='Current Position')
            self.ax.set_xlabel('X Position (m)', fontsize=14)
            self.ax.set_ylabel('Y Position (m)', fontsize=14)
            self.ax.tick_params(axis='both', which='major', labelsize=12)
            self.ax.grid(True)
            self.ax.set_title('Drone Path', fontsize=16)
            self.ax.legend()
            self.canvas.draw()
        
        # Handle alert indicator
        if self.alert_active:
            if time.time() - self.last_alert_time < 10:  # Blink for 10 seconds
                # Blink effect
                if int(time.time() * 2) % 2:  # Toggle every 0.5 seconds
                    self.alert_indicator.config(text="⚠️ ALERT ACTIVE ⚠️", foreground="red")
                else:
                    self.alert_indicator.config(text="⚠️ ALERT ACTIVE ⚠️", foreground="orange")
            else:
                self.alert_active = False
                self.alert_indicator.config(text="NO ALERTS", foreground="green")
        
        # Process any pending messages
        self.process_message()
        
        # Schedule next update
        self.root.after(100, self.update_ui)

    def on_closing(self):
        self.running = False
        if hasattr(self, 'server_socket'):
            self.server_socket.close()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = DroneMonitorUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()


