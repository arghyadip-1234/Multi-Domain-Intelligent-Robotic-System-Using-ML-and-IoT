import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import os
import cv2
import threading
import time
import datetime
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import serial
import sys
import requests
import numpy as np
import webbrowser  # Import the webbrowser module

# === Global Variables ===
camera_thread = None
camera_running = False
camera_label = None  # This will now only be used for Farming Bot's ESP32 stream
captured_image_label = None  # This will now only be used for Farming Bot's captured image
latest_frame = None  # For saving/capturing current frame
handcam_thread = None
ESP32_CAM_URL = "http://192.168.29.46/"  # Replace with your ESP32-CAM IP

# Define your Streamlit application URL here
STREAMLIT_URL = "http://localhost:8501"

# Global for bot arm stream label (Updated to be global)
bot_stream_img_label = None

# === Hand Tracking Section ===

# Configuration
WRITE_VIDEO = True
DEBUG = True  # Set to False when using real Arduino
CAM_SOURCE = 0

if not DEBUG:
    try:
        ser = serial.Serial('COM5', 115200)
    except serial.SerialException as e:
        print(f"Could not open serial port COM5: {e}")
        DEBUG = True

# Servo ranges
X_MIN, X_MID, X_MAX = 0, 75, 150
PALM_ANGLE_MIN, PALM_ANGLE_MID = -50, 20
Y_MIN, Y_MID, Y_MAX = 0, 90, 180
WRIST_Y_MIN, WRIST_Y_MAX = 0.3, 0.9
Z_MIN, Z_MID, Z_MAX = 10, 90, 180
PALM_SIZE_MIN, PALM_SIZE_MAX = 0.1, 0.3
CLAW_OPEN_ANGLE, CLAW_CLOSE_ANGLE = 50, 180

# Right hand
X1_MIN, X1_MID, X1_MAX = 0, 75, 150
PALM1_ANGLE_MIN, PALM1_ANGLE_MID = -50, 20
Y1_MIN, Y1_MID, Y1_MAX = 0, 90, 180
WRIST1_Y_MIN, WRIST1_Y_MAX = 0.3, 0.9
Z1_MIN, Z1_MID, Z1_MAX = 10, 90, 180
PALM1_SIZE_MIN, PALM1_SIZE_MAX = 0.1, 0.3
CLAW1_OPEN_ANGLE, CLAW1_CLOSE_ANGLE = 50, 180

# Initial angles
servo_angle = [X_MID, Y_MID, Z_MID, CLAW_OPEN_ANGLE]
prev_servo_angle = servo_angle.copy()
FIST_THRESHOLD = 7

servo1_angle = [X1_MID, Y1_MID, Z1_MID, CLAW1_OPEN_ANGLE]
prev1_servo_angle = servo1_angle.copy()
FIST1_THRESHOLD = 7

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Stop flag for thread control
stop_handcam = False

def clamp(value, min_value, max_value):
    return max(min(max_value, value), min_value)

def map_range(value, in_min, in_max, out_min, out_max):
    if in_min == in_max:
        return out_min if value <= in_min else out_max
    return abs(int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min))

def is_fist(hand_landmarks, palm_size):
    WRIST = hand_landmarks.landmark[0]
    distance_sum = sum(
        ((WRIST.x - hand_landmarks.landmark[i].x) ** 2 +
         (WRIST.y - hand_landmarks.landmark[i].y) ** 2 +
         (WRIST.z - hand_landmarks.landmark[i].z) ** 2) ** 0.5
        for i in [7, 8, 11, 12, 15, 16, 19, 20]
    )
    if palm_size == 0:
        return False
    return distance_sum / palm_size < FIST_THRESHOLD

def landmark_to_servo_angle(hand_landmarks):
    WRIST = hand_landmarks.landmark[0]
    MCP = hand_landmarks.landmark[5]
    palm_size = ((WRIST.x - MCP.x)*2 + (WRIST.y - MCP.y)*2 + (WRIST.z - MCP.z)*2)*0.5

    angles = [X_MID, Y_MID, Z_MID, CLAW_OPEN_ANGLE]
    angles[3] = CLAW_CLOSE_ANGLE if is_fist(hand_landmarks, palm_size) else CLAW_OPEN_ANGLE

    if palm_size != 0:
        angle = (WRIST.x - MCP.x) / palm_size
        angle = int(angle * 180 / 3.1415926)
    else:
        angle = PALM_ANGLE_MID

    angle = clamp(angle, PALM_ANGLE_MIN, PALM_ANGLE_MID)
    angles[0] = map_range(angle, PALM_ANGLE_MIN, PALM_ANGLE_MID, X_MAX, X_MIN)

    wrist_y = clamp(WRIST.y, WRIST_Y_MIN, WRIST_Y_MAX)
    angles[1] = map_range(wrist_y, WRIST_Y_MIN, WRIST_Y_MAX, Y_MAX, Y_MIN)

    palm_size = clamp(palm_size, PALM_SIZE_MIN, PALM_SIZE_MAX)
    angles[2] = map_range(palm_size, PALM_SIZE_MIN, PALM_SIZE_MAX, Z_MAX, Z_MIN)

    return [int(i) for i in angles]

def is_fist1(hand_landmarks1, palm_size1):
    WRIST1 = hand_landmarks1.landmark[0]
    distance_sum1 = sum(
        ((WRIST1.x - hand_landmarks1.landmark[i].x) ** 2 +
         (WRIST1.y - hand_landmarks1.landmark[i].y) ** 2 +
         (WRIST1.z - hand_landmarks1.landmark[i].z) ** 2) ** 0.5
        for i in [7, 8, 11, 12, 15, 16, 19, 20]
    )
    if palm_size1 == 0:
        return False
    return distance_sum1 / palm_size1 < FIST1_THRESHOLD

def landmark_to_servo_angle1(hand_landmarks1):
    WRIST1 = hand_landmarks1.landmark[0]
    MCP1 = hand_landmarks1.landmark[5]
    palm_size1 = ((WRIST1.x - MCP1.x)*2 + (WRIST1.y - MCP1.y)*2 + (WRIST1.z - MCP1.z)*2)*0.5

    angles1 = [X1_MID, Y1_MID, Z1_MID, CLAW1_OPEN_ANGLE]
    angles1[3] = CLAW1_CLOSE_ANGLE if is_fist1(hand_landmarks1, palm_size1) else CLAW1_OPEN_ANGLE

    if palm_size1 != 0:
        angle1 = (WRIST1.x - MCP1.x) / palm_size1
        angle1 = int(angle1 * 180 / 3.1415926)
    else:
        angle1 = PALM1_ANGLE_MID

    angle1 = clamp(angle1, PALM1_ANGLE_MIN, PALM1_ANGLE_MID)
    angles1[0] = map_range(angle1, PALM1_ANGLE_MIN, PALM1_ANGLE_MID, X1_MIN, X1_MAX)

    wrist_y1 = clamp(WRIST1.y, WRIST1_Y_MIN, WRIST1_Y_MAX)
    angles1[1] = map_range(wrist_y1, WRIST1_Y_MIN, WRIST1_Y_MAX, Y1_MAX, Y1_MIN)

    palm_size1 = clamp(palm_size1, PALM1_SIZE_MIN, PALM1_SIZE_MAX)
    angles1[2] = map_range(palm_size1, PALM1_SIZE_MIN, PALM1_SIZE_MAX, Z1_MAX, Z1_MIN)

    return [int(i) for i in angles1]

def run_hand_tracking():
    global stop_handcam, servo_angle, prev_servo_angle, servo1_angle, prev1_servo_angle, bot_stream_img_label
    cap = cv2.VideoCapture(CAM_SOURCE)
    if WRITE_VIDEO:
        out = None
    else:
        out = None

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened() and not stop_handcam:
            success, image = cap.read()
            if not success:
                print("Empty camera frame or camera not available.")
                cap.release()
                cap = cv2.VideoCapture(CAM_SOURCE)
                time.sleep(1)
                continue

            if WRITE_VIDEO and out is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                height, width, _ = image.shape
                out = cv2.VideoWriter('output.avi', fourcc, 30.0, (width, height))

            image = cv2.flip(image, 1)
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = MessageToDict(handedness)['classification'][0]['label']

                    if label == 'Left':
                        cv2.putText(image, label + ' Hand', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
                        servo_angle = landmark_to_servo_angle(hand_landmarks)
                    elif label == 'Right':
                        cv2.putText(image, label + ' Hand', (460, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
                        servo1_angle = landmark_to_servo_angle1(hand_landmarks)

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            if servo_angle != prev_servo_angle or servo1_angle != prev1_servo_angle:
                print("Left:", servo_angle, "Right:", servo1_angle)
                prev_servo_angle = servo_angle.copy()
                prev1_servo_angle = servo1_angle.copy()
                if not DEBUG:
                    try:
                        if ser.is_open:
                            ser.write(bytearray(servo_angle + servo1_angle))
                        else:
                            print("Serial port is not open. Skipping serial write.")
                    except serial.SerialException as e:
                        print(f"Serial write error: {e}")

            if bot_stream_img_label and bot_stream_img_label.winfo_exists():
                label_width = bot_stream_img_label.winfo_width()
                label_height = bot_stream_img_label.winfo_height()
                if label_width < 100 or label_height < 100:
                    label_width, label_height = 500, 350  # Make bigger
                original_height, original_width, _ = image.shape
                aspect_ratio = original_width / original_height
                if label_width / aspect_ratio <= label_height:
                    new_width = label_width
                    new_height = int(label_width / aspect_ratio)
                else:
                    new_height = label_height
                    new_width = int(label_height * aspect_ratio)
                new_width = max(1, new_width)
                new_height = max(1, new_height)
                img_resized = cv2.resize(image, (new_width, new_height))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(img_pil)
                def update_label_in_main_thread():
                    if bot_stream_img_label and bot_stream_img_label.winfo_exists():
                        bot_stream_img_label.imgtk = imgtk
                        bot_stream_img_label.configure(image=imgtk)
                bot_stream_img_label.after(0, update_label_in_main_thread)

            if WRITE_VIDEO and out is not None:
                out.write(image)

            if stop_handcam:
                break

    if WRITE_VIDEO and out is not None:
        out.release()
    cap.release()
    if bot_stream_img_label and bot_stream_img_label.winfo_exists():
        bot_stream_img_label.after(0, lambda: bot_stream_img_label.configure(image=None))
    cv2.destroyAllWindows()

def start_handcam():
    global stop_handcam, handcam_thread
    if handcam_thread and handcam_thread.is_alive():
        print("Handcam thread is already running.")
        return
    stop_handcam = False
    handcam_thread = threading.Thread(target=run_hand_tracking, daemon=True)
    handcam_thread.start()
    return handcam_thread

def stop_handcam_func():
    global stop_handcam, handcam_thread
    if handcam_thread and handcam_thread.is_alive():
        stop_handcam = True
        handcam_thread.join(timeout=1.0)
        if handcam_thread.is_alive():
            print("Handcam thread did not terminate in time.")
        else:
            print("Handcam thread stopped.")
    else:
        print("Handcam thread is not running.")

def load_image(path, size=(120, 120)):
    try:
        if not os.path.exists(path):
            print(f"Image not found: {path}")
            return None
        img = Image.open(path)
        img = img.resize(size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

root = tk.Tk()
root.title("Bot Selector")
root.geometry("1100x650")

# Update the paths below if your images are in a subfolder, e.g., "images/FarmBot.png"
farming_img = load_image("FarmBot.png")
surgery_img = load_image("surgery.png")
industrial_img = load_image("industry.png")

if not all([farming_img, surgery_img, industrial_img]):
    messagebox.showerror("Image Load Error", "One or more bot images could not be loaded. Please check paths.")

global_images = {
    "farming": farming_img,
    "surgery": surgery_img,
    "industrial": industrial_img
}

canvas = tk.Canvas(root, width=1100, height=650)
canvas.pack(fill="both", expand=True)

def draw_vertical_gradient(canvas, color1, color2):
    width, height = 1100, 650
    gradient = Image.new("RGB", (1, height), "#000")
    draw = ImageDraw.Draw(gradient)
    r1, g1, b1 = canvas.winfo_rgb(color1)
    r2, g2, b2 = canvas.winfo_rgb(color2)
    r1, g1, b1 = [val >> 8 for val in (r1, g1, b1)]
    r2, g2, b2 = [val >> 8 for val in (r2, g2, b2)]

    for y in range(height):
        ratio = y / height
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        draw.point((0, y), fill=(r, g, b))

    gradient = gradient.resize((width, height), Image.Resampling.LANCZOS)
    bg_img = ImageTk.PhotoImage(gradient)
    canvas.create_image(0, 0, anchor="nw", image=bg_img)
    canvas.bg_img = bg_img

draw_vertical_gradient(canvas, "#e3f2fd", "#90caf9")

def start_esp32_cam_stream(esp32_url=ESP32_CAM_URL):
    global camera_running, camera_thread, latest_frame, camera_label
    if camera_label is None or not camera_label.winfo_exists():
        print("ESP32 Live Cam label is not available. This stream is only for Farming Bot.")
        return

    if camera_thread and camera_thread.is_alive():
        print("ESP32 camera stream is already running.")
        return

    def stream():
        global camera_running, latest_frame, camera_label
        bytes_data = bytes()
        stream_request = None
        try:
            stream_request = requests.get(esp32_url, stream=True, timeout=5)
            for chunk in stream_request.iter_content(chunk_size=1024):
                if not camera_running:
                    break
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    img_np = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    if frame is not None:
                        latest_frame = frame.copy()
                        fixed_width, fixed_height = 300, 200
                        frame_resized = cv2.resize(frame, (fixed_width, fixed_height))
                        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                        imgtk = ImageTk.PhotoImage(image=img)
                        if camera_label and camera_label.winfo_exists():
                            camera_label.imgtk = imgtk
                            camera_label.configure(image=imgtk)
                    time.sleep(0.01)
        except requests.exceptions.ConnectionError as ce:
            print(f"ESP32 stream connection error: {ce}. Is the ESP32-CAM on and reachable at {esp32_url}?")
            messagebox.showerror("Connection Error", f"Could not connect to ESP32-CAM at {esp32_url}. Please ensure it's powered on and on the same network.\nError: {ce}")
        except Exception as e:
            print(f"ESP32 stream error: {e}")
            messagebox.showerror("Stream Error", f"An error occurred during ESP32 stream: {e}")
        finally:
            if stream_request:
                stream_request.close()
            if camera_label and camera_label.winfo_exists():
                camera_label.after(0, lambda: camera_label.configure(image=None))
            print("ESP32 stream stopped.")

    camera_running = True
    camera_thread = threading.Thread(target=stream, daemon=True)
    camera_thread.start()

def stop_esp32_cam_stream():
    global camera_running, camera_thread
    if camera_thread and camera_thread.is_alive():
        camera_running = False
        camera_thread.join(timeout=1.0)
        if camera_thread.is_alive():
            print("ESP32 camera thread did not terminate in time.")
        else:
            print("ESP32 camera thread stopped.")
    else:
        print("ESP32 camera stream is not running.")
    if camera_label and camera_label.winfo_exists():
        camera_label.after(0, lambda: camera_label.configure(image=None))

def capture_image():
    global latest_frame, captured_image_label
    if captured_image_label is None or not captured_image_label.winfo_exists():
        print("Captured Image label is not available. This feature is only for Farming Bot.")
        return

    if latest_frame is not None:
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads", "CapturedImages")
        os.makedirs(downloads_dir, exist_ok=True)
        filename = f"captured_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(downloads_dir, filename)
        try:
            cv2.imwrite(filepath, latest_frame)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image: {e}")
            print(f"Error saving image: {e}")
            return

        img_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        label_width = captured_image_label.winfo_width()
        label_height = captured_image_label.winfo_height()
        if label_width < 100 or label_height < 100:
            label_width, label_height = 300, 200
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height
        if label_width / aspect_ratio <= label_height:
            new_width = label_width
            new_height = int(label_width / aspect_ratio)
        else:
            new_height = label_height
            new_width = int(label_height * aspect_ratio)
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(img)
        if captured_image_label and captured_image_label.winfo_exists():
            captured_image_label.imgtk = imgtk
            captured_image_label.configure(image=imgtk)
        print(f"Image captured and saved to {filepath}")
        messagebox.showinfo("Image Captured", f"Image saved to: {filepath}")
    else:
        print("No frame available to capture.")
        messagebox.showwarning("No Frame", "No live camera frame available to capture.")

def open_streamlit_website():
    try:
        webbrowser.open(STREAMLIT_URL)
        print(f"Opened Streamlit website: {STREAMLIT_URL}")
        messagebox.showinfo("Website Opened", f"Your Streamlit website should now be open in your default browser at {STREAMLIT_URL}")
    except Exception as e:
        print(f"Error opening Streamlit website: {e}")
        messagebox.showerror("Error", f"Could not open website. Please ensure your Streamlit app is running at {STREAMLIT_URL} and you have a web browser installed.\nError: {e}")

def add_tooltip(widget, text):
    tooltip = tk.Toplevel(widget)
    tooltip.withdraw()
    tooltip.overrideredirect(True)
    label = tk.Label(tooltip, text=text, background="#263238", foreground="white",
                     font=("Segoe UI", 9), padx=8, pady=4, borderwidth=1, relief="solid")
    label.pack()
    def enter(event):
        x = event.x_root + 10
        y = event.x_root + 10
        tooltip.geometry(f"+{x}+{y}")
        tooltip.deiconify()
    def leave(event):
        tooltip.withdraw()
    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)

def style_modern_buttons():
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("Modern.TButton",
                    font=("Segoe UI", 11, "bold"),
                    padding=8,
                    borderwidth=0,
                    background="#222",
                    foreground="white")
    style.map("Modern.TButton",
              background=[("active", "#444")],
              foreground=[("active", "white")])

style_modern_buttons()

def setup_farming_bot_ui(parent_frame, title):
    global camera_label, captured_image_label, bot_stream_img_label

    header_font = ("Segoe UI", 24, "bold")
    label_font = ("Segoe UI", 13)
    button_font = ("Segoe UI", 11, "bold")

    tk.Label(parent_frame, text=title, font=header_font, bg="#F5F5F5", fg="#1976D2").pack(pady=10)
    content_frame = tk.Frame(parent_frame, bg="#F5F5F5")
    content_frame.pack(expand=True, fill="both", padx=10, pady=10)

    controls_frame = tk.Frame(content_frame, width=170, bg="#FFFFFF", relief="ridge", bd=2, highlightbackground="#90caf9", highlightthickness=2)
    controls_frame.pack(side="left", fill="y", padx=10, pady=10)
    tk.Label(controls_frame, text="Controls", font=("Segoe UI", 15, "bold"), bg="#FFFFFF", fg="#1976D2").pack(pady=(15, 10))

    status_var = tk.StringVar(value="Ready")

    def dummy_action(name):
        global handcam_thread
        status_var.set(f"{name.replace('_', ' ').capitalize()} clicked")
        if name == "start cam":
            start_esp32_cam_stream()
        elif name == "stop cam":
            stop_esp32_cam_stream()
        elif name == "capture img":
            capture_image()
        elif name == "start handcam":
            if handcam_thread is None or not handcam_thread.is_alive():
                handcam_thread = start_handcam()
        elif name == "stop handcam":
            stop_handcam_func()
        else:
            print(f"{name} clicked")

    btn_info = [
        ("start cam", "Start ESP32 camera stream"),
        ("stop cam", "Stop ESP32 camera stream"),
        ("capture img", "Capture image from ESP32"),
        ("start handcam", "Start hand gesture control"),
        ("stop handcam", "Stop hand gesture control"),
    ]
    for text, tip in btn_info:
        btn = ttk.Button(controls_frame, text=text.title(), style="Modern.TButton", command=lambda t=text: dummy_action(t))
        btn.pack(pady=7, padx=18, fill="x")
        add_tooltip(btn, tip)

    center_frame = tk.Frame(content_frame, bg="#E3F2FD", highlightbackground="#90caf9", highlightthickness=2)
    center_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    bot_stream_frame = tk.Frame(center_frame, bg="#B3E5FC", width=500, height=350, relief="ridge", bd=2)
    bot_stream_frame.pack(padx=10, pady=10)
    bot_stream_frame.pack_propagate(False)
    tk.Label(bot_stream_frame, text="Bot Arm Stream", font=("Segoe UI", 18, "bold"), bg="#B3E5FC", fg="#1976D2").pack(pady=5)
    bot_stream_img_label = tk.Label(bot_stream_frame, bg="#B3E5FC", width=500, height=350)
    bot_stream_img_label.pack(expand=False, padx=10, pady=10)

    website_frame = tk.Frame(center_frame, height=100, bg="#E1F5FE", relief="ridge", bd=2)
    website_frame.pack(fill="both", expand=True, padx=10, pady=10)
    website_btn = ttk.Button(website_frame, text="Go to Website", style="Modern.TButton", command=open_streamlit_website)
    website_btn.pack(pady=20)
    add_tooltip(website_btn, "Open the Streamlit website for model prediction")

    right_section = tk.Frame(content_frame, width=300, bg="#FFFFFF", relief="ridge", bd=2, highlightbackground="#90caf9", highlightthickness=2)
    right_section.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    camera_frame = tk.Frame(right_section, height=150, bg="#FFECB3", relief="ridge", bd=2)
    camera_frame.pack(fill="both", expand=True, padx=10, pady=10)
    tk.Label(camera_frame, text="ESP32 Live Cam", font=("Segoe UI", 16, "bold"), bg="#FFECB3", fg="#FF6F00").pack()
    camera_img_label = tk.Label(camera_frame, bg="#FFECB3")
    camera_img_label.pack(expand=True, padx=10, pady=10)
    camera_label = camera_img_label

    picture_frame = tk.Frame(right_section, height=150, bg="#FFE0B2", relief="ridge", bd=2)
    picture_frame.pack(fill="both", expand=True, padx=10, pady=10)
    tk.Label(picture_frame, text="Captured Image", font=("Segoe UI", 14, "bold"), bg="#FFE0B2", fg="#F57C00").pack()
    captured_img_label = tk.Label(picture_frame, bg="#FFE0B2")
    captured_img_label.pack(expand=True, padx=10, pady=10)
    captured_image_label = captured_img_label

    status_bar = tk.Label(parent_frame, textvariable=status_var, bd=1, relief="sunken", anchor="w", bg="#E3F2FD", font=("Segoe UI", 10))
    status_bar.pack(side="bottom", fill="x")

    def on_close():
        stop_esp32_cam_stream()
        stop_handcam_func()
        parent_frame.destroy()

    parent_frame.protocol("WM_DELETE_WINDOW", on_close)

def setup_simplified_bot_ui(parent_frame, title):
    global bot_stream_img_label
    global camera_label, captured_image_label
    camera_label = None
    captured_image_label = None

    header_font = ("Segoe UI", 24, "bold")
    label_font = ("Segoe UI", 13)
    button_font = ("Segoe UI", 11, "bold")

    tk.Label(parent_frame, text=title, font=header_font, bg="#F5F5F5", fg="#1976D2").pack(pady=10)
    content_frame = tk.Frame(parent_frame, bg="#F5F5F5")
    content_frame.pack(expand=True, fill="both", padx=10, pady=10)

    controls_frame = tk.Frame(content_frame, width=170, bg="#FFFFFF", relief="ridge", bd=2, highlightbackground="#90caf9", highlightthickness=2)
    controls_frame.pack(side="left", fill="y", padx=10, pady=10)
    tk.Label(controls_frame, text="Controls", font=("Segoe UI", 15, "bold"), bg="#FFFFFF", fg="#1976D2").pack(pady=(15, 10))

    status_var = tk.StringVar(value="Ready")

    def dummy_action(name):
        global handcam_thread
        status_var.set(f"{name.replace('_', ' ').capitalize()} clicked")
        if name == "start handcam":
            if handcam_thread is None or not handcam_thread.is_alive():
                handcam_thread = start_handcam()
        elif name == "stop handcam":
            stop_handcam_func()
        else:
            print(f"{name} clicked")

    btn_info = [
        ("start handcam", "Start hand gesture control for bot arm"),
        ("stop handcam", "Stop hand gesture control"),
    ]
    for text, tip in btn_info:
        btn = ttk.Button(controls_frame, text=text.title(), style="Modern.TButton", command=lambda t=text: dummy_action(t))
        btn.pack(pady=7, padx=18, fill="x")
        add_tooltip(btn, tip)

    bot_stream_frame = tk.Frame(content_frame, bg="#B3E5FC", relief="ridge", bd=2, highlightbackground="#90caf9", highlightthickness=2)
    bot_stream_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    tk.Label(bot_stream_frame, text="Bot Arm Stream", font=("Segoe UI", 18, "bold"), bg="#B3E5FC", fg="#1976D2").pack(pady=5)
    bot_stream_img_label = tk.Label(bot_stream_frame, bg="#B3E5FC")
    bot_stream_img_label.pack(expand=True, padx=10, pady=10)

    status_bar = tk.Label(parent_frame, textvariable=status_var, bd=1, relief="sunken", anchor="w", bg="#E3F2FD", font=("Segoe UI", 10))
    status_bar.pack(side="bottom", fill="x")

    def on_close():
        stop_handcam_func()
        parent_frame.destroy()

    parent_frame.protocol("WM_DELETE_WINDOW", on_close)

def open_new_page(title):
    new_window = tk.Toplevel(root)
    new_window.title(title)
    new_window.geometry("1000x600")
    new_window.configure(bg="#F5F5F5")
    new_window.protocol("WM_DELETE_WINDOW", lambda: on_new_window_close(new_window, title))
    if title == "Farming Bot":
        setup_farming_bot_ui(new_window, title)
    else:
        setup_simplified_bot_ui(new_window, title)

def on_new_window_close(window, title):
    if title == "Farming Bot":
        stop_esp32_cam_stream()
    stop_handcam_func()
    window.destroy()

container = tk.Frame(canvas, bg="#FFFFFF")
canvas.create_window(550, 325, window=container, anchor="center")

def create_bot_section(parent, img, name):
    frame = tk.Frame(parent, bg="white", bd=0, relief="flat", highlightbackground="#CFD8DC", highlightthickness=2)
    frame.grid_propagate(False)
    frame.configure(width=300, height=320)
    button = tk.Button(
        frame,
        text=name,
        image=img,
        compound="top",
        font=("Segoe UI", 14, "bold"),
        fg="white",
        bg="#222",
        activebackground="#444",
        activeforeground="white",
        relief="flat",
        padx=20,
        pady=20,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: open_new_page(name)
    )
    button.image = img
    button.pack(expand=True, fill="both", padx=20, pady=20)
    return frame

create_bot_section(container, farming_img, "Farming Bot").grid(row=0, column=0, padx=40)
create_bot_section(container, surgery_img, "Surgery Bot").grid(row=0, column=1, padx=40)
create_bot_section(container, industrial_img, "Industrial Bot").grid(row=0, column=2, padx=40)

root.mainloop()