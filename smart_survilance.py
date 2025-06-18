from email.message import EmailMessage
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import smtplib
import ssl
from email.message import EmailMessage
import sqlite3
import hashlib
import re
# Email credentials
FROM_EMAIL = "#email"
EMAIL_PASSWORD = "#email_app_password"

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  password TEXT, 
                  email TEXT,
                  registration_date TEXT)''')
    conn.commit()
    conn.close()

    init_activity_logs()  # Initialize activity logs table

def init_activity_logs():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    
    # Create the table with the correct schema
    c.execute('''CREATE TABLE IF NOT EXISTS activity_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    timestamp TEXT,
                    confidence REAL,
                    frame BLOB
                )''')
    conn.commit()
    conn.close()

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# User authentication system
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'user' not in st.session_state:
    st.session_state['user'] = None

def register():
    st.subheader("Register")
    username = st.text_input("Username", key="reg_username")
    password = st.text_input("Password", type="password", key="reg_password")
    email = st.text_input("Email", key="reg_email")
    
    if st.button("Register", key="reg_button"):
        # Validate email format
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, email):
            st.error("Please enter a valid email address!")
            return
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Check if username exists
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        if c.fetchone() is not None:
            st.warning("Username already taken!")
        else:
            # Hash password
            hashed_pw = hash_password(password)
            registration_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Insert new user
            c.execute("INSERT INTO users VALUES (?, ?, ?, ?)",
                     (username, hashed_pw, email, registration_time))
            conn.commit()
            st.success("Registration successful! Please login.")
        conn.close()

def login():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login", key="login_button"):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Get user data
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user_data = c.fetchone()
        conn.close()
        
        if user_data and user_data[1] == hash_password(password):
            st.session_state['authenticated'] = True
            st.session_state['user'] = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials!")

def get_user_email(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE username=?", (username,))
    email = c.fetchone()[0]
    conn.close()
    return email

def logout():
    st.session_state['authenticated'] = False
    st.session_state['user'] = None
    st.rerun()

class AnomalyDetector:
    def __init__(self, model_path='best_anomaly_detector.h5', image_size=(64, 64)):
        self.model = load_model(model_path)
        self.image_size = image_size
        # Lower threshold for more sensitive detection
        self.threshold = 0.2  # Confidence threshold for anomaly detection
        
    def preprocess_frame(self, frame):
        # Resize frame to model input size
        frame = cv2.resize(frame, self.image_size)
        
        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # Normalize pixel values
        frame = frame.astype('float32') / 255.0
        
        # Add channel dimension
        frame = np.expand_dims(frame, axis=-1)
        
        # Add batch dimension
        frame = np.expand_dims(frame, axis=0)
        
        return frame
        
    def detect_anomaly(self, frame):
        processed_frame = self.preprocess_frame(frame)
        prediction = self.model.predict(processed_frame)[0][0]
        
        # Return prediction value along with binary classification
        return prediction > self.threshold, prediction

def send_email(to_email, frame, confidence):
    msg = EmailMessage()
    msg['Subject'] = "Anomaly Detected Alert"
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email
    msg.set_content(f"An anomaly was detected with confidence: {confidence:.2f}")
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, img_encoded = cv2.imencode('.jpg', frame_rgb)
    msg.add_attachment(img_encoded.tobytes(), maintype='image', subtype='jpeg', filename='anomaly.jpg')
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(FROM_EMAIL, EMAIL_PASSWORD)
        smtp.send_message(msg)

def process_image(image_file, detector):
    """
    Process an uploaded image and display results
    """
    # Read image
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
    
    # Detect anomaly
    is_anomaly, confidence = detector.detect_anomaly(img)
    
    # Draw results
    if is_anomaly:
        cv2.putText(img, f"ANOMALY DETECTED! (Confidence: {confidence:.2f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(img, "NORMAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img_rgb, is_anomaly, confidence

def process_video(video_file, detector):
    """
    Process an uploaded video and display results
    """
    # Save video to temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return None, None, None, None
        
    frame_count = 0
    anomaly_count = 0
    any_anomaly_detected = False
    highest_confidence = 0
    best_frame = None  # Store the frame with the highest confidence

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Detect anomaly
        is_anomaly, confidence = detector.detect_anomaly(frame)
        
        if is_anomaly:
            anomaly_count += 1
            any_anomaly_detected = True
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_frame = frame.copy()  # Save the frame with the highest confidence
        else:
            any_anomaly_detected = False

    cap.release()
    
    # Calculate statistics
    anomaly_rate = (anomaly_count / frame_count) * 100 if frame_count > 0 else 0
    
    # Create statistics dataframe
    stats = pd.DataFrame({
        'Metric': ['Total Frames', 'Anomalies Detected', 'Anomaly Rate'],
        'Value': [frame_count, anomaly_count, f'{anomaly_rate:.2f}%']
    })
    
    # Determine final status based on any anomaly detection
    if any_anomaly_detected:
        final_status = "ANOMALY DETECTED!"
        final_confidence = highest_confidence
    else:
        final_status = "NORMAL"
        final_confidence = 0

    # Convert the best frame to RGB for display
    if best_frame is not None:
        best_frame_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    else:
        best_frame_rgb = None

    return stats, final_status, final_confidence, best_frame_rgb, anomaly_rate

def log_activity(username, confidence, frame):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 🚨 Make sure frame is in BGR format before saving
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Encode the frame as a binary blob
    _, img_encoded = cv2.imencode('.jpg', frame_bgr)
    frame_blob = img_encoded.tobytes()
    
    c.execute('''INSERT INTO activity_logs (username, timestamp, confidence, frame)
                 VALUES (?, ?, ?, ?)''', (username, timestamp, float(confidence), frame_blob))  # Ensure confidence is stored as float
    conn.commit()
    conn.close()
    
    return frame  # Return the frame for display

def fetch_activity_logs(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Fetch id also
    c.execute('''SELECT id, timestamp, CAST(confidence AS REAL), frame FROM activity_logs 
                 WHERE username = ? 
                 ORDER BY timestamp DESC''', (username,))
    logs = c.fetchall()
    conn.close()
    
    return logs


# Initialize Streamlit app
def main():
    # Initialize database
    init_db()
    
    st.set_page_config(
        page_title="Smart Surveillance",
        page_icon="🚨",
        layout="wide"
    )
    st.title("Smart Surveillance: Detecting Abnormal Human Activities with Deep Learning")
    
    
    if not st.session_state['authenticated']:
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            login()
        with tab2:
            register()
    else:
        st.sidebar.button("Logout", on_click=logout)
        # Initialize detector
        detector = AnomalyDetector()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Image Upload", "Video Upload", "Activity Logs"])
        
        with tab1:
            st.header("Upload an Image")
            image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if image_file is not None:
                # Process image
                img_rgb, is_anomaly, confidence = process_image(image_file, detector)
                
                # Display results
                if confidence > 0.60:
                    user_email = get_user_email(st.session_state['user'])
                    send_email(user_email, img_rgb, confidence)
                    logged_frame = log_activity(st.session_state['user'], confidence, img_rgb)  # Log activity
                    
                    st.success(f"Alert sent to {user_email}")
                    st.image(logged_frame, caption="Logged Frame (Anomaly Detected)", width=300)
                else:
                    st.success("No Anomaly Detected")
                    st.image(img_rgb, caption=f"Confidence: {confidence:.2f}", width=300)
        
        with tab2:
            st.header("Upload a Video")
            video_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])
            
            if video_file is not None:
                # Process video
                stats, final_status, final_confidence, best_frame_rgb, anomaly_rate = process_video(video_file, detector)
                
                # Display final status
                if final_status == "ANOMALY DETECTED!" and final_confidence > 0.60:
                    user_email = get_user_email(st.session_state['user'])
                    send_email(user_email, best_frame_rgb, final_confidence)
                    logged_frame = log_activity(st.session_state['user'], final_confidence, best_frame_rgb)  # Log activity
                    
                    st.success(f"Alert sent to {user_email}")
                    st.image(logged_frame, caption="Logged Frame (Anomaly Detected)", width=300)
                else:
                    st.success("No Anomaly Detected")
                
                # Display statistics
                st.subheader("Processing Statistics")
                st.dataframe(stats)
        
        with tab3:
            st.header("Activity Logs")
            
            # Fetch logs for the current user
            logs = fetch_activity_logs(st.session_state['user'])
            
            if logs:
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                
                for idx, log in enumerate(logs):
                    log_id, timestamp, confidence, frame_blob = log

                    frame_array = np.frombuffer(frame_blob, np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    st.subheader(f"Timestamp: {timestamp}")
                    st.text(f"Confidence: {float(confidence):.2f}")
                    st.image(frame_rgb, caption="Logged Frame", width=300)

                    delete_key = f"delete_{log_id}"

                    if st.button(f"Delete", key=delete_key):
                        # Set a session state variable to trigger confirmation
                        st.session_state[f"confirm_delete_{log_id}"] = True

                    if st.session_state.get(f"confirm_delete_{log_id}", False):
                        st.warning("Are you sure you want to delete this log?")
                        if st.button(f"Yes, Delete", key=f"confirm_yes_{log_id}"):
                            c.execute('DELETE FROM activity_logs WHERE id = ?', (log_id,))
                            conn.commit()
                            st.success("Log deleted successfully!")
                            st.session_state.pop(f"confirm_delete_{log_id}")
                            st.rerun()
                        if st.button(f"No, Cancel", key=f"confirm_no_{log_id}"):
                            st.session_state.pop(f"confirm_delete_{log_id}")
                            
                    st.markdown("---")

                conn.close()
            else:
                st.info("No activity logs found.")



if __name__ == "__main__":
    main()