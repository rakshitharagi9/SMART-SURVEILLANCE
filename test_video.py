import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime

class VideoAnomalyDetector:
    def __init__(self, model_path='best_anomaly_detector.h5', image_size=(64, 64)):
        self.model = load_model(model_path)
        self.image_size = image_size
        self.threshold = 0.5  # Confidence threshold for anomaly detection
        
    def preprocess_frame(self, frame):
        """
        Preprocess a frame for model prediction
        """
        # Resize frame to model input size
        frame = cv2.resize(frame, self.image_size)
        
        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Normalize pixel values
        frame = frame.astype('float32') / 255.0
        
        # Add channel dimension
        frame = np.expand_dims(frame, axis=-1)
        
        # Add batch dimension
        frame = np.expand_dims(frame, axis=0)
        
        return frame
        
    def detect_anomaly(self, frame):
        """
        Detect anomaly in a frame
        Returns: (is_anomaly, confidence)
        """
        processed_frame = self.preprocess_frame(frame)
        prediction = self.model.predict(processed_frame)[0][0]
        return prediction > self.threshold, prediction
        
    def process_video(self, video_path):
        """
        Process a video file and detect anomalies in real-time
        """
        # Normalize the path
        video_path = os.path.normpath(video_path)
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
            
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer
        output_path = os.path.splitext(video_path)[0] + '_anomaly_result.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\nProcessing video: {video_path}")
        print(f"Output will be saved to: {output_path}")
        print(f"Video FPS: {fps}")
        print(f"Resolution: {width}x{height}")
        
        frame_count = 0
        anomaly_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Detect anomaly
            is_anomaly, confidence = self.detect_anomaly(frame)
            
            # Draw results on frame
            if is_anomaly:
                cv2.putText(frame, f"ANOMALY DETECTED! (Confidence: {confidence:.2f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                anomaly_count += 1
            else:
                cv2.putText(frame, "NORMAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (10, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write to output video
            out.write(frame)
            
            # Display frame
            cv2.imshow('Anomaly Detection', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Anomalies detected: {anomaly_count}")
        print(f"Anomaly rate: {(anomaly_count/frame_count)*100:.2f}%")
        print(f"Output video saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    detector = VideoAnomalyDetector()
    
    # Use the video file in the directory
    video_path = r"C:\\MAJOR PROJECT\\testing photos videos\\robbery.mp4"
    
    
    print("\nStarting video processing...")
    detector.process_video(video_path)
