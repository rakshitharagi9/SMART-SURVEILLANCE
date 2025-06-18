import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self, model_path='anomaly_detector.h5', image_size=(64, 64)):
        self.model = load_model(model_path)
        self.image_size = image_size
        self.threshold = 0.5  # Confidence threshold for anomaly detection
        
    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, self.image_size)
        frame = frame / 255.0
        return np.expand_dims(frame, axis=0)
    
    def detect_anomaly(self, frame):
        processed_frame = self.preprocess_frame(frame)
        prediction = self.model.predict(processed_frame)[0][0]
        
        if prediction > self.threshold:
            return True, prediction
        return False, prediction
    
    def process_video(self, video_path):
        # Normalize the path
        video_path = os.path.normpath(video_path)
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect anomaly
            is_anomaly, confidence = self.detect_anomaly(frame)
            
            # Display results
            if is_anomaly:
                cv2.putText(frame, f"ANOMALY DETECTED! (Confidence: {confidence:.2f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "NORMAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Surveillance', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

class AnomalyDetectorTester:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        
    def preprocess_frame(self, frame):
        # Resize frame to 64x64
        frame = cv2.resize(frame, (64, 64))
        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Normalize pixel values
        frame = frame.astype('float32') / 255.0
        # Add channel dimension
        frame = np.expand_dims(frame, axis=-1)
        # Add batch dimension
        frame = np.expand_dims(frame, axis=0)
        return frame
        
    def load_data(self, dataset_path):
        X = []
        y = []
        frame_count = 0
        skipped_frames = 0
        
        # Load labels
        labels_path = os.path.join(dataset_path, 'Labels')
        
        print("\nProcessing categories:")
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if not os.path.isdir(category_path) or category == 'Labels':
                continue
                
            print(f"\nProcessing {category}...")
            
            # Load category labels
            label_file = os.path.join(labels_path, f"{category}.csv")
            if not os.path.exists(label_file):
                print(f"Warning: Label file not found for {category}")
                continue
                
            df = pd.read_csv(label_file, header=None)
            
            for _, row in df.iterrows():
                frame_count += 1
                
                # Get video name and frame number
                filename = row[0]
                print(f"\nProcessing file: {filename}")
                
                try:
                    parts = filename.split('_')
                    print(f"Parts after split: {parts}")
                    video_name = '_'.join(parts[:-1])
                    print(f"Video name: {video_name}")
                    frame_num = int(parts[-1])
                    print(f"Frame number: {frame_num}")
                except Exception as e:
                    print(f"Error extracting frame number from {filename}: {str(e)}")
                    skipped_frames += 1
                    continue
                
                # Fix typo in video name if it's a RoadAccidents video
                if category == 'RoadAccidents':
                    video_name = video_name.replace('oadAccidents', 'RoadAccidents')
                    video_name = video_name.replace('RRoadAccidents', 'RoadAccidents')
                
                # Get video directory path
                video_dir = os.path.join(category_path, f"{video_name}.mp4")
                
                if not os.path.exists(video_dir):
                    print(f"Warning: Video directory not found: {video_dir}")
                    skipped_frames += 1
                    continue
                    
                try:
                    # Get the specific frame file
                    frame_file = os.path.join(video_dir, f"{video_name}_{frame_num}.mp4")
                    
                    if not os.path.exists(frame_file):
                        print(f"Warning: Frame file not found: {frame_file}")
                        skipped_frames += 1
                        continue
                        
                    cap = cv2.VideoCapture(frame_file)
                    ret, frame = cap.read()
                    
                    if ret:
                        processed_frame = self.preprocess_frame(frame)
                        X.append(processed_frame[0])  # Remove batch dimension
                        
                        # Validate and convert label to integer
                        label = row[2]
                        if pd.isna(label):
                            print(f"Warning: NaN label found for file: {filename}")
                            skipped_frames += 1
                            continue
                            
                        label = int(label)
                        if label not in [0, 1]:
                            print(f"Warning: Invalid label {label} for file: {filename}")
                            skipped_frames += 1
                            continue
                            
                        y.append(label)
                    else:
                        print(f"Warning: Could not read frame {frame_num} from {video_dir}")
                        skipped_frames += 1
                    
                    cap.release()
                except Exception as e:
                    print(f"Error processing frame {frame_num} from {video_dir}: {str(e)}")
                    skipped_frames += 1
                    continue
        
        X = np.array(X)
        y = np.array(y)
        
        # Validate that X and y have the same length
        if len(X) != len(y):
            print(f"Warning: X and y have different lengths: {len(X)} vs {len(y)}")
            print("Attempting to fix by truncating longer array...")
            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y[:min_length]
        
        print(f"\nData loading complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames skipped: {skipped_frames}")
        print(f"Final dataset size: {len(X)} frames")
        print(f"Class distribution: Normal={np.sum(y==0)}, Anomaly={np.sum(y==1)}")
        
        return X, y
        
    def test(self, dataset_path):
        """
        Test the model on a dataset
        """
        # # Load test data
        # X_test, y_test = self.load_data(dataset_path)
        
        # if len(X_test) == 0:
        #     print("Error: No test data found")
        #     return
            
        # print(f"\nTesting on {len(X_test)} samples...")

        X, y = self.load_data(dataset_path)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\nTesting on {len(X_test)} samples...")
        # Evaluate the model
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Calculate F1 score
        f1 = f1_score(y_test, y_pred)
        print(f"\nF1 Score: {f1:.4f}")
        
        # Calculate accuracy using sklearn's accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest accuracy: {accuracy:.4f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('test_roc_curve.png')
        plt.close()
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

if __name__ == "__main__":
    # Initialize tester with the saved model
    tester = AnomalyDetectorTester('best_anomaly_detector.h5')
    
    # Test the model
    results = tester.test('DCSASS Dataset')
    
    print("\nTest Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    
   # detector = AnomalyDetector()
    
    # Example usage: Replace with your video path
   # video_path = os.path.join("DCSASS Dataset", "Abuse", "Abuse001_x264.mp4")
   # detector.process_video(video_path)
