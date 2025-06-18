import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        
    def preprocess_frame(self, frame):
        # Resize frame to 64x64
        frame = cv2.resize(frame, (64, 64))
        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Normalize pixel values
        frame = frame.astype('float32') / 255.0
        # Add channel dimension
        frame = np.expand_dims(frame, axis=-1)
        return frame
        
    def load_data(self):
        X = []
        y = []
        frame_count = 0
        skipped_frames = 0
        
        # Load labels
        labels_path = os.path.join(self.dataset_path, 'Labels')
        
        print("\nProcessing categories:")
        for category in os.listdir(self.dataset_path):
            category_path = os.path.join(self.dataset_path, category)
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
                        X.append(processed_frame)
                        
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
        
    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        return model
        
    def train(self):
        # Load and preprocess data
        X, y = self.load_data()
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()
        
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        
        # Create callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'best_anomaly_detector.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            class_weight=class_weights,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        # Save the final model
        self.model.save('anomaly_detector.h5')
        
        return history

if __name__ == "__main__":
    # Initialize detector
    detector = AnomalyDetector('DCSASS Dataset')
    
    # Train the model
    history = detector.train()
    
    # Print training results
    print("\nTraining complete!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
