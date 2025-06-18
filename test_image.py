import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

class ImageAnomalyDetector:
    def __init__(self, model_path='best_anomaly_detector.h5', image_size=(64, 64)):
        self.model = load_model(model_path)
        self.image_size = image_size
        self.threshold = 0.5  # Confidence threshold for anomaly detection
        
    def preprocess_image(self, image_path):
        """
        Preprocess an image file for model prediction
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Resize to model input size
        img = cv2.resize(img, self.image_size)
        
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add channel dimension
        img = np.expand_dims(img, axis=-1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
        
    def detect_anomaly(self, image_path):
        """
        Detect anomaly in a single image
        """
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(image_path)
            
            # Get prediction
            prediction = self.model.predict(processed_img)[0][0]
            
            # Read original image for visualization
            original_img = cv2.imread(image_path)
            
            # Draw prediction result
            if prediction > self.threshold:
                cv2.putText(original_img, f"ANOMALY DETECTED! (Confidence: {prediction:.2f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(original_img, "NORMAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save and display the result
            result_path = os.path.splitext(image_path)[0] + '_result.jpg'
            cv2.imwrite(result_path, original_img)
            print(f"\nResults saved to: {result_path}")
            print(f"Prediction: {'Anomaly' if prediction > self.threshold else 'Normal'}")
            print(f"Confidence: {prediction:.4f}")
            
            # Display the image
            cv2.imshow('Anomaly Detection Result', original_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            
    def test_directory(self, directory_path):
        """
        Test the model on all images in a directory
        """
        if not os.path.exists(directory_path):
            print(f"Error: Directory not found: {directory_path}")
            return
            
        # Get all image files
        image_files = [f for f in os.listdir(directory_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            print("No image files found in the directory")
            return
            
        print(f"\nTesting {len(image_files)} images...")
        
        predictions = []
        for image_file in image_files:
            image_path = os.path.join(directory_path, image_file)
            print(f"\nProcessing {image_file}...")
            
            try:
                # Preprocess the image
                processed_img = self.preprocess_image(image_path)
                
                # Get prediction
                prediction = self.model.predict(processed_img)[0][0]
                predictions.append((image_file, prediction > self.threshold, prediction))
                
                # Save and display the result
                original_img = cv2.imread(image_path)
                
                if prediction > self.threshold:
                    cv2.putText(original_img, f"ANOMALY DETECTED! (Confidence: {prediction:.2f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(original_img, "NORMAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save the result
                result_path = os.path.splitext(image_path)[0] + '_result.jpg'
                cv2.imwrite(result_path, original_img)
                print(f"Results saved to: {result_path}")
                
                # Display the image
                cv2.imshow('Anomaly Detection Result', original_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
        
        # Print summary of results
        print("\nTest Summary:")
        normal_count = sum(1 for _, is_anomaly, _ in predictions if not is_anomaly)
        anomaly_count = len(predictions) - normal_count
        print(f"Total images: {len(predictions)}")
        print(f"Normal images: {normal_count}")
        print(f"Anomaly images: {anomaly_count}")

if __name__ == "__main__":
    # Example usage: Test a single image
    detector = ImageAnomalyDetector()
    
    # Example image path (replace with your own image)
    image_path = "test_image.jpg"
    
    # Test a single image
    print("\nTesting single image...")
    detector.detect_anomaly(image_path)
    
    # Example directory path (replace with your own directory)
    # directory_path = "test_images"
    # print("\nTesting directory of images...")
    # detector.test_directory(directory_path)
