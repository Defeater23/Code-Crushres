# Offline inference script
import os
import numpy as np
import matplotlib.pyplot as plt

# For CPU-only operation if needed (uncomment if having GPU issues)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("TensorFlow not found. Please install it locally using: pip install tensorflow")
    exit(1)

def load_and_predict(image_path=None, test_index=None):
    # Load the model
    if not os.path.exists('model.h5'):
        print("Error: Model file 'model.h5' not found. Please train the model first.")
        return
    
    model = load_model('model.h5')
    print("Model loaded successfully!")
    
    # Option 1: Use a specific index from test data
    if test_index is not None:
        if not os.path.exists('input_test.csv') or not os.path.exists('labels_test.csv'):
            print("Error: Test data files not found.")
            return
            
        X_test = np.loadtxt('input_test.csv', delimiter=',')
        Y_test = np.loadtxt('labels_test.csv', delimiter=',')
        
        X_test = X_test.reshape(len(X_test), 100, 100, 3) / 255.0
        Y_test = Y_test.reshape(len(Y_test), 1)
        
        if test_index >= len(Y_test):
            print(f"Error: Index {test_index} is out of range. Max index: {len(Y_test)-1}")
            return
            
        # Display the image
        plt.figure(figsize=(5, 5))
        plt.imshow(X_test[test_index,:])
        plt.title("Test Image")
        plt.savefig('prediction_result.png')
        plt.show()
        
        # Make prediction
        prediction = model.predict(X_test[test_index,:].reshape(1, 100, 100, 3))
        print(f"Prediction: {prediction[0][0]}")
        print(f"Actual label: {Y_test[test_index][0]}")
    
    # For future extension: add code to load and predict custom images
    
if __name__ == "__main__":
    print("Offline Inference Tool")
    print("----------------------")
    try:
        test_index = int(input("Enter test image index (or press Enter for random): ") or "-1")
        if test_index < 0:
            import random
            X_test = np.loadtxt('input_test.csv', delimiter=',')
            test_index = random.randint(0, len(X_test)-1)
            print(f"Using random index: {test_index}")
        
        load_and_predict(test_index=test_index)
    except ValueError:
        print("Please enter a valid number.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()