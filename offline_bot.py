# Offline-compatible ML model training
import matplotlib.pyplot as plt
import random
import numpy as np
import os

# Check if TensorFlow is available locally
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not found. Please install it locally using: pip install tensorflow")
    exit(1)

def train_model():
    # Ensure data files exist
    required_files = ['input.csv', 'labels.csv', 'input_test.csv', 'labels_test.csv']
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: Required data file '{file}' not found.")
            print("Please place all data files in the same directory as this script.")
            return

    # Loading Dataset 
    print("Loading training data...")
    X_train = np.loadtxt('input.csv', delimiter=',')
    Y_train = np.loadtxt('labels.csv', delimiter=',')

    print("Loading test data...")
    X_test = np.loadtxt('input_test.csv', delimiter=',')
    Y_test = np.loadtxt('labels_test.csv', delimiter=',')

    # Reshape data
    X_train = X_train.reshape(len(X_train), 100, 100, 3)
    X_test = X_test.reshape(len(X_test), 100, 100, 3)
    Y_train = Y_train.reshape(len(Y_train), 1)
    Y_test = Y_test.reshape(len(Y_test), 1)

    # Normalize data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Create Model
    print("Building model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # Train model
    print("Training model...")
    model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_data=(X_test, Y_test))

    # Evaluate model
    print("Evaluating model...")
    model.evaluate(X_test, Y_test)

    # Save model for later offline use
    model.save('model.h5')
    print("Model saved as 'model.h5'")

    # Test with a random image
    print("Testing with a random image...")
    index = random.randint(0, len(Y_test)-1)
    plt.figure(figsize=(5, 5))
    plt.imshow(X_test[index,:])
    plt.title("Test Image")
    plt.savefig('test_image.png')  # Save the image in case plt.show() doesn't work in some environments
    plt.show()

    y_output = model.predict(X_test[index,:].reshape(1, 100, 100, 3))
    print(f"Prediction: {y_output[0][0]}")
    print(f"Actual label: {Y_test[index][0]}")
    
    return model

if __name__ == "__main__":
    # Enable offline logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    
    print("Starting offline ML training...")
    try:
        model = train_model()
        print("Training completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()