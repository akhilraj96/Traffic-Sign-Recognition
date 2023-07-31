import os
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from src.utils import load_data
from sklearn.utils import shuffle

class ModelTrainer:
    def __init__(self, train_path, val_path, batch_size, epochs, learning_rate):
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = self.create_model()

    def create_model(self):
        model = Sequential([
            Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, kernel_size=(5, 5), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(120, activation='relu'),
            Dense(84, activation='relu'),
            Dense(43, activation='softmax')
        ])
        return model

    def train(self, save_path):
        X_train, y_train = load_data(self.train_path)
        X_valid, y_valid = load_data(self.val_path)

        # Convert labels to one-hot encoding
        y_train = to_categorical(y_train, 43)
        y_valid = to_categorical(y_valid, 43)

        # Create a TensorFlow Dataset for efficient data batching
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(self.batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(self.batch_size)

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # Training loop
        self.model.fit(train_dataset, epochs=self.epochs, validation_data=valid_dataset)

        # Save the trained model as a pickle file
        os.makedirs(save_path, exist_ok=True)
        self.model.save(save_path)
        print("Model saved")

if __name__ == "__main__":
    train_path = "artifacts/train.p"
    val_path = "artifacts/valid.p"

    trainer = ModelTrainer(train_path=train_path, val_path=val_path, batch_size=128, epochs=10, learning_rate=0.001)
    trainer.train("artifacts/model")
