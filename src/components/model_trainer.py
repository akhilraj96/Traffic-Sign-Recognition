import tensorflow as tf
import pickle
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from src.utils import load_data
from sklearn.utils import shuffle

class model(tf.keras.Model):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = Conv2D(6, kernel_size=(5, 5), activation='relu')
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv2 = Conv2D(16, kernel_size=(5, 5), activation='relu')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.fc1 = Dense(120, activation='relu')
        self.fc2 = Dense(84, activation='relu')
        self.fc3 = Dense(43, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

class ModelTrainer:
    def __init__(self, train_path, val_path, batch_size, epochs, learning_rate):
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = model()

    def train(self):
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

    def save_model(self, save_path):
        # Save the trained model as a pickle file
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)
        print("Model saved")

if __name__ == "__main__":
    train_path = "artifacts/train.p"
    val_path = "artifacts/valid.p"

    trainer = ModelTrainer(train_path=train_path, val_path=val_path, batch_size=128, epochs=10, learning_rate=0.001)
    trainer.train()
    trainer.save_model("artifacts/model.pkl")
