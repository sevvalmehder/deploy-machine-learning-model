# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


class FashionMNISTClassifier():

    # Constructor
    def __init__(self):
        self.dataset = tf.keras.datasets.fashion_mnist
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def get_dataset(self):
        return self.dataset.load_data()

    def get_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        return model

    def train(self, train_images, train_labels, model, epochs=10, callbacks=None):
        if callbacks:
            model.fit(train_images, train_labels,
                      epochs=epochs, callbacks=[callbacks])
        else:
            model.fit(train_images, train_labels, epochs)

    def evaluate(self, test_images, test_labels, model, verbose=2):
        test_loss, test_acc = model.evaluate(
            test_images,  test_labels, verbose)
        print('\nTest accuracy:', test_acc)

    def predict(self, image, model):
        predictions = model.predict(image)
        predicted_label = np.argmax(predictions[0])

        return self.class_names[predicted_label]

    def get_callbacks(self):
        callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='models/model.{epoch:02d}-{accuracy:.2f}.h5', verbose=1)
        return callback

    def load_model(self, model_path):
        model = tf.keras.models.load_model(model_path)
        return model
