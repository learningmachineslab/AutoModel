

from wrapper import Wrapper
import numpy as np

class kerasWrapper(Wrapper):   

    def __init__(self, model, model_params):
        super().__init__(model, model_params)

    def fit(self, train_x, train_y, epochs=2):
        return self.model.fit(train_x, train_y, epochs=epochs)
    
    def score(self, test_x, test_y):
        """
        Returns the default scoring (accuracy).
        """
        test_loss, test_acc = self.model.evaluate(test_x, test_y)
        return test_acc

    def predict_proba(self, test_x):
        """
        Returns the predicted probability of belonging to each
        class.
        """
        return self.model.predict(test_x)

    def predict(self, test_x):
        """
        Returns the predicted class belonging for input data
        """
        predicted_probabilities = self.model.predict(test_x)
        predicted = np.argmax(predicted_probabilities, axis=1)
        return predicted

    # def decision_function(self):
    #     pass

    # def inverse_transform(self):
    #     pass

if __name__ == '__main__':
    # from __future__ import absolute_import, division, print_function, unicode_literals

    # TensorFlow and tf.keras
    import tensorflow as tf
    from tensorflow import keras

    # Helper libraries
    import numpy as np
    import matplotlib.pyplot as plt

    print(tf.__version__)

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat'
    ,'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0

    test_images = test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
              )

    kw = kerasWrapper(model=model, model_params=None)
    kw.fit(train_images, train_labels)
    kw.score(test_x=test_images, test_y=test_labels)
    predictions = kw.predict(test_images)
    print('predicted class:', predictions[0])

    predicted_prob = kw.predict_proba(test_images)
    print('predicted probability:', predicted_prob[0])
