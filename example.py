# the OFFICIAL MEMEFUL codebase for the 18.065 final project

# mnist ML tensorflow example

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from meme_utils import *




if __name__ == "__main__":

    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(y_train)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10,activation= tf.nn.softmax)
    ])

    model.compile(optimizer='adam', 
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=5)
    test_loss, test_acc =   model.evaluate(x_test, y_test)
    print('test loss =', test_loss)
    print('test accuracy = ', test_acc)
    # test_loss, test_acc = model.evaluate(x_test[:1], y_test[:1])

    # # sanity check - get the prediction for the first '5' from the training set
    # num_predictions = 3
    # predictions = model.predict(x_test[:num_predictions])
    # print(predictions)



    while True:
        # user-interactive test for stuff in test datasets
        ix = input("Type index for picture from test >>>")
        ix = int(ix)
        if ix > len(x_test):
            print("Index out of range, please enter indices up to ", num_predictions-1)

        else:
            res = np.argmax(predictions[ix])
            # show one picture  of the test set3
            show_img(x_test[ix], "Expectation = "+ str(y_test[ix]) + " result=" + str(res))