import tensorflow as tf


def get_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_data, train_label), (test_data, test_label) = fashion_mnist.load_data()
    train_data = train_data.reshape(60000, 28 * 28)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_data = train_data / 255
    return train_data, train_label, class_names
