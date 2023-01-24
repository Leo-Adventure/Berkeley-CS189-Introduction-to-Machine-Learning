import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # For the MNIST dataset, write code that sets aside 10,000 training images as a validation set.
    mnist_data = np.load("../data/mnist-data.npz")
    mnist_train = mnist_data["training_data"]
    mnist_train_label = mnist_data["training_labels"]
    mnist_state = np.random.get_state()
    np.random.shuffle(mnist_train)
    np.random.set_state(mnist_state)
    np.random.shuffle(mnist_train_label)
    mnist_validation = mnist_train[:9999]
    mnist_validation_label = mnist_train_label[:9999]
    mnist_train = mnist_train[10000:]
    mnist_train_label = mnist_train_label[10000:]
    # For the spam dataset, write code that sets aside 20\% of the training data as a validation set.
    spam_data = np.load("../data/spam-data.npz")
    spam_train = spam_data["training_data"]
    spam_train_label = spam_data["training_labels"]
    state = np.random.get_state()
    np.random.shuffle(spam_train)
    np.random.set_state(state)
    np.random.shuffle(spam_train_label)
    len_spam_train = int(len(spam_train)*0.8)
    len_spam_validation = len(spam_train) - len_spam_train
    spam_validation = spam_train[:len_spam_validation - 1]
    spam_validation_label = spam_train_label[:len_spam_validation-1]
    spam_train = spam_train[len_spam_validation:]
    spam_train_label = spam_train_label[len_spam_validation:]
    # For the CIFAR-10 dataset, write code that sets aside 5,000 training images as a validation set.
    cifar_data = np.load("../data/cifar10-data.npz")
    cifar_train = cifar_data["training_data"]
    cifar_train_label = cifar_data["training_labels"]
    state = np.random.get_state()
    np.random.shuffle(cifar_train)
    np.random.set_state(state)
    np.random.shuffle(cifar_train_label)

    cifar_validation = cifar_train[:4999]
    cifar_validation_label = cifar_train_label[:4999]

    cifar_train = cifar_train[5000:]
    cifar_train_label = cifar_train_label[5000:]