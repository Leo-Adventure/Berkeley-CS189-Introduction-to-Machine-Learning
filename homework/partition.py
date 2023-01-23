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
    
    c_arr = [5e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 3.5e-4, 4e-4, 4.5e-4, 5e-4]

    mnist_train = mnist_train[:10000]
    mnist_train_label = mnist_train_label[:10000]

    nsamples, n1, n2, n3 = mnist_validation.shape
    mnist_validation = mnist_validation.reshape(nsamples, n1*n2*n3)
    
    nsamples, n1, n2, n3 = mnist_train.shape
    mnist_train = mnist_train.reshape(nsamples, n1*n2*n3)

    val_acc_arr = []
    for i in range(len(c_arr)):
        svm_model = SVC(kernel="linear", C=c_arr[i])
        svm_model.fit(mnist_train, mnist_train_label)
        mnist_pred = svm_model.predict(mnist_validation)
        accuracy = metrics.accuracy_score(y_true=mnist_validation_label, y_pred=mnist_pred)
        val_acc_arr.append(accuracy)

    print(val_acc_arr)
    plt.plot(c_arr, val_acc_arr, label='mnist training accuracy', marker="x")
    plt.xlabel('Value of C')
    plt.ylabel('accuracy')
    plt.title("accuracy on the validation sets versus the value of C in SVM model")
    plt.grid(True)
    plt.legend()
    plt.show()


    '''Q3
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

    size_arr = [100, 200, 500, 1000, 2000, len_spam_train]

    spam_train_acc_array = []
    spam_validation_acc_array = []

    for i in range(len(size_arr)):
        spam_train_section = spam_train[:size_arr[i] - 1]
        spam_train_label_section = spam_train_label[:size_arr[i] - 1]
        svm_model.fit(spam_train_section, spam_train_label_section)

        spam_train_pred = svm_model.predict(spam_train_section)
        spam_train_acc = metrics.accuracy_score(y_true=spam_train_label_section, y_pred=spam_train_pred)
        spam_train_acc_array.append(spam_train_acc)

        spam_val_pred = svm_model.predict(spam_validation)
        spam_val_acc = metrics.accuracy_score(y_true=spam_validation_label, y_pred = spam_val_pred)
        spam_validation_acc_array.append(spam_val_acc)

    plt.plot(size_arr, spam_train_acc_array, label='spam training accuracy', marker=".")
    plt.plot(size_arr, spam_validation_acc_array, label='spam validation accuracy', marker="x")
    plt.xlabel('number of training examples')
    plt.ylabel('accuracy')
    plt.title("accuracy on the training and validation sets versus the number of training examples")
    plt.grid(True)
    plt.legend()
    plt.show()

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

    size_arr = [100, 200, 500, 1000, 2000, 5000]
    cifar_train_acc_array = []
    cifar_validation_acc_array = []
    for i in range(len(size_arr)):
        cifar_train_section = cifar_train[:size_arr[i]-1]
        cifar_train_label_section = cifar_train_label[:size_arr[i]-1]

        svm_model.fit(cifar_train_section, cifar_train_label_section)

        cifar_train_pred = svm_model.predict(cifar_train_section)
        cifar_train_acc = metrics.accuracy_score(y_true=cifar_train_label_section, y_pred = cifar_train_pred)
        cifar_train_acc_array.append(cifar_train_acc)

        cifar_validation_pred = svm_model.predict(cifar_validation)
        cifar_validation_acc = metrics.accuracy_score(y_true = cifar_validation_label, y_pred=cifar_validation_pred)
        cifar_validation_acc_array.append(cifar_validation_acc)

    plt.plot(size_arr, cifar_train_acc_array, label='cifar10 training accuracy', marker=".")
    plt.plot(size_arr, cifar_validation_acc_array, label='cifar10 validation accuracy', marker="x")
    plt.xlabel('number of training examples')
    plt.ylabel('accuracy')
    plt.title("accuracy on the training and validation sets versus the number of training examples")
    plt.grid(True)
    plt.legend()
    plt.show()
'''