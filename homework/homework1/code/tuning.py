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