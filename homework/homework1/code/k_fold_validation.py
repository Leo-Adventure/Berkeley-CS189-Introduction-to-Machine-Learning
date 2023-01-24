import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt

if __name__ == "__main__":
    spam_data = np.load("../data/spam-data.npz")
    spam_train = spam_data["training_data"]
    spam_train_label = spam_data["training_labels"]

    state = np.random.get_state()
    np.random.shuffle(spam_train)
    np.random.set_state(state)
    np.random.shuffle(spam_train_label)

    len_spam_train = len(spam_train)
    len_partition = int(len_spam_train / 5)
    c_array = [1, 2, 3, 4, 5, 6, 7, 8]
    acc_arr = []
    for j in range(len(c_array)):
        svm_model = SVC(kernel="linear", C=c_array[j])
        sub_acc_arr = []
        for i in range(5):
            training_set = spam_train[len_partition*(i):len_partition*(i+1)-1]
            training_set_label = spam_train_label[len_partition*(i):len_partition*(i+1)-1]
            j = (i+1)%5
            validation_set = spam_train[len_partition*(j):len_partition*(j+1)-1]
            validation_set_label = spam_train_label[len_partition*(j):len_partition*(j+1)-1]

            svm_model.fit(training_set, training_set_label)
            pred = svm_model.predict(validation_set)
            acc = metrics.accuracy_score(y_true=validation_set_label, y_pred=pred)
            sub_acc_arr.append(acc)
        num_arr = np.array(sub_acc_arr)
        avg_val = np.mean(num_arr)
        acc_arr.append(avg_val)

    print(acc_arr)
    

    plt.plot(c_array, acc_arr, label='spam validation accuracy', marker=".")
    plt.xlabel('number of training examples')
    plt.ylabel('accuracy')
    plt.title("accuracy on the training and validation sets versus the number of training examples")
    plt.grid(True)
    plt.legend()
    plt.show()