import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Usage: results_to_csv(clf.predict(X_test))
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv('cifar_submission.csv', index_label='Id')


if __name__ == "__main__":
    cifar10_data = np.load("../data/cifar10-data.npz")
    cifar10_train = cifar10_data["training_data"]
    cifar10_train_label = cifar10_data["training_labels"]
    cifar10_test = cifar10_data["test_data"]
    print(cifar10_test.shape)

    # len_cifar10_train = len(cifar10_train)
    # len_partition = int(len_cifar10_train / 5)
    state = np.random.get_state()
    np.random.shuffle(cifar10_train)
    np.random.set_state(state)
    np.random.shuffle(cifar10_train_label)

    # len_cifar10_train = int(len(cifar10_train)*0.8)
    # len_cifar10_validation = len(cifar10_train) - len_cifar10_train

    cifar10_validation = cifar10_train[:4999]
    cifar10_validation_label = cifar10_train_label[:4999]

    cifar10_train = cifar10_train[5000:10000]
    cifar10_train_label = cifar10_train_label[5000:10000]
    params = {"C": [0.1, 1, 10, 100, 1000]}

    folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

    # instantiating a model with cost=1
    model = SVC()

    # set up grid search scheme
    # note that we are still using the 5 fold CV scheme we set up earlier
    svm_model = GridSearchCV(estimator = model, param_grid = params, 
                            scoring= 'accuracy', 
                            cv = folds, 
                            verbose = 1,
                        return_train_score=True) 
    # svm_model = SVC(kernel="linear", C=c)
    print("here")
    svm_model.fit(cifar10_train, cifar10_train_label)
    print("after fit")
    # pred = svm_model.predict(cifar10_validation)
    # print("after")
    # acc = metrics.accuracy_score(y_true=cifar10_validation_label, y_pred=pred)
    # print(acc)
    pred = svm_model.predict(cifar10_test)
    results_to_csv(pred)