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
    df.to_csv('submission.csv', index_label='Id')

if __name__ == "__main__":
    # For the MNIST dataset, write code that sets aside 10,000 training images as a validation set.
    mnist_data = np.load("../data/mnist-data.npz")
    mnist_train = mnist_data["training_data"]
    mnist_train_label = mnist_data["training_labels"]
    mnist_test = mnist_data["test_data"]

    mnist_state = np.random.get_state()
    np.random.shuffle(mnist_train)
    np.random.set_state(mnist_state)
    np.random.shuffle(mnist_train_label)

    mnist_validation = mnist_train[:9999]
    mnist_validation_label = mnist_train_label[:9999]

    mnist_train = mnist_train[10000:]
    mnist_train_label = mnist_train_label[10000:]
    
    # c_arr = [1.05e-2]

    mnist_train = mnist_train[:15000]
    mnist_train_label = mnist_train_label[:15000]

    nsamples, n1, n2, n3 = mnist_validation.shape
    mnist_validation = mnist_validation.reshape(nsamples, n1*n2*n3)
    
    nsamples, n1, n2, n3 = mnist_train.shape
    mnist_train = mnist_train.reshape(nsamples, n1*n2*n3)

    nsamples, n1, n2, n3 = mnist_test.shape
    mnist_test = mnist_test.reshape(nsamples, n1*n2*n3)

    # svm_model = SVC(kernel="linear", C=c_arr[0]) 93.7%
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
    print("here")
    # svm_model = SVC(C=10, gamma=0.001, kernel="rbf")
    svm_model.fit(mnist_train, mnist_train_label)
    print("after")
    pred = svm_model.predict(mnist_validation)
    acc = metrics.accuracy_score(y_true=mnist_validation_label, y_pred=pred)
    print("mnist accuracy = ", str(acc))
    pred = svm_model.predict(mnist_test)
    results_to_csv(pred)