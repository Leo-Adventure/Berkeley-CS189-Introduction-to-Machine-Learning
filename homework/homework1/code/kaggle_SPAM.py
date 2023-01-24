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
    df.to_csv('spam_submission.csv', index_label='Id')

if __name__ == "__main__":
    spam_data = np.load("../data/spam-data.npz")
    spam_train = spam_data["training_data"]
    spam_train_label = spam_data["training_labels"]
    spam_test = spam_data["test_data"]
    print(spam_test.shape)

    # state = np.random.get_state()
    # np.random.shuffle(spam_train)
    # np.random.set_state(state)
    # np.random.shuffle(spam_train_label)

    # len_spam_train = len(spam_train)
    # len_partition = int(len_spam_train / 5)
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
    svm_model.fit(spam_train, spam_train_label)
    pred = svm_model.predict(spam_validation)
    acc = metrics.accuracy_score(y_true=spam_validation_label, y_pred=pred)
    print(acc)
    pred = svm_model.predict(spam_test)
    results_to_csv(pred)