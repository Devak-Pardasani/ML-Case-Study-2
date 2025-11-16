import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def main():
    X = np.loadtxt("data_points.csv", delimiter=",")
    Y = np.loadtxt("Train_1/labels_train_1.csv", delimiter=",")

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.5)

    log = LogisticRegression(multi_class="ovr", solver="liblinear", max_iter=1000)
    knn = KNeighborsClassifier(n_neighbors=5)
    log.fit(X_train, Y_train)
    acc = log.score(X=X_test,y=Y_test)
    print(acc)
    """
    X_train = X[]
    X_test = Y[]
    Y_train = Y[]
    Y_test = Y[]
    """

if __name__ == "__main__":
    main()