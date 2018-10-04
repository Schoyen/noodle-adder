import numpy as np
import sklearn.metrics as skm
import sklearn.model_selection as skms
from sklearn.neural_network import MLPClassifier


def eight_bit(number):
    return bin(number)[2:].zfill(8)


def binary_array(number):
    bin_str = eight_bit(number)
    return np.array([int(_) for _ in bin_str])


def number_from_array(arr):
    val = 0

    for i in range(len(arr)):
        val += arr[i] << (7 - i)

    return val


def output_binary_addition(a, b, res, res_pred):
    print("  ", end="")
    for i in a:
        print("", i, end="")
    print("\n+ ", end="")
    for i in b:
        print("", i, end="")
    print("\n" + 18 * "-" + "\n= ", end="")
    for i in res:
        print("", i, end="")
    print("\n~ ", end="")
    for i in res_pred:
        print("", i, end="")
    print("\n")


res = []
X = []

for i in range(int(2 ** 8)):
    for j in range(int(2 ** 8)):

        val = (i + j) & 0xff
        res.append(eight_bit(val))
        X.append(eight_bit(i) + eight_bit(j))


res = np.array(list(map(lambda x: [int(_) for _ in x], res)))
X = np.array(list(map(lambda x: [int(_) for _ in x], X)))

X_train, X_test, y_train, y_test = skms.train_test_split(X, res, test_size=0.4)

clf = MLPClassifier(hidden_layer_sizes=(200,), verbose=True)

clf.fit(X_train, y_train)

print(skm.accuracy_score(y_test, clf.predict(X_test)))

for i in range(len(y_test)):
    y_pred = clf.predict(X_test[i].reshape(1, -1))

    if not np.allclose(y_pred[0], y_test[i]):
        a = number_from_array(X_test[i, :8])
        b = number_from_array(X_test[i, 8:])
        res = number_from_array(y_test[i])
        res_pred = number_from_array(y_pred[0])

        print("{0} + {1} = {2} != {3}".format(a, b, res, res_pred))
        output_binary_addition(
            X_test[i, :8], X_test[i, 8:], y_test[i], y_pred[0]
        )
