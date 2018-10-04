import numpy as np
import sklearn.metrics as skm
import sklearn.model_selection as skms
from sklearn.neural_network import MLPClassifier


np.random.seed(2018)


def eight_bit(number):
    """Function creating a string of the first eight bits from a given
    number. For instance:

        1 -> 00000001
        2 -> 00000010
        3 -> 00000011

    """
    return bin(number & 0xff)[2:].zfill(8)


def binary_array(number):
    """Function creating a bit array of an eight bit string"""
    bin_str = eight_bit(number)
    return np.array([int(_) for _ in bin_str])


def number_from_array(arr):
    """Function computing the decimal number represented by an eight bit binary
    array"""
    val = 0

    for i in range(len(arr)):
        val += arr[i] << (7 - i)

    return val


def output_binary_addition(a, b, res, res_pred):
    """Function printing the addition of two eight bit binary numbers (as
    binary arrays) and comparing it to two results, where res should be
    correct."""
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

# Generate the truth table for addition of two bytes to a single byte
for i in range(int(2 ** 8)):
    for j in range(int(2 ** 8)):

        # Make sure the value does not exceed a byte, i.e., 0xff
        val = (i + j) & 0xff
        res.append(eight_bit(val))
        X.append(eight_bit(i) + eight_bit(j))


# Create the binary array representation of the bytes
res = np.array(list(map(lambda x: [int(_) for _ in x], res)))
X = np.array(list(map(lambda x: [int(_) for _ in x], X)))

X_train, X_test, y_train, y_test = skms.train_test_split(X, res, test_size=0.4)

# Create a neural net
clf = MLPClassifier(hidden_layer_sizes=(100, 200), max_iter=28, verbose=True)
# Train the neural net on this daunting task!
clf.fit(X_train, y_train)

print(
    "Accuracy on test data: {0}".format(
        skm.accuracy_score(y_test, clf.predict(X_test))
    )
)

# Go through each prediction iteratively
for i in range(len(y_test)):
    y_pred = clf.predict(X_test[i].reshape(1, -1))

    # Only print results which are wrong
    if not np.allclose(y_pred[0], y_test[i]):
        a = number_from_array(X_test[i, :8])
        b = number_from_array(X_test[i, 8:])
        res = number_from_array(y_test[i])
        res_pred = number_from_array(y_pred[0])

        print("{0} + {1} = {2} != {3}".format(a, b, res, res_pred))
        output_binary_addition(
            X_test[i, :8], X_test[i, 8:], y_test[i], y_pred[0]
        )
