import numpy as np
from h1_util import numerical_grad_check


def softmax(X):
    """
    Compute the softmax of each row of an input matrix (2D numpy array).

    the numpy functions amax, log, exp, sum may come in handy as well as the keepdims=True option and the axis option.
    Remember to handle the numerical problems as discussed in the description.
    You should compute lg softmax first and then exponentiate

    More precisely this is what you must do.

    For each row x do:
    compute max of x
    compute the log of the denominator sum for softmax but subtracting out the max i.e (log sum exp x-max) + max
    compute log of the softmax: x - logsum
    exponentiate that

    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation of the corresponding row in X i.e res[i, :] = softmax(X[i, :])
    """

    res = np.zeros(X.shape)
    ### YOUR CODE HERE
    rows = X.shape[0]

    for row in range(rows):
        xMax = np.amax(X[row])
        smlog = X[row] - xMax - np.log(np.sum(np.exp(X[row] - xMax)))
        res[row] = np.exp(smlog)
    ### END CODE
    return res


def one_in_k_encoding(vec, k):
    """One-in-k encoding of vector to k classes

    Args:
       vec: numpy array - data to encode
       k: int - number of classes to encode to (0,...,k-1)
    """
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc


class SoftmaxClassifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.W = None

    def cost_grad(self, X, y, W):
        """
        Compute the average negative log likelihood cost and the gradient under the softmax model
        using data X, Y and weight matrix W.

        the functions np.log, np.nonzero, np.sum, np.dot (@), may come in handy
        Args:
           X: numpy array shape (n, d) float - the data each row is a data point
           y: numpy array shape (n, ) int - target values in 0,1,...,k-1
           W: numpy array shape (d x K) float - weight matrix
        Returns:
            totalcost: Average Negative Log Likelihood of w
            gradient: The gradient of the average Negative Log Likelihood at w
        """
        cost = np.nan
        grad = np.zeros(W.shape) * np.nan
        Yk = one_in_k_encoding(y, self.num_classes)  # may help - otherwise you may remove it
        ### YOUR CODE HERE
        n = X.shape[0]
        XW = X @ W
        softmaxXW = softmax(XW)
        cost = -np.mean(np.log(Yk.T @ softmaxXW))
        grad = -1/n * (X.T @ (Yk - softmaxXW)) # rounding might be wrong

        ### END CODE
        return cost, grad

    def fit(self, X, Y, W=None, lr=0.01, epochs=10, batch_size=16):
        """
        Run Mini-Batch Gradient Descent on data X,Y to minimize the in sample error (1/n)NLL for softmax regression.
        Printing the performance every epoch is a good idea to see if the algorithm is working

        Args:
           X: numpy array shape (n, d) - the data each row is a data point
           Y: numpy array shape (n,) int - target labels numbers in {0, 1,..., k-1}
           W: numpy array shape (d x K)
           lr: scalar - initial learning rate
           batchsize: scalar - size of mini-batch
           epochs: scalar - number of iterations through the data to use

        Sets:
           W: numpy array shape (d, K) learned weight vector matrix  W
           history: list/np.array len epochs - value of cost function after every epoch. You know for plotting
        """
        if W is None:
            W = np.zeros((X.shape[1], self.num_classes))
        history = []
        ### YOUR CODE HERE

        # Number of batches (Iterations) per epoch
        numOfBatches = int(np.floor(X.shape[0] / batch_size))

        # Yk is the one in k encoding of Y
        Yk = one_in_k_encoding(Y, self.num_classes)
        yrows, ycols = Yk.shape

        # Here we do mini-batch stochastic gradient descent
        for _ in range(epochs):
            cost_best = float("inf")
            # Here we get our sample of size batch_size
            XY = np.hstack((X, Yk))
            XYShuff = np.random.permutation(X.shape[0])
            XYs = XYShuff[j * batch_size : (j + 1) * batch_size, :]
            XShuff = XYs[:, :-ycols]
            YShuff = XYs[:, :-ycols]
            

            for j in range(numOfBatches):

                # WE ARE HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # Here we compute g (gradient) and the new w
                eta = lr  # step size
                cost, grad = self.cost_grad(XShuff, YShuff, W)
                W = W - eta * grad

                if cost_best > cost:
                    cost_best = cost
                    self.W = W

                # We save the cost of the current epoch
            history.append(cost)

        ### END CODE
        self.W = W
        self.history = history

    def score(self, X, Y):
        """Compute accuracy of classifier on data X with labels Y

        Args:
           X: numpy array shape (n, d) - the data each row is a data point
           Y: numpy array shape (n,) int - target labels numbers in {0, 1,..., k-1}
        Returns:
           out: float - mean accuracy
        """
        out = 0
        ### YOUR CODE HERE
        
        out = np.mean(self.predict(X) == Y)
        
        ### END CODE
        return out

    def predict(self, X):
        """Compute classifier prediction on each data point in X

        Args:
           X: numpy array shape (n, d) - the data each row is a data point
        Returns
           out: np.array shape (n, ) - prediction on each data point (number in 0,1,..., num_classes-1)
        """
        out = None
        ### YOUR CODE HERE
        prediction = softmax(X @ self.W)
        for i in range(prediction.shape[0]):
            out[i] = np.argmax(prediction[i])
        ### END CODE
        return out


def test_encoding():
    print("*" * 10, "test encoding")
    labels = np.array([0, 2, 1, 1])
    m = one_in_k_encoding(labels, 3)
    res = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
    assert res.shape == m.shape, "encoding shape mismatch"
    assert np.allclose(m, res), m - res
    print("Test Passed")


def test_softmax():
    print("Test softmax")
    X = np.zeros((3, 2))
    X[0, 0] = np.log(4)
    X[1, 1] = np.log(2)
    print("Input to Softmax: \n", X)
    sm = softmax(X)
    expected = np.array([[4.0 / 5.0, 1.0 / 5.0], [1.0 / 3.0, 2.0 / 3.0], [0.5, 0.5]])
    print("Result of softmax: \n", sm)
    assert np.allclose(expected, sm), "Expected {0} - got {1}".format(expected, sm)
    print("Test softmax complete")


def test_grad():
    print("*" * 5, "Testing  Gradient")
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
    w = np.ones((2, 3))
    y = np.array([0, 1, 2])
    scl = SoftmaxClassifier(num_classes=3)
    f = lambda z: scl.cost_grad(X, y, W=z)
    numerical_grad_check(f, w)
    print("Test Success")

def test_grad_2():
    print("*" * 5, "Testing  Gradient  2")
    X = np.array([[10.0, 2.0], [7.0, 4.0], [3.0, -8.0]])
    w = np.ones((2, 3))
    y = np.array([0, 1, 2])
    scl = SoftmaxClassifier(num_classes=3)
    f = lambda z: scl.cost_grad(X, y, W=z)
    numerical_grad_check(f, w)
    print("Test Success")


if __name__ == "__main__":
    test_encoding()
    test_softmax()
    # test_grad_2()
    test_grad()
