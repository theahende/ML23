import numpy as np
from h1_util import numerical_grad_check


def logistic(z):
    """
    Helper function
    Computes the logistic function 1/(1+e^{-x}) to each entry in input vector z.

    np.exp may come in handy
    Args:
        z: numpy array shape (d,)
    Returns:
       logi: numpy array shape (d,) each entry transformed by the logistic function
    """
    logi = np.zeros(z.shape)
    ### YOUR CODE HERE

    # Rows is equal to the amount of points we want to apply the sigmoid function on
    rows = z.shape[0]

    # We take the sigmoid function each index in z and save it on the equivalent index in logi,
    # which we return.
    for i in range(rows):
        logi[i] = 1 / (1 + np.exp(-z[i]))
    ### END CODE
    assert logi.shape == z.shape
    return logi


class LogisticRegressionClassifier:
    def __init__(self):
        self.w = None

    def cost_grad(self, X, y, w):
        """
        Compute the average negative log likelihood and gradient under the logistic regression model
        using data X, targets y, weight vector w

        np.log, np.sum, np.choose, np.dot may be useful here
        Args:
           X: np.array shape (n,d) float - Features
           y: np.array shape (n,)  int - Labels
           w: np.array shape (d,)  float - Initial parameter vector

        Returns:
           cost: scalar: the average negative log likelihood for logistic regression with data X, y
           grad: np.array shape(d, ) gradient of the average negative log likelihood at w
        """

        cost = 0
        grad = np.zeros(w.shape)

        ### SECOND TRY START
        Xw = np.dot(X, w)  # (n,1)
        XT = X.T  # (d,n)

        yXw = y * Xw  # position wise multiplication, (n,1)
        cost = np.mean(np.log(1 + np.exp(-yXw)))

        # The y should have negative sign, but it doesn't work if we have it
        grad = ( XT @ ( -y * logistic(-yXw) ) ) / X.shape[0]

        ### SECOND TRY END
        assert grad.shape == w.shape
        return cost, grad

    def fit(self, X, y, w=None, lr=0.1, batch_size=16, epochs=10):
        """
        Run mini-batch stochastic Gradient Descent for logistic regression
        use batch_size data points to compute gradient in each step.

        The function np.random.permutation may prove useful for shuffling the data before each epoch
        It is wise to print the performance of your algorithm at least after every epoch to see if progress is being made.
        Remember the stochastic nature of the algorithm may give fluctuations in the cost as iterations increase.

        Args:
           X: np.array shape (n,d) dtype float32 - Features
           y: np.array shape (n,) dtype int32 - Labels
           w: np.array shape (d,) dtype float32 - Initial parameter vector
           lr: scalar - learning rate for gradient descent
           batch_size: number of elements to use in minibatch
           epochs: Number of scans through the data

        sets:
           w: numpy array shape (d,) learned weight vector w
           history: list/np.array len epochs - value of loss function (in-sample error) after every epoch. Used for plotting
        """
        if w is None:
            w = np.zeros(X.shape[1])
        history = []

        ### YOUR CODE HERE

        # Initialize w from N(0,1)
        w = np.random.standard_normal(w.shape[0])

        # Number of batches (Iterations) per epoch
        numOfBatches = int(np.floor(X.shape[0] / batch_size))

        # Here we do mini-batch stochastic gradient descent
        for _ in range(epochs):
            cost_best = float("inf")
            # Here we get our sample of size batch_size
            Xy = np.hstack((X, y[:, np.newaxis]))
            XyShuff = np.random.permutation(Xy)

            for j in range(numOfBatches):
                Xys = XyShuff[j * batch_size : (j + 1) * batch_size, :]
                XShuff = Xys[:, :-1]
                yShuff = Xys[:, -1]

                # Here we compute g (gradient) and the new w
                eta = 0.1  # step size
                cost, g = self.cost_grad(XShuff, yShuff, w)
                w = w - eta * g

                if cost_best > cost:
                    cost_best = cost
                    self.w = w

                # We save the cost of the current epoch
            history.append(cost)

        ### END CODE
        self.w = w
        self.history = history

    def predict(self, X):
        """Classify each data element in X.

        Args:
            X: np.array shape (n,d) dtype float - Features

        Returns:
           p: numpy array shape (n, ) dtype int32, class predictions on X (-1, 1). NOTE: We want a class here,
           not a probability between 0 and 1. You should thus return the most likely class!

        """
        ### YOUR CODE HERE
        z = X @ self.w.T
        probabilities = logistic(z)
        print(probabilities)
        prediction = np.sign(probabilities - 0.5)
        for i in range(len(prediction)):
            if prediction[i] == 0:
                prediction[i] = 1
        ### END CODE
        return prediction

    def score(self, X, y):
        """Compute model accuracy  on Data X with labels y

        Args:
            X: np.array shape (n,d) dtype float - Features
            y: np.array shape (n,) dtype int - Labels

        Returns:
           s: float, number of correct predictions divided by n. NOTE: This is accuracy, not in-sample error!

        """
        score = 0
        ### YOUR CODE HERE
        rows, _ = X.shape
        correct_predictions = 0
        prediction = self.predict(X)
        print("Prediction:", prediction)
        print("Y:", y)
        for i in range(rows):
            if prediction[i] == y[i]:
                correct_predictions += 1

        # The fraction of prediction that are wrong
        score = correct_predictions / rows
        ### END CODE
        return score


def test_logistic():
    print("*" * 5, "Testing logistic function")
    a = np.array([0, 1, 2, 3])
    lg = logistic(a)
    target = np.array([0.5, 0.73105858, 0.88079708, 0.95257413])
    assert np.allclose(lg, target), "Logistic Mismatch Expected {0} - Got {1}".format(target, lg)
    print("Test Success!")


def test_cost():
    print("*" * 5, "Testing Cost Function")
    X = np.array([[1.0, 0.0], [1.0, 1.0], [3, 2]])
    y = np.array([-1, -1, 1], dtype="int64")
    w = np.array([0.0, 0.0])
    print("shapes", X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    cost, _ = lr.cost_grad(X, y, w)
    target = -np.log(0.5)
    assert np.allclose(cost, target), "Cost Function Error:  Expected {0} - Got {1}".format(target, cost)
    print("Test Success cost")


def test_grad():
    print("*" * 5, "Testing  Gradient")
    X = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    w = np.array([0.0, 0.0])
    y = np.array([-1, -1, 1]).astype("int64")
    print("shapes", X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    f = lambda z: lr.cost_grad(X, y, w=z)
    numerical_grad_check(f, w)
    print("Test Success")


if __name__ == "__main__":
    test_logistic()
    test_cost()
    test_grad()
