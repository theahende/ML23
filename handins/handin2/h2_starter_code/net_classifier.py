import numpy as np

def one_in_k_encoding(vec, k):
    """ One-in-k encoding of vector to k classes 
    
    Args:
       vec: numpy array - data to encode
       k: int - number of classes to encode to (0,...,k-1)
    """
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc

def derive_relu(x, z):
    der_z = x
    der_z[z <= 0] = 0
    return der_z

def softmax(X):
    """ 
    You can take this from handin I
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
    
    You can do all of it without for loops using numpys vectorized operations.

    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation of the corresponding row in X i.e res[i, :] = softmax(X[i, :])
    """
    res = np.zeros(X.shape)
    ### YOUR CODE HERE
    rows = X.shape[0]

    for row in range(rows):
        xMax = np.max(X[row])
        smlog = X[row] - xMax - np.log(np.sum(np.exp(X[row] - xMax)))
        res[row] = np.exp(smlog)
    ### END CODE
    return res

def relu(x):
    """ Compute the relu activation function on every element of the input
    
        Args:
            x: np.array
        Returns:
            res: np.array same shape as x
        Beware of np.max and look at np.maximum
    """
    ### YOUR CODE HERE
    res = np.maximum(0, x)
    ### END CODE
    return res

def make_dict(W1, b1, W2, b2):
    """ Trivial helper function """
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def get_init_params(input_dim, hidden_size, output_size):
    """ Initializer function using Xavier/he et al Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

    Args:
      input_dim: int
      hidden_size: int
      output_size: int
    Returns:
       dict of randomly initialized parameter matrices.
    """
    W1 = np.random.normal(0, np.sqrt(2./(input_dim+hidden_size)), size=(input_dim, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.normal(0, np.sqrt(4./(hidden_size+output_size)), size=(hidden_size, output_size))
    b2 = np.zeros((1, output_size))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

  
class NetClassifier():
    
    def __init__(self):
        """ Trivial Init """
        self.params = None
        self.hist = None

    def predict(self, X, params=None):
        """ Compute class prediction for all data points in class X
        
        Args:
            X: np.array shape n, d
            params: dict of params to use (if none use stored params)
        Returns:
            np.array shape n, 1
        """
        if params is None:
            params = self.params
        pred = None
        ### YOUR CODE HERE
        hidden_layer = relu(X @ params['W1'] + params['b1'])
        out_layer = softmax(hidden_layer @ params['W2'] + params['b2'])
        pred = np.argmax(out_layer, axis=1)
        ### END CODE
        return pred
     
    def score(self, X, y, params=None):
        """ Compute accuracy of model on data X with labels y (mean 0-1 loss)
        
        Args:
            X: np.array shape n, d
            y: np.array shape n, 1
            params: dict of params to use (if none use stored params)

        Returns:
            acc: float, number of correct predictions divided by n. NOTE: This is accuracy, not in-sample error!
        """
        if params is None:
            params = self.params
        acc = None
        ### YOUR CODE HERE
        acc: float = np.mean(self.predict(X, params) == y)
        ### END CODE
        return acc
    
    @staticmethod
    def cost_grad(X, y, params, c=0.0):
        """ Compute cost and gradient of neural net on data X with labels y using weight decay parameter c
        You should implement a forward pass and store the intermediate results 
        and then implement the backwards pass using the intermediate stored results
        
        Use the derivative for cost as a function for input to softmax as derived above
        
        Args:
            X: np.array shape n, self.input_size
            y: np.array shape n, 1
            params: dict with keys (W1, W2, b1, b2)
            c: float - weight decay parameter
            params: dict of params to use for the computation
        
        Returns 
            cost: scalar - average cross entropy cost with weight decay parameter c
            dict with keys
            d_w1: np.array shape w1.shape, entry d_w1[i, j] = \partial cost/ \partial W1[i, j]
            d_w2: np.array shape w2.shape, entry d_w2[i, j] = \partial cost/ \partial W2[i, j]
            d_b1: np.array shape b1.shape, entry d_b1[1, j] = \partial cost/ \partial b1[1, j]
            d_b2: np.array shape b2.shape, entry d_b2[1, j] = \partial cost/ \partial b2[1, j]
            
        """
        cost = 0
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']
        d_w1 = None
        d_w2 = None
        d_b1 = None
        d_b2 = None
        labels = one_in_k_encoding(y, W2.shape[1]) # shape n x k
                        
        ### YOUR CODE HERE - FORWARD PASS - compute cost with weight decay and store relevant values for backprop
        batch_size = X.shape[0]
        # Forward pass
        g = X @ W1
        d = g + b1
        our_c = relu(d)

        e = our_c @ W2
        z = e + b2
        sm_z = softmax(z)

        sm_z_correct = np.choose(y, sm_z.T)
        cost = np.mean(-np.log(sm_z_correct)) + c * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        
        # Backwards pass
        d_sm_z = sm_z - labels #  # Why can we neglect the delta
        d_b2 = np.mean(d_sm_z, axis=0, keepdims=True) #??? 
        d_w2 = (our_c.T @ d_sm_z) / batch_size + c * 2 * W2 # WHY divide by batch size

        d_sm_z_2 = d_sm_z @ W2.T 
        d_relu = derive_relu(d_sm_z_2, d)
        d_b1 = np.mean(d_relu, axis=0, keepdims=True)
        d_w1 = (X.T @ d_relu) / batch_size + 2 * c * W1
        ### END CODE
        # the return signature
        return cost, {'d_w1': d_w1, 'd_w2': d_w2, 'd_b1': d_b1, 'd_b2': d_b2}
        
    def fit(self, X_train, y_train, X_val, y_val, init_params, batch_size=32, lr=0.1, c=1e-4, epochs=30):
        """ Run Mini-Batch Gradient Descent on data X, Y to minimize the in sample error for Neural Net classification
        Printing the performance every epoch is a good idea to see if the algorithm is working
    
        Args:
           X_train: numpy array shape (n, d) - the training data each row is a data point
           y_train: numpy array shape (n,) int - training target labels numbers in {0, 1,..., k-1}
           X_val: numpy array shape (n, d) - the validation data each row is a data point
           y_val: numpy array shape (n,) int - validation target labels numbers in {0, 1,..., k-1}
           init_params: dict - has initial setting of parameters
           lr: scalar - initial learning rate
           batch_size: scalar - size of mini-batch
           c: scalar - weight decay parameter 
           epochs: scalar - number of iterations through the data to use

        Sets: 
           params: dict with keys {W1, W2, b1, b2} parameters for neural net
        returns
           hist: dict:{keys: train_loss, train_acc, val_loss, val_acc} each an np.array of size epochs of the the given cost after every epoch
           loss is the NLL loss and acc is accuracy
        """
        
        W1 = init_params['W1']
        b1 = init_params['b1']
        W2 = init_params['W2']
        b2 = init_params['b2']
        self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        hist = {
            'train_loss': None,
            'train_acc': None,
            'val_loss': None,
            'val_acc': None, 
        }

        
        ### YOUR CODE HERE
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        numOfBatches = int(np.floor(X_train.shape[0] / batch_size))        
        
        for _ in range(epochs):
            cost_best = float('inf')
            
            # We shuffle the indexes with same dimensions as X
            shuff_index = np.random.permutation(X_train.shape[0])
            
            for j in range(numOfBatches):
                shuff_batch = shuff_index[j * batch_size : (j + 1) * batch_size]
                X_batch = X_train[shuff_batch]
                Y_batch = y_train[shuff_batch]
                
                eta = lr
                cost, grad = self.cost_grad(X_batch, Y_batch, self.params, c)
                W1 = W1 - eta * grad['d_w1']
                W2 = W2 - eta * grad['d_w2']
                b1 = b1 - eta * grad['d_b1']
                b2 = b2 - eta * grad['d_b2']
                
                if cost_best > cost:
                    cost_best = cost
                    self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
                    
            train_loss.append(cost)
            train_acc.append(self.score(X_train, y_train))
            val_loss.append(self.cost_grad(X_val, y_val, self.params)[0])
            val_acc.append(self.score(X_val, y_val))
            
        # hist['train_loss'] = train_loss
        # hist['train_acc'] = train_acc
        # hist['val_loss'] = val_loss
        # hist['val_acc'] = val_acc
        
        ### END CODE
        # hist dict should look like this with something different than none
        hist = {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}
        ## self.params should look like this with something better than none, i.e. the best parameters found.
        # self.params = {'W1': None, 'b1': None, 'W2': None, 'b2': None}
        return hist
        

def numerical_grad_check(f, x, key):
    """ Numerical Gradient Checker """
    eps = 1e-6
    h = 1e-5
    # d = x.shape[0]
    cost, grad = f(x)
    grad = grad[key]
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:    
        dim = it.multi_index    
        print(dim)
        tmp = x[dim]
        x[dim] = tmp + h
        cplus, _ = f(x)
        x[dim] = tmp - h 
        cminus, _ = f(x)
        x[dim] = tmp
        num_grad = (cplus-cminus)/(2*h)
        # print('cplus cminus', cplus, cminus, cplus-cminus)
        # print('dim, grad, num_grad, grad-num_grad', dim, grad[dim], num_grad, grad[dim]-num_grad)
        assert np.abs(num_grad - grad[dim]) < eps, 'numerical gradient error index {0}, numerical gradient {1}, computed gradient {2}'.format(dim, num_grad, grad[dim])
        it.iternext()

def test_grad():
    stars = '*'*5
    print(stars, 'Testing  Cost and Gradient Together')
    input_dim = 7
    hidden_size = 1
    output_size = 3
    nc = NetClassifier()
    params = get_init_params(input_dim, hidden_size, output_size)

    nc = NetClassifier()
    X = np.random.randn(7, input_dim)
    y = np.array([0, 1, 2, 0, 1, 2, 0])

    f = lambda z: nc.cost_grad(X, y, params, c=1.0)
    print('\n', stars, 'Test Cost and Gradient of b2', stars)
    numerical_grad_check(f, params['b2'], 'd_b2')
    print(stars, 'Test Success', stars)
    
    print('\n', stars, 'Test Cost and Gradient of w2', stars)
    numerical_grad_check(f, params['W2'], 'd_w2')
    print('Test Success')
    
    print('\n', stars, 'Test Cost and Gradient of b1', stars)
    numerical_grad_check(f, params['b1'], 'd_b1')
    print('Test Success')
    
    print('\n', stars, 'Test Cost and Gradient of w1', stars)
    numerical_grad_check(f, params['W1'], 'd_w1')
    print('Test Success')

if __name__ == '__main__':
    input_dim = 3
    hidden_size = 5
    output_size = 4
    batch_size = 7
    nc = NetClassifier()
    params = get_init_params(input_dim, hidden_size, output_size)
    X = np.random.randn(batch_size, input_dim)
    Y = np.array([0, 1, 2, 0, 1, 2, 0])
    nc.cost_grad(X, Y, params, c=0)
    test_grad()
