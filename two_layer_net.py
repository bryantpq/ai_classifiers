import numpy as np
from past.builtins import xrange

class TwoLayeredNet(object):

    def __init__(self, input_s, hidden_s, output_s, std=1e-4):
        """
        Starts the model with small random values for weights and zeroes for bias 
        """
        self.params = {}
        self.params['W1'] = np.random.randn(input_s, hidden_s) * std
        self.params['b1'] = np.zeros(hidden_s)
        self.params['W2'] = np.random.randn(hidden_s, output_s) * std
        self.params['b2'] = np.zeros(output_s)

    def loss(self, X, y=None, r=0.0):
        """
        Calculates loss and gradient 
        """

        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        N, D = X.shape

        scores = None

        y_s = np.dot(X,W1) + b1
        h = np.maximum(y_s, 0)
        scores = np.dot(h, W2) + b2

        if y is None:
            return scores

        loss = None

        scores -= np.amax(scores)
        smax = np.exp(scores) / np.sum(np.exp(scores), axis = 1, keepdims = True)

        c_lprobs = -np.log(smax[range(N), y])
        c_loss = np.sum(c_lprobs) / N
        r_loss = r * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = c_loss + r_loss

        grads = {}

        dsmax = smax
        dsmax[range(N), y] -= 1
        dsmax /= N

        dW2 = h.T.dot(dsmax)
        dh = dsmax.dot(W2.T)
        dy_s = dh * (y_s >= 0)
        dW1 = X.T.dot(dy_s)
        db1 = np.sum(dy_s, axis = 0)
        db2 = np.sum(dsmax, axis = 0)

        grads['W1'] = dW1 + r * W1 * 2
        grads['W2'] = dW2 + r * W2 * 2
        grads['b1'] = db1
        grads['b2'] = db2

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Trains the model with stochastic gradient descent
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)


        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            r_indices = np.random.choice(num_train, batch_size)
            X_batch = X[r_indices]
            y_batch = y[r_indices]

            loss, grads = self.loss(X_batch, y=y_batch, r=reg)

            self.params['W1'] += -learning_rate * grads['W1']
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b1'] += -learning_rate * grads['b1']
            self.params['b2'] += -learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:
                # Decay learning rate
                learning_rate *= learning_rate_decay


    def predict(self, X):
        """
        Uses the network to predict labels for data. 
        """

        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        y_s = np.dot(X, W1) + b1
        h = np.maximum(y_s, 0)
        y_pred = np.argmax(np.dot(h, W2) + b2, axis=1)
        return y_pred