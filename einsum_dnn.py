"""Implementation of a 2-layer DNN with numpy and einsum"""

import numpy as np
from scipy.special import logsumexp, expit, softmax
from sklearn.utils import shuffle
from sys import stderr


### Data functions
def vectorize(numerical_category, dim):
    '''Vectorize numerical categories'''
    temp = np.zeros(dim)
    temp[numerical_category] = 1
    return temp

# This data set reshape works
def batchify_2d(twodee, batch_size, featlen=-1):
    """Reshape a 2D matrix with training data. If the number of samples is not
    divisible by the batch size, we discard the last M samples from the training set.
    We return the remainder for testing purposes (but the remainder is not sufficient
    as a test set because it is smaller than a minibatch)"""
    nbatch = int(len(twodee)/ batch_size)
    return twodee[:batch_size * nbatch, :].reshape(nbatch, batch_size, featlen)

def maxnorm_array(array):
    """Return a matrix normalised by the maximum"""
    return array - np.max(array)

def maxnorm_dict(dict_of_arrays):
    """Applies maxnorn to a dictionary of arrays"""
    for key in dict_of_arrays:
        dict_of_arrays[key] = maxnorm_array(dict_of_arrays[key])

def sigmoid(hidden_activations):
    """Compute the sigmoid function"""
    # expit(x) = 1/(1+exp(-x)
    return expit(hidden_activations)

def sigmoidgrad(hidden_activations):
    """Compute the gradient if the sigmoid"""
    maxnormed = maxnorm_array(hidden_activations)
    return maxnormed * (1-maxnormed)

def softmax_crossentropy(logits, y_true):
    """ Compute crossentropy from logits[batch,n_classes] and ids of correct answers.
    y_true is NOT one-hot encoded"""
    logits_for_y_true = logits[np.arange(len(logits)), y_true]
    xentropy = -logits_for_y_true + logsumexp(logits, axis=-1)
    return xentropy


# def grad_softmax_crossentropy_with_logits(logits, y_true):
#     # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
#     ones_for_answers = np.zeros_like(logits)
#     ones_for_answers[np.arange(len(logits)), y_true] = 1
#     softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
#     return (- ones_for_answers + softmax) / logits.shape[0]

def clip(paramdict, minimum=-50, maximum=50):
    """Clip gradients to a specified range"""
    for k in paramdict:
        paramdict[k] = np.clip(paramdict[k], minimum, maximum)

def cross_entropy(pred, true, eps=1e-15):
    """Computes xent loss"""
    num_examples = true.shape[0]
    # print("XWNT", true.shape, pred.shape)
    # Get the softmax probability for each correct label for each sample
    pred += eps
    xent_loss = np.where(true==1, -np.log(pred), 0).sum() / num_examples
    error = np.where(true==1, -1/pred, 0) / num_examples
    return xent_loss, error


class FFNet(object):
    """A DNN model implemented with np.einsum"""
    def __init__(self, batch_size, feature_dim, hidden_dim, output_dim):
        self.nonlin = sigmoid
        self.gradfn = sigmoidgrad
        self.batch_size = batch_size
        self.init_params(feature_dim, hidden_dim, output_dim)
        self.eval = False


    def init_params(self, feature_dim, hidden_dim, output_dim):
        """Initialize parameters"""
        self.params = {}
        self.params['V'] = np.random.randn(hidden_dim, output_dim) * np.sqrt(1./output_dim)
        self.params['W'] = np.random.randn(feature_dim, hidden_dim) * np.sqrt(1./hidden_dim)
        self.params['b'] = np.random.normal(size=(hidden_dim))
        self.params['c'] = np.random.normal(size=(output_dim,))


    def detect_nan(self, graddict=None, string=None):
        """Test whether there are NaNs in the param matrices"""
        # Test the model parameters
        for k in self.params:
            if np.any(np.isnan(self.params[k])):
                print(f'NaNs in parameter of {k} @ {string}', file=stderr)
                print(self.params[k], file=stderr)
                raise RuntimeError

        # Test other params like the gradients
        if graddict is not None:
            for k in graddict:
                if np.any(np.isnan(graddict[k])):
                    print(f'NaNs in gradient of {k} @ {string}', file=stderr)
                    raise RuntimeError


    def forward(self, batch, nonlinfunc=sigmoid):
        """The forward pass - batched version"""
        # batched fprop
        l1_ha = np.einsum('ih, Bi -> Bh', self.params['W'], batch) + self.params['b']
        l1_h = nonlinfunc(l1_ha)

        # Store the hidden states before activation if we are training
        if not self.eval:
            self.l1_h = l1_h
            self.l1_ha = l1_ha
        l2_ha = np.einsum('ho, Bh -> Bo', self.params['V'], l1_h) + self.params['c']
        return l2_ha


    def backward(self, error, inputbatch):
        """The backward pass - batched version"""
        # backward prop
        # Sanity check that we are training
        assert self.eval is False
        # print(error.shape, self.l1_h.shape)
        grad_v = np.einsum('Bo, Bh ->ho', error, self.l1_h) / self.batch_size
        grad_c = np.einsum('Bo ->o', error) / self.batch_size # new einsum exp
        # hidden_activations l1_ha is precomputed in the forward pass
        grad_ha = np.einsum('ho, Bo  -> Bh', self.params['V'], error) * self.gradfn(self.l1_ha)
        grad_w = np.einsum('Bi, Bh -> ih', inputbatch, grad_ha) / self.batch_size
        grad_b = np.einsum('Bh -> h', grad_ha) / self.batch_size
        return {'V':grad_v, 'c':grad_c, 'W':grad_w, 'b':grad_b}


    def update(self, paramdict, learning_rate):
        """Update the model parameters with gradients computed in the backward pass"""
        for param in ('W', 'V'):  # weight matrices
            # Compute the update in tmp
            tmp = self.params[param] - np.einsum('ij, ->ij', paramdict[param], learning_rate)
            # Check that the update has changed the weights
            if np.array_equal(self.params[param], tmp):
                print(f'{param} was not updated', file=stderr)
            else:
                self.params[param] = tmp
        # Compute the bias of the hidden layer and check that the update changed the bias
        c_bias = self.params['c'] - np.einsum('i, -> i', paramdict['c'], learning_rate)
        if np.array_equal(self.params['c'], c_bias):
            print('c was not updated', file=stderr)
        else:
            self.params['c'] = c_bias

        # Compute the bias of the input layer and check that the update changed the bias
        b_bias = self.params['b'] - np.multiply(paramdict['b'], learning_rate)
        if np.array_equal(self.params['b'], b_bias):
            print('b was not updated', file=stderr)
        else:
            self.params['b'] = b_bias


    def train(self, xtrain, ytrain, num_epochs=10, learning_rate=0.01):
        """Train the parameters of the model and print progress stats"""
        assert len(xtrain) == len(ytrain)
        self.detect_nan(string='{}'.format(-1))
        losses = []
        for nepoch in range(num_epochs):
            for i, inputbatch in enumerate(xtrain):
                if i == 0:
                    x_eval = inputbatch
                    y_eval = ytrain[i]
                    continue
                logits = self.forward(inputbatch)
                logprobs = softmax(logits)
                loss, error = cross_entropy(logprobs, ytrain[i])
                losses.append(loss)
                #print(logprobs, ytrain[i], error)
                grads = self.backward(error, inputbatch)
                maxnorm_dict(grads)
                clip(grads)
                self.detect_nan(graddict=grads, string=f'{nepoch}/{i}')
                self.update(grads, learning_rate)
                #print('--')

            self.eval = True
            eval_pred = self.forward(x_eval)
            y_pred = np.argmax(eval_pred, axis=1) # axis=1 means pr row/sample in batch
            y_true = np.argmax(y_eval, axis=1)
            avg_loss = np.mean(losses)
            self.eval = False
            # print(y_true, y_pred)
            print('Epoch {} Avg. loss {} accuracy: {}'.format(nepoch, avg_loss, accuracy_score(y_true, y_pred)))


def main(data_input, data_target):
    """Run a test of the DNN training"""

    batch_size = 2
    hidden_dimension = 8
    n_classes = 10

    # Reshape images to vectors
    xtrain = data_input.reshape(len(data_input), -1)
    ytrain = np.array([vectorize(x, n_classes) for x in data_target])

    feature_length = len(xtrain[0])
    xtrain, ytrain = shuffle(xtrain, ytrain)
    xb_train = batchify_2d(xtrain, batch_size)
    yb_train = batchify_2d(ytrain, batch_size)

    output_dimension = n_classes
    model = FFNet(batch_size, feature_length, hidden_dimension, output_dimension)
    model.train(xb_train, yb_train, learning_rate=0.0001, num_epochs=120)

if __name__ == '__main__':
    #import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.metrics import classification_report, accuracy_score
    import random
    seed_value = 7777
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)    # 'python' randome number generator
    np.random.seed(seed_value) # 'numpy' ditto


    # load toy digits data set from sklearn
    DATA = datasets.load_digits()
    IMS = DATA.images[:]
    TARGETS = DATA.target[:]
    #plt.gray()
    #plt.matshow(DATA.images[0])
    del DATA
    main(IMS, TARGETS)
