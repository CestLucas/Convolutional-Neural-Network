#!/usr/bin/env python
# coding: utf-8

# In[13]:


import sys
import pickle
import numpy as np
import gzip

def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]


def load_mnist():
    data_file = gzip.open("mnist.pkl.gz", "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_results = [one_hot(y, 10) for y in train_data[1]]
    train_data = np.array(train_inputs).reshape(-1, 784), np.array(train_results).reshape(-1, 10)

    val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_results = [one_hot(y, 10) for y in val_data[1]]
    val_data = np.array(val_inputs).reshape(-1, 784), np.array(val_results).reshape(-1, 10)

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return train_data, val_data, test_data


class NN(object):
    def __init__(self,
                 hidden_dims=(784, 256),
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=64,
                 seed=None,
                 activation="relu",
                 init_method="glorot",
                 data=None
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if data is None:
            # for testing, do NOT remove or modify
            self.train, self.valid, self.test = (
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400)))
        )
        else:
            self.train, self.valid, self.test = data


    def initialize_weights(self, dims):        
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            n_inputs = all_dims[layer_n-1]
            n_outputs = all_dims[layer_n]
            n_weights = n_inputs * n_outputs
            
            if self.init_method=="zero":
                init_weights = np.zeros(n_weights)
            elif self.init_method=="normal":
                init_weights = np.random.standard_normal(n_weights)
            elif self.init_method=="glorot":
                bound = np.sqrt(6.0/(n_inputs+n_outputs))
                init_weights = np.random.uniform(-bound, bound, n_weights)
            else:
                raise Exception("invalid init_method")
            
            # weight dimensions: (layer_n-1(input), layer_n(this layer))
            self.weights[f"W{layer_n}"] = np.reshape(init_weights, (n_inputs,-1))
            # bias dimensions: (layer_n, )
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

            
    def relu(self, x, grad=False):
        def helper(m):
            return 1 if m > 0 else 0
        
        if grad:
            return np.vectorize(helper)(x)
        
        return np.maximum(0,x)

    
    def sigmoid(self, x, grad=False):
        def helper(m):
            return 1.0 / (1.0 + np.exp(-m))
            
        if grad:
            # sigmoid(x) = sigmoid(x)(1-sigmoid(x))
            sigmoid = self.sigmoid(x)
            return sigmoid * (1.0 - sigmoid)
        
        return np.vectorize(helper)(x)
    

    def tanh(self, x, grad=False):
        if grad:
            # tanh'(x) = 1 - tanh(x) ^ 2
            return 1.0 - self.tanh(x) ** 2
        
        return np.tanh(x)
    

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        else:
            raise Exception("invalid")
        return 0

    
    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        exps = np.exp(x - np.max(x))
        return exps/exps.sum(axis=1, keepdims=True) if x.ndim > 1 else exps/exps.sum()

    
    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        
        n_layers = self.n_hidden + 2
        for layer_n in range(1, n_layers):
            # input dimensions: (batch_size, features)
            # weight dimensions: (layer_n-1(input), layer_n(this layer))
            x = cache[f"Z{layer_n-1}"]
            w = self.weights[f"W{layer_n}"]
            b = self.weights[f"b{layer_n}"]

            cache[f"A{layer_n}"] = x @ w + b
            
            if layer_n != n_layers-1:
                cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"])
            else:  # use softmax activation for the output layer
                cache[f"Z{layer_n}"] = self.softmax(cache[f"A{layer_n}"])
        
        return cache
    
    
    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        
        grads[f"dA{self.n_hidden+1}"] = output - labels  # shape (batch_size, n_classes)
        
        grads[f"dW{self.n_hidden+1}"] = cache[f"Z{self.n_hidden}"].T @ grads[f"dA{self.n_hidden+1}"]                                         / self.batch_size
        
        grads[f"db{self.n_hidden+1}"] = np.mean(grads[f"dA{self.n_hidden+1}"], axis = 0, keepdims = True)
        
        
        for l in range(self.n_hidden, 0, -1):
            
            grads[f"dZ{l}"] = grads[f"dA{l+1}"] @ self.weights[f"W{l+1}"].T
            
            grads[f"dA{l}"] = grads[f"dZ{l}"] * self.activation(cache[f"A{l}"], grad=True)
            
            grads[f"dW{l}"] = cache[f"Z{l-1}"].T @ grads[f"dA{l}"] / self.batch_size
           
            grads[f"db{l}"] = np.mean(grads[f"dA{l}"], axis = 0, keepdims = True)

        return grads
    

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] -= self.lr * grads[f"dW{layer}"]
            self.weights[f"b{layer}"] -= self.lr * grads[f"db{layer}"]
        

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        
        N = prediction.shape[0]
        return -np.sum(labels*np.log(prediction))/N

    
    def compute_loss_and_accuracy(self, X, y):
        one_y = y
        y = np.argmax(y, axis=1)  # Change y to integers
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    
    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = y_train
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                #sys.stdout.flush()
                #sys.stdout.write(f"\repoch{epoch}: batch {batch} in {n_batches} batches.")
                
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                
                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)
            
            #print(f"\ntrain_accuracy: {train_accuracy}, train_loss: {train_loss}, valid_accuracy: {valid_accuracy}, valid_loss: {valid_loss}")

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy


# In[15]:


#mnist_zero = NN(init_method="zero", seed=42)
#log_zero = mnist_zero.train_loop(n_epochs=10)


# In[ ]:




