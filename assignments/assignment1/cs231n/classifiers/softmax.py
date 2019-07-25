import numpy as np


def softmax(j, z, axis=0):
    return np.exp(j) / np.exp(z).sum(axis=axis).reshape(-1,1)
    
    
def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
 
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
         that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    num_class = W.shape[1]
    
    # Compute loss of each training example
    for i in range(num_train):
        scores = X[i] @ W # (num_class, )
        shift = -scores.max()
        scores += shift
        correct_class_score = scores[y[i]] #(1, )
        loss += -np.log(softmax(correct_class_score, scores))
        
        for j in range(num_class):
            term = softmax(scores[j], scores).ravel()
            if j == y[i]:
                dW[:, j] += (term - 1) * X[i]
            else:
                dW[:, j] += term * X[i]
        
    loss /= num_train
    dW /= num_train
    
    # Add regularization
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                               END OF YOUR CODE                            #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    
    scores = X @ W # (num_train, num_class)
    shift = -scores.max(axis=1).reshape(num_train, 1) # (num_train, 1)
    scores += shift
    softmax_term = softmax(scores, scores, axis=1) # (num_train, num_class)
    
    loss = -np.log(softmax_term[np.arange(num_train), y]).sum(axis=0)
    softmax_term[np.arange(num_train), y] -= 1
    dW = X.T @ softmax_term 
    
    loss /= num_train
    dW /= num_train
    
    # Add regularization
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    #                               END OF YOUR CODE                            #
    #############################################################################

    return loss, dW
