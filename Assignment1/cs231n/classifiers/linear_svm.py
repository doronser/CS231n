from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]] 
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
            
                # gradient computation: when the margin is smaller than the desired value (delta)
                # increment x[i] for rows of dData that are not the correct class
                # decrement x[i]*(num of classes with bad margin) for the row of dData that correspond to the correct class
                dW[:,j] += X[i,:] 
                dW[:,y[i]] -= X[i,:]
                

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW   /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_samples = y.shape[0]
    # scores is NxC matrix where each row is a 1xD vector
    # of scores for a single training example
    scores = X @ W

    # true scores is a Nx1 vector of the true class score for each training example
    true_scores = scores[np.arange(num_samples),y].reshape(num_samples,-1)
    margin = scores - true_scores + 1
    
    # in order to transition from margin to loss, we need to remove negative values (just like max(0,margin) would)
    # and make sure the correct class does not contribute to the loss
    loss_mat = margin.clip(min=0) # max(0,margin) is equivalent to clipping the array with min=0
    loss_mat[np.arange(num_samples),y] = 0 # the correct class should not contribute any loss
    loss = np.sum(loss_mat) # sum up the loss matrix
    loss /= num_samples # divide by N to average over Li
    loss += reg * np.sum(W * W) # add the reg term

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # X.T @ help_mat would give us the desired result
    # each column in help_mat represents the the gradient of a single class, which is built in 2 steps: 
    # 1. for training examples that don't correspond to the current column class, the gradient is their mean
    # 2. for training examples that correspond to the current column class, the gradient is a weighted sum of examples in
    # where the weight of every example is the number of wrong classes whose margin is bigger than 0.

    # the 1st gradient is the sum of all columns in the loss matrix with a positive margin
    help_mat = loss_mat
    help_mat[loss_mat > 0] = 1

    #the 2nd gradient for each example is the sum of all classes with margin>1 times X
    row_sum = np.sum(help_mat, axis=1)
    help_mat[np.arange(num_samples),y] = -row_sum.T
    
    # multiply X by the help matrix to obtain the sum of relevant columns/rows
    dW = X.T @ help_mat
    dW /= num_samples
     
    #add the reg term
    dW+= 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
