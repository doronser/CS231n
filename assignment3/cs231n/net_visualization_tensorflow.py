import tensorflow as tf
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, numpy array of shape (N, H, W, 3)
    - y: Labels for X, numpy of shape (N,)
    - model: A SqueezeNet model that will be used to compute the saliency map.

    Returns:
    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
    input images.
    """
    saliency = None
    # Compute the score of the correct class for each example.
    # This gives a Tensor with shape [N], the number of examples.
    #
    # Note: this is equivalent to scores[np.arange(N), y] we used in NumPy
    # for computing vectorized losses.

    ###############################################################################
    # TODO: Produce the saliency maps over a batch of images.                     #
    #                                                                             #
    # 1) Define a gradient tape object and watch input Image variable             #
    # 2) Compute the “loss” for the batch of given input images.                  #
    #    - get scores output by the model for the given batch of input images     #
    #    - use tf.gather_nd or tf.gather to get correct scores                    #
    # 3) Use the gradient() method of the gradient tape object to compute the     #
    #    gradient of the loss with respect to the image                           #
    # 4) Finally, process the returned gradient to compute the saliency map.      #
    ###############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, H, W, _ = X.shape
    X_tf = tf.Variable(X)
    y_tf = tf.Variable(y)
    with tf.GradientTape() as tape:
      scores = model(X_tf)
      # correct_scores = tf.gather_nd(scores, tf.stack((tf.range(N), y_tf), axis=1))
      correct_scores = tf.reduce_max(scores, axis=1)
      grad = tape.gradient(correct_scores,X_tf)
      grad = grad.numpy()
      saliency = np.max(np.abs(grad), axis=3)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency

def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image, a numpy array of shape (1, 224, 224, 3)
    - target_y: An integer in the range [0, 1000)
    - model: Pretrained SqueezeNet model

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """

    # Make a copy of the input that we will modify
    X_fooling = X.copy()

    # Step size for the update
    learning_rate = 1

    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. Use gradient *ascent* on the target class score, using #
    # the model.scores Tensor to get the class scores for the model.image.   #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop, where in each iteration, you make an     #
    # update to the input image X_fooling (don't modify X). The loop should      #
    # stop when the predicted class for the input is the same as target_y.       #
    #                                                                            #
    # HINT: Use tf.GradientTape() to keep track of your gradients and            #
    # use tape.gradient to get the actual gradient with respect to X_fooling.    #
    #                                                                            #
    # HINT 2: For most examples, you should be able to generate a fooling image  #
    # in fewer than 100 iterations of gradient ascent. You can print your        #
    # progress over iterations to check your algorithm.                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X_tf = tf.Variable(X_fooling)
    y_tf = tf.Variable(target_y)
    N, H, W, _ = X.shape
    for step in range(100):
      with tf.GradientTape() as tape:
        tape.watch(X_tf)
        scores = model(X_tf)  # forward pass
        if step == 0:
          original_class = tf.math.argmax(scores, axis=1)
          original_class = original_class[0]
          print(f"original class is {original_class}")
        pred_class = tf.math.argmax(scores, axis=1)
        
        # stop when network is fooled
        if pred_class == target_y:
          X_fooling = X_tf
          print(f"Network fooled after {step} iterations")
          print(f"Iteration #{step}: fool class score: {scores[:,target_y].numpy()} correct class score: {scores[:,original_class].numpy()}")
          break
        else:
          # perform gradient ascent step
          # calculate grad of target score W.R.T the image
          target_score = scores[:,target_y]
          grad = tape.gradient(target_score,X_tf)
          dX = learning_rate * grad / tf.norm(grad, ord=2)
          X_tf = X_tf + dX
      
      # Log every 10 iterations.
      if step % 10 == 0:
          print(f"Iteration #{step}: fool class score: {target_score.numpy()} correct class score: {scores[:,original_class].numpy()}")


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling

def class_visualization_update_step(X, model, target_y, l2_reg, learning_rate):
    ########################################################################
    # TODO: Compute the value of the gradient of the score for             #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. You should use   #
    # the tf.GradientTape() and tape.gradient to compute gradients.        #
    #                                                                      #
    # Be very careful about the signs of elements in your code.            #
    ########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X_tf = tf.Variable(X)
    y_tf = tf.Variable(target_y)
    with tf.GradientTape() as tape:
      tape.watch(X_tf)
      scores = model(X_tf)  # forward pass
      target_score = scores[:, target_y]
      reg_term = l2_reg * tf.norm(X_tf)**2
      pseudo_loss =  target_score - reg_term
      grad = tape.gradient(pseudo_loss,X_tf)
      X_tf = X_tf + learning_rate * grad

    X = X_tf.numpy()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return X

def blur_image(X, sigma=1):
    X = gaussian_filter1d(X, sigma, axis=1)
    X = gaussian_filter1d(X, sigma, axis=2)
    return X

def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: Tensor of shape (N, H, W, C)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new Tensor of shape (N, H, W, C)
    """
    if ox != 0:
        left = X[:, :, :-ox]
        right = X[:, :, -ox:]
        X = tf.concat([right, left], axis=2)
    if oy != 0:
        top = X[:, :-oy]
        bottom = X[:, -oy:]
        X = tf.concat([bottom, top], axis=1)
    return X
