import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    self.params['W1'] = weight_scale * np.random.randn(num_filters,
        input_dim[0], filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale * np.random.randn(
        num_filters * input_dim[1] * input_dim[2] * 0.25, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################

    c1, c1_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    a2, a2_cache = affine_relu_forward(c1, W2, b2)
    scores, scores_cache = affine_forward(a2, W3, b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) +
        np.sum(W3 * W3))

    da1, grads['W3'], grads['b3'] = affine_backward(dscores, scores_cache)
    grads['W3'] += self.reg * W3
    dc1, grads['W2'], grads['b2'] = affine_relu_backward(da1, a2_cache)
    grads['W2'] += self.reg * W2
    _, grads['W1'], grads['b1'] = conv_relu_pool_backward(dc1, c1_cache)
    grads['W1'] += self.reg * W1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

class ConvNet(object):
  def __init__(self, input_dim=(3, 32, 32), pieces_patterns=None,
      num_filters=None, filter_sizes=None, strides=None, pads=None,
      hidden_dims=None, num_classes=10, weight_scale=1e-3, reg=0.0,
      dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - pieces_patterns: List of strings, describing each piece of architecture
      for this network, except input and output affine layer.
    - num_filters: List of numbers of filters to use in the convolutional
      layers.
    - filter_sizes: List of sizes of filters to use in the convolutional
      layers.
    - strides: List of strides to use in convolutional layers.
    - pads: List of pads to use in convolutional layers (supposed to not to
      decrease input size).
    - hidden_dims: List of numbers of units to use in the fully-connected
      hidden layers.
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}

    self.pieces_patterns = pieces_patterns
    self.num_filters = num_filters
    self.filter_sizes = filter_sizes
    self.hidden_dims = hidden_dims
    self.conv_pieces_num = len(num_filters)
    self.affine_pieces_num = len(hidden_dims)
    self.num_layers = self.conv_pieces_num + self.affine_pieces_num + 1

    if strides == None:
      strides = [1 for i in xrange(self.conv_pieces_num)]
    self.strides = strides
    if pads == None:
      pads = [(w - 1) / 2 for w in filter_sizes]
    self.pads = pads

    self.weight_scale = weight_scale
    self.reg = reg
    self.dtype = dtype

    self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers)]

    C, H, W = input_dim
    conv_i = 0
    affine_i = 0
    pool_pieces_num = 0
    for i in xrange(len(pieces_patterns)):
      stri = str(i + 1)
      pattern = pieces_patterns[i]
      # All layers suppose to have batch normalization
      # conv-relu or conv-relu-pool
      if 'cr' == pattern or 'crp' == pattern:
        self.__init_conv_piece(stri, num_filters[conv_i], C,
            filter_sizes[conv_i])
        C = num_filters[conv_i]
        conv_i += 1
        if 'crp' == pattern:
          pool_pieces_num += 1
      # affine-relu
      elif 'ar' == pattern:
        # Checks if we come from conv-like layer
        prev_dim = None
        if affine_i == 0:
          # Since we're using padding to not to reduce input data dimensions
          prev_dim = num_filters[-1] * H * W * 0.25**pool_pieces_num
        else:
          prev_dim = hidden_dims[affine_i - 1]
        self.__init_affine_piece(stri, prev_dim, hidden_dims[affine_i])
        affine_i += 1
      else:
        raise ValueError('Incorrect piece pattern: %s.' % pattern)
    self.__init_final_layer(hidden_dims[-1], num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    scores, caches = self.__forward_all(X)

    if y is None:
      return scores

    loss, dscores = softmax_loss(scores, y)
    grads = {}

    reg_loss = self.__backward_all(dscores, caches, grads)
    loss += reg_loss

    return loss, grads


  def __init_conv_piece(self, stri, F, C, HH):
    WW = HH
    self.params['W' + stri] = self.weight_scale * np.random.randn(F, C, HH, WW)
    self.params['b' + stri] = np.zeros(F)
    # Assuming using batchnorm by default
    self.params['spatial_gamma' + stri] = np.ones(F)
    self.params['spatial_beta' + stri] = np.zeros(F)


  def __init_affine_piece(self, stri, prev_dim, current_dim):
    self.params['W' + stri] = self.weight_scale * np.random.randn(prev_dim,
        current_dim)
    self.params['b' + stri] = np.zeros(current_dim)
    # Assuming using batchnorm by default
    self.params['gamma' + stri] = np.ones(current_dim)
    self.params['beta' + stri] = np.zeros(current_dim)


  def __init_final_layer(self, prev_dim, num_classes):
    stri = str(self.conv_pieces_num + self.affine_pieces_num + 1)
    self.params['W' + stri] = self.weight_scale * np.random.randn(prev_dim,
        num_classes)
    self.params['b' + stri] = np.zeros(num_classes)


  def __forward_cr(self, stri, inp, bn_param, conv_param):
    return conv_bn_relu_forward(inp,
        self.params['W' + stri],
        self.params['b' + stri],
        conv_param,
        self.params['spatial_gamma' + stri],
        self.params['spatial_beta' + stri],
        bn_param)


  def __forward_crp(self, stri, inp, bn_param, conv_param, pool_param):
    return conv_bn_relu_pool_forward(inp,
        self.params['W' + stri],
        self.params['b' + stri],
        conv_param, pool_param,
        self.params['spatial_gamma' + stri],
        self.params['spatial_beta' + stri],
        bn_param)


  def __forward_ar(self, stri, inp, bn_param):
    return affine_bn_relu_forward(inp,
        self.params['W' + stri],
        self.params['b' + stri],
        self.params['gamma' + stri],
        self.params['beta' + stri],
        bn_param)


  def __forward_final(self, inp):
    strn = str(self.num_layers)
    return affine_forward(inp,
        self.params['W' + strn],
        self.params['b' + strn])


  def __forward_all(self, X):
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    inp = X
    caches = []

    conv_i = 0
    for i in xrange(self.num_layers - 1):
      stri = str(i + 1)
      pattern = self.pieces_patterns[i]
      bn_param = self.bn_params[i]
      if 'cr' == pattern:
        conv_param = {'stride': self.strides[conv_i], 'pad': self.pads[conv_i]}
        inp, cache = self.__forward_cr(stri, inp, bn_param, conv_param)
        conv_i += 1
      elif 'crp' == pattern:
        conv_param = {'stride': self.strides[conv_i], 'pad': self.pads[conv_i]}
        inp, cache = self.__forward_crp(stri, inp, bn_param, conv_param,
            pool_param)
        conv_i += 1
      elif 'ar' == pattern:
        inp, cache = self.__forward_ar(stri, inp, bn_param)
      else:
        raise ValueError('Incorrect piece pattern: %s.' % pattern)
      caches.append(cache)
    # Final layer pass
    scores, cache = self.__forward_final(inp)
    caches.append(cache)
    return scores, caches


  def __backward_cr(self, stri, dres, cache, grads):
    # Backward for single conv-relu separately
    dx, dw, db, dgamma, dbeta = conv_bn_relu_backward(dres, cache)
    grads['W' + stri] = dw
    # Regularize
    grads['W' + stri] += self.reg * self.params['W' + stri]

    grads['b' + stri] = db
    grads['spatial_gamma' + stri] = dgamma
    grads['spatial_beta' + stri] = dbeta
    return dx


  def __backward_crp(self, stri, dres, cache, grads):
    # Backward for single conv-relu separately
    dx, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(dres, cache)
    grads['W' + stri] = dw
    # Regularize
    grads['W' + stri] += self.reg * self.params['W' + stri]

    grads['b' + stri] = db
    grads['spatial_gamma' + stri] = dgamma
    grads['spatial_beta' + stri] = dbeta
    return dx


  def __backward_ar(self, stri, dres, cache, grads):
    dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dres, cache)
    grads['W' + stri] = dw
    # Regularize
    grads['W' + stri] += self.reg * self.params['W' + stri]

    grads['b' + stri] = db
    grads['gamma' + stri] = dgamma
    grads['beta' + stri] = dbeta
    return dx


  def __backward_all(self, dscores, caches, grads):
    reg_loss = 0
    # Compute grads for out layer separately
    strn = str(self.num_layers)
    dres, grads['W' + strn], grads['b' + strn] = affine_backward(
            dscores, caches[self.num_layers - 1])
    for i in xrange(self.num_layers - 2, -1, -1):
      stri = str(i + 1)
      cache = caches[i]
      pattern = self.pieces_patterns[i]
      if 'cr' == pattern:
        dres = self.__backward_cr(stri, dres, cache, grads)
      elif 'crp' == pattern:
        dres = self.__backward_crp(stri, dres, cache, grads)
      elif 'ar' == pattern:
        dres = self.__backward_ar(stri, dres, cache, grads)
      else:
        raise ValueError('Incorrect piece pattern: %s.' % pattern)
      reg_loss += .5 * self.reg * np.sum(np.square(self.params['W' + stri]))
    return reg_loss


pass
