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
  
# [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
class GeneralConvNet(object):
  def __init__(self, input_dim=(3, 32, 32), conv_pieces_num=3,
      affine_pieces_num=2, num_filters=None, filter_sizes=None,
      hidden_dims=None, num_classes=10, weight_scale=1e-3, reg=0.0,
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
    self.conv_pieces_num = conv_pieces_num
    self.affine_pieces_num = affine_pieces_num
    self.num_layers = conv_pieces_num + 1 + affine_pieces_num + 1
    self.reg = reg
    self.dtype = dtype

    self.bn_params = [{'mode': 'train'} for i in xrange(conv_pieces_num +
      affine_pieces_num + 1)]

    if (num_filters == None):
      num_filters = [32] + np.tile(64, conv_pieces_num - 1).tolist()
    if (filter_sizes == None):
      filter_sizes = np.tile(3, conv_pieces_num).tolist()
    if (hidden_dims == None):
      hidden_dims = np.tile(100, affine_pieces_num).tolist()

    self.num_filters = num_filters
    self.filter_sizes = filter_sizes
    self.hidden_dims = hidden_dims

    C, H, W = input_dim
    # [conv-relu-pool] x conv_pieces_num
    for i in xrange(conv_pieces_num):
      HH = filter_sizes[i]
      WW = HH
      F = num_filters[i]
      stri = str(i + 1)
      self.params['W' + stri] = weight_scale * np.random.randn(F, C, HH, WW)
      self.params['b' + stri] = np.zeros(F)
      # Assuming using batchnorm by default
      self.params['spatial_gamma' + stri] = np.ones(F)
      self.params['spatial_beta' + stri] = np.zeros(F)
      C = F

    # conv-relu
    strc = str(conv_pieces_num + 1)
    self.params['W' + strc] = weight_scale * np.random.randn(num_filters[-1],
        C, filter_sizes[-1], filter_sizes[-1])
    self.params['b' + strc] = np.zeros(num_filters[-1])
    self.params['spatial_gamma' + strc] = np.ones(num_filters[-1])
    self.params['spatial_beta' + strc] = np.zeros(num_filters[-1])

    # [affine] x affine_pieces_num
    prev_dim = num_filters[-1] * H * W * 0.25**conv_pieces_num
    for i in xrange(affine_pieces_num):
      stri = str(conv_pieces_num + i + 2)
      current_dim = hidden_dims[i]
      self.params['W' + stri] = weight_scale * np.random.randn(prev_dim,
          current_dim)
      self.params['b' + stri] = np.zeros(current_dim)
      # Assuming using batchnorm by default
      self.params['gamma' + stri] = np.ones(current_dim)
      self.params['beta' + stri] = np.zeros(current_dim)
      prev_dim = current_dim
    # Output layer
    strn = str(conv_pieces_num + affine_pieces_num + 2)
    self.params['W' + strn] = weight_scale * np.random.randn(current_dim,
        num_classes)
    self.params['b' + strn] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    inp = X
    caches = []
    for i in xrange(self.conv_pieces_num):
      stri = str(i + 1)
      # print "Forward conv-relu-pool #%02d" % (i + 1) 
      # print self.params['W' + stri].shape
      conv_param = {'stride': 1, 'pad': (self.filter_sizes[i] - 1) / 2}
      inp, cache = conv_bn_relu_pool_forward(inp,
          self.params['W' + stri],
          self.params['b' + stri],
          conv_param, pool_param,
          self.params['spatial_gamma' + stri],
          self.params['spatial_beta' + stri],
          self.bn_params[i])
      # print inp.shape
      caches.append(cache)

    # print "Forward conv-relu #%02d" % (self.conv_pieces_num + 1) 
    strc = str(self.conv_pieces_num + 1)
    # print self.params['W' + strc].shape
    inp, cache = conv_bn_relu_forward(inp,
          self.params['W' + strc],
          self.params['b' + strc],
          conv_param,
          self.params['spatial_gamma' + strc],
          self.params['spatial_beta' + strc],
          self.bn_params[self.conv_pieces_num])
    # print inp.shape
    caches.append(cache)

    for i in xrange(self.affine_pieces_num):
      stri = str(self.conv_pieces_num + i + 2)
      # print "Forward affine-relu #%02d" % (self.conv_pieces_num + i + 2) 
      # print self.params['W' + stri].shape
      inp, cache = affine_bn_relu_forward(inp,
          self.params['W' + stri],
          self.params['b' + stri],
          self.params['gamma' + stri],
          self.params['beta' + stri],
          self.bn_params[self.conv_pieces_num + i + 1])
      # print inp.shape
      caches.append(cache)
    strn = str(self.conv_pieces_num + self.affine_pieces_num + 2)
    scores, cache = affine_forward(inp,
          self.params['W' + strn],
          self.params['b' + strn])
    caches.append(cache)

    if y is None:
      return scores
    
    loss, grads = 0, {}

    loss, dscores = softmax_loss(scores, y)
    # loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) +
    #     np.sum(W3 * W3))

    # da1, grads['W3'], grads['b3'] = affine_backward(dscores, scores_cache)
    # grads['W3'] += self.reg * W3
    # dc1, grads['W2'], grads['b2'] = affine_relu_backward(da1, a2_cache)
    # grads['W2'] += self.reg * W2
    # _, grads['W1'], grads['b1'] = conv_relu_pool_backward(dc1, c1_cache)
    # grads['W1'] += self.reg * W1

    dres = dscores
    # Compute grads for out layer separately
    dres, grads['W' + strn], grads['b' + strn] = affine_backward(
            dscores, caches[self.num_layers - 1])
    for i in xrange(self.affine_pieces_num - 1, -1, -1):
      stri = str(self.conv_pieces_num + i + 2)
      # print "Backward affine-relu #%02d" % (self.conv_pieces_num + i + 2)
      cache = caches[self.conv_pieces_num + i + 1]
      dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dres, cache)
      grads['W' + stri] = dw
      # Regularize
      Wi = self.params['W' + stri]
      loss += .5 * self.reg * np.sum(Wi * Wi)
      grads['W' + stri] += self.reg * Wi

      grads['b' + stri] = db
      grads['gamma' + stri] = dgamma
      grads['beta' + stri] = dbeta
      # Setting base derivative for next iteration
      dres = dx

    # Backward for single conv-relu separately
    strc = str(self.conv_pieces_num + 1) 
    # print "Backward conv-relu #%02d" % (self.conv_pieces_num + 1) 
    dx, dw, db, dgamma, dbeta = conv_bn_relu_backward(dres,
        caches[self.conv_pieces_num])
    grads['W' + strc] = dw
    # Regularize
    Wi = self.params['W' + strc]
    loss += .5 * self.reg * np.sum(Wi * Wi)
    grads['W' + strc] += self.reg * Wi

    grads['b' + strc] = db
    grads['spatial_gamma' + strc] = dgamma
    grads['spatial_beta' + strc] = dbeta
    # Setting base derivative for next iteration
    dres = dx

    for i in xrange(self.conv_pieces_num - 1, -1, -1):
      stri = str(i + 1)
      # print "Backward conv-relu-pool #%02d" % (i + 1) 
      cache = caches[i]
      dx, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(dres, cache)
      grads['W' + stri] = dw
      # Regularize
      Wi = self.params['W' + stri]
      loss += .5 * self.reg * np.sum(Wi * Wi)
      grads['W' + stri] += self.reg * Wi

      grads['b' + stri] = db
      grads['spatial_gamma' + stri] = dgamma
      grads['spatial_beta' + stri] = dbeta
      # Setting base derivative for next iteration
      dres = dx
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

class ConvNet(object):
  def __init__(self, input_dim=(3, 32, 32), pieces_patterns=None,
      num_filters=None, filter_sizes=None, hidden_dims=None, num_classes=10,
      weight_scale=1e-3, reg=0.0, dtype=np.float32):
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
    self.conv_pieces_num = len(num_filters) + 1
    self.affine_pieces_num = len(hidden_dims)
    self.num_layers = self.conv_pieces_num + self.affine_pieces_num + 1

    self.weight_scale = weight_scale
    self.reg = reg
    self.dtype = dtype

    self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers)]

    C, H, W = input_dim
    conv_i = 0
    affine_i = 0
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
      # affine-relu
      elif 'ar' == pattern:
        # Checks if we come from conv-like layer
        prev_dim = None
        if affine_i == 0:
          # Since we're using padding to not to reduce input data dimensions
          prev_dim = num_filters[-1] * H * W * 0.25**self.conv_pieces_num
        else:
          prev_dim = hidden_dims[affine_i - 1]
        self.__init_affine_piece(stri, prev_dim, hidden_dims[affine_i])
        affine_i += 1
      else:
        raise ValueError('Incorrect piece pattern: %s.' % pattern) 
    self.__init_final_layer(hidden_dims[-1], num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

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

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    inp = X
    caches = []
    for i in xrange(self.conv_pieces_num):
      stri = str(i + 1)
      # print "Forward conv-relu-pool #%02d" % (i + 1) 
      # print self.params['W' + stri].shape
      conv_param = {'stride': 1, 'pad': (self.filter_sizes[i] - 1) / 2}
      inp, cache = conv_bn_relu_pool_forward(inp,
          self.params['W' + stri],
          self.params['b' + stri],
          conv_param, pool_param,
          self.params['spatial_gamma' + stri],
          self.params['spatial_beta' + stri],
          self.bn_params[i])
      # print inp.shape
      caches.append(cache)

    # print "Forward conv-relu #%02d" % (self.conv_pieces_num + 1) 
    strc = str(self.conv_pieces_num + 1)
    # print self.params['W' + strc].shape
    inp, cache = conv_bn_relu_forward(inp,
          self.params['W' + strc],
          self.params['b' + strc],
          conv_param,
          self.params['spatial_gamma' + strc],
          self.params['spatial_beta' + strc],
          self.bn_params[self.conv_pieces_num])
    # print inp.shape
    caches.append(cache)

    for i in xrange(self.affine_pieces_num):
      stri = str(self.conv_pieces_num + i + 2)
      # print "Forward affine-relu #%02d" % (self.conv_pieces_num + i + 2) 
      # print self.params['W' + stri].shape
      inp, cache = affine_bn_relu_forward(inp,
          self.params['W' + stri],
          self.params['b' + stri],
          self.params['gamma' + stri],
          self.params['beta' + stri],
          self.bn_params[self.conv_pieces_num + i + 1])
      # print inp.shape
      caches.append(cache)
    strn = str(self.conv_pieces_num + self.affine_pieces_num + 2)
    scores, cache = affine_forward(inp,
          self.params['W' + strn],
          self.params['b' + strn])
    caches.append(cache)

    if y is None:
      return scores
    
    loss, grads = 0, {}

    loss, dscores = softmax_loss(scores, y)
    dres = dscores
    # Compute grads for out layer separately
    dres, grads['W' + strn], grads['b' + strn] = affine_backward(
            dscores, caches[self.num_layers - 1])
    for i in xrange(self.affine_pieces_num - 1, -1, -1):
      stri = str(self.conv_pieces_num + i + 2)
      # print "Backward affine-relu #%02d" % (self.conv_pieces_num + i + 2)
      cache = caches[self.conv_pieces_num + i + 1]
      dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dres, cache)
      grads['W' + stri] = dw
      # Regularize
      Wi = self.params['W' + stri]
      loss += .5 * self.reg * np.sum(Wi * Wi)
      grads['W' + stri] += self.reg * Wi

      grads['b' + stri] = db
      grads['gamma' + stri] = dgamma
      grads['beta' + stri] = dbeta
      # Setting base derivative for next iteration
      dres = dx

    # Backward for single conv-relu separately
    strc = str(self.conv_pieces_num + 1) 
    # print "Backward conv-relu #%02d" % (self.conv_pieces_num + 1) 
    dx, dw, db, dgamma, dbeta = conv_bn_relu_backward(dres,
        caches[self.conv_pieces_num])
    grads['W' + strc] = dw
    # Regularize
    Wi = self.params['W' + strc]
    loss += .5 * self.reg * np.sum(Wi * Wi)
    grads['W' + strc] += self.reg * Wi

    grads['b' + strc] = db
    grads['spatial_gamma' + strc] = dgamma
    grads['spatial_beta' + strc] = dbeta
    # Setting base derivative for next iteration
    dres = dx

    for i in xrange(self.conv_pieces_num - 1, -1, -1):
      stri = str(i + 1)
      # print "Backward conv-relu-pool #%02d" % (i + 1) 
      cache = caches[i]
      dx, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(dres, cache)
      grads['W' + stri] = dw
      # Regularize
      Wi = self.params['W' + stri]
      loss += .5 * self.reg * np.sum(Wi * Wi)
      grads['W' + stri] += self.reg * Wi

      grads['b' + stri] = db
      grads['spatial_gamma' + stri] = dgamma
      grads['spatial_beta' + stri] = dbeta
      # Setting base derivative for next iteration
      dres = dx
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads


pass
