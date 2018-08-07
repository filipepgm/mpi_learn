### Optimizers used to update master process weights

import numpy as np
import numexpr as ne
import copy
import pickle
import os
import tensorflow as tf

from ..utils import weights_from_shapes
from ..train.trace import Trace, trace

class Optimizer(object):
    """Base class for optimization algorithms.
        Currently doesn't do anything."""

    def __init__(self):
        pass

    def reset(self):
        pass

    def apply_update(self, weights, gradient):
        raise NotImplementedError

    def save(self, fn = None):
        if fn is None:
            fn = 'master-opt-{}.algo'.format( os.getpid())
        d= open(fn,'wb')
        pickle.dump(self, d)
        d.close()

    def load(self, fn = 'algo_.pkl'):
        d = open(fn, 'rb')
        self = pickle.load( d )
        d.close()


class MultiOptimizer(Optimizer):
    def __init__(self, opt, s):
        self.opts = [copy.deepcopy(opt) for i in range(s)]

    def reset(self):
        for o in self.opts:
            o.reset()

    def apply_update(self, weights, gradient):
        r = []
        for o,w,g in zip(self.opts, weights, gradient):
            r.append( o.apply_update(w,g) )
        return r

class VanillaSGD(Optimizer):
    """Stochastic gradient descent with no extra frills.
          learning_rate: learning rate parameter for SGD"""

    def __init__(self, learning_rate=0.01):
        super(VanillaSGD, self).__init__()
        self.learning_rate = learning_rate

    @trace
    def apply_update(self, weights, gradient):
        """Move weights in the direction of the gradient, by the amount of the
            learning rate."""
        new_weights = []
        for w, g in zip(weights, gradient):
            if type(w) == list:
                new_weights.append( [] )
                for ww, gg in zip(w,g):
                    new_weights[-1].append( np.subtract( ww, self.learning_rate*gg) )
            else:
                new_weights.append(np.subtract(w, self.learning_rate*g))
        return new_weights


class RunningAverageOptimizer(Optimizer):
    """Base class for AdaDelta, Adam, and RMSProp optimizers.
        rho (tunable parameter): decay constant used to compute running parameter averages
        epsilon (tunable parameter): small constant used to prevent division by zero
        running_g2: running average of the squared gradient, where squaring is done componentwise"""

    def __init__(self, rho=0.95, epsilon=1e-8):
        super(RunningAverageOptimizer, self).__init__()
        self.init_rho = rho
        self.init_epsilon = epsilon
        RunningAverageOptimizer.reset(self)

    def reset(self):
        super(RunningAverageOptimizer, self).reset()
        self.epsilon = self.init_epsilon
        self.rho = self.init_rho
        self.running_g2 = None

    @trace
    def running_average_square_np(self, previous, update):
        """Computes and returns the running average of the square of a numpy array.
            previous (numpy array): value of the running average in the previous step
            update (numpy array): amount of the update"""
#        #Trace.begin("rasn_1")
#        square = np.square(update)
#        #Trace.end("rasn_1")
#        #Trace.begin("rasn_2")
#        new_contribution = (1-self.rho) * square
#        #Trace.end("rasn_2")
#        #Trace.begin("rasn_3")
#        old_contribution = self.rho * previous
#        #Trace.end("rasn_3")
#        return new_contribution + old_contribution
        
        #rho = previous.dtype.type(self.rho)
        #return ne.evaluate("(1-rho) * update * update + rho * previous")

        rho = tf.constant(self.rho, dtype=tf.float32)
        rho_sym = tf.constant(1-self.rho, dtype=tf.float32)

        square = tf.square(update)
        new_contribution = tf.scalar_mul(rho_sym, square)
        old_contribution = tf.scalar_mul(rho, previous)

        return new_contribution + old_contribution

        #print (previous.shape)
#
        #matrix = np.stack((previous, np.square(update)))
        #result =  np.average(matrix, axis = 0, weights = [self.rho, 1-self.rho]).astype(np.float32)
#
        ##print (result.shape)
        #return result


    @trace
    def running_average_square(self, previous, update):
        """Returns the running average of the square of a quantity.
            previous (list of numpy arrays): value of the running average in the previous step
            update (list of numpy arrays): amount of the update"""
        if previous == 0:
            previous = [ np.zeros_like(u) for u in update ]
        result = []
        for prev, up in zip(previous, update):
            result.append( self.running_average_square_np( prev, up ) )
        return result

    def sqrt_plus_epsilon(self, value):
        """Computes running RMS from the running average of squares.
            value: numpy array containing the running average of squares"""
        return np.sqrt( np.add(value, self.epsilon) )

class Adam(RunningAverageOptimizer):
    """Adam optimizer.
        Note that the beta_2 parameter is stored internally as 'rho'
        and "v" in the algorithm is called "running_g2"
        for consistency with the other running-average optimizers
        Attributes:
          learning_rate: base learning rate
          beta_1: decay rate for the first moment estimate
          m: running average of the first moment of the gradient
          t: time step
        """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
            epsilon=1e-8):
        super(Adam, self).__init__(rho=beta_2, epsilon=epsilon)
        self.sess = tf.Session()
        self.init_learning_rate = learning_rate
        self.init_beta_1 = beta_1
        self.reset()

    def reset(self):
        super(Adam, self).reset()
        self.beta_1 = self.init_beta_1
        self.learning_rate = self.init_learning_rate
        self.t = 0
        self.sess.run([self.running_g2.initializer,self.m.initializer])

    def setup_running_average_square_np(self, previous, update, rho, rho_sym):
        """Computes and returns the running average of the square of a numpy array.
            previous (numpy array): value of the running average in the previous step
            update (numpy array): amount of the update"""
        square = tf.square(update)
        new_contribution = tf.scalar_mul(rho_sym, square)
        old_contribution = tf.scalar_mul(rho, previous)

        return new_contribution + old_contribution

    def setup_update_graph(self, weights_shape):
        self.weights = tf.placeholder(tf.float32, shape=weights_shape)
        self.gradient = tf.placeholder(tf.float32, shape=weights_shape)

        self.running_g2 = [ tf.Variable(np.zeros_like(w), dtype=tf.float32) for w in self.weights ]
        self.m = [ tf.Variable(np.zeros_like(w), dtype=tf.float32) for w in self.weights ]

        beta_1 = tf.constant(self.beta_1, dtype=tf.float32)
        beta_1_sym = tf.constant(1-self.beta_1, dtype=tf.float32)

        updated_m = [
            tf.scalar_mul(beta_1_sym, update) + tf.scalar_mul(beta_1, previous)
            for previous, update in zip(self.m, self.gradient)
        ]

        self.update_op_m = [
            var.assign(updated)  for var, updated in zip(self.m, updated_m)
        ]

        #######################################

        rho = tf.constant(self.rho, dtype=tf.float32)
        rho_sym = tf.constant(1-self.rho, dtype=tf.float32)

        updated_running_g2 = [
            self.setup_running_average_square_np(old, new, rho, rho_sym)
            for old, new in zip(self.running_g2, self.gradient)
        ]

        self.update_op_g2 = [
            var.assign(updated)  for var, updated in zip(self.running_g2, updated_running_g2)
        ]
        #######################################

        self.t_ph = tf.placeholder(tf.float32, shape=())

        alpha_t = self.learning_rate * (1 - self.rho**self.t_ph)**(0.5) / (1 - self.beta_1**self.t_ph)

        self.new_weights = [
            w - alpha_t * g / ( tf.square(g2) + self.epsilon )
            for w, g, g2 in zip(self.weights, updated_m, updated_running_g2)
        ]

    @trace
    def apply_update(self, weights, gradient):
        #update vars
        gradient_dict = {placeholder: value for placeholder, value in zip(self.gradient,gradient)}
        self.sess.run(self.update_op_g2+self.update_op_m, feed_dict=gradient_dict)

        weights_dict = {placeholder: value for placeholder, value in zip(self.weights,weights)}
        weights_dict[self.t_ph] = self.t
        self.t+=1
        return self.sess.run(self.new_weights, feed_dict=weights_dict)


class AdaDelta(RunningAverageOptimizer):
    """ADADELTA adaptive learning rate method.
        running_dx2: running average of squared parameter updates
        """

    def __init__(self, rho=0.95, epsilon=1e-8):
        super(AdaDelta, self).__init__(rho, epsilon)
        AdaDelta.reset(self)

    def reset(self):
        super(AdaDelta, self).reset()
        self.running_dx2 = None

    @trace
    def apply_update(self, weights, gradient):
        """Update the running averages of gradients and weight updates,
            and compute the Adadelta update for this step."""
        if self.running_g2 is None:
            self.running_g2 = [ np.zeros_like(g) for g in gradient ]
        if self.running_dx2 is None:
            self.running_dx2 = [ np.zeros_like(g) for g in gradient ]

        self.running_g2 = self.running_average_square( self.running_g2, gradient )
        new_weights = []
        updates = []
        for w, g, g2, dx2 in zip(weights, gradient, self.running_g2, self.running_dx2):
            update = np.multiply( np.divide( self.sqrt_plus_epsilon(dx2), self.sqrt_plus_epsilon(g2) ), g )
            new_weights.append( np.subtract( w, update ) )
            updates.append(update)
        self.running_dx2 = self.running_average_square( self.running_dx2, updates )
        return new_weights

class RMSProp(RunningAverageOptimizer):
    """RMSProp adaptive learning rate method.
        learning_rate: base learning rate, kept constant
        """

    def __init__(self, rho=0.9, epsilon=1e-8, learning_rate=0.001):
        super(RMSProp, self).__init__(rho, epsilon)
        self.init_learning_rate = learning_rate
    def reset(self):
        super(RMSProp, self).reset()
        self.learning_rate = self.init_learning_rate

    def apply_update(self, weights, gradient):
        """Update the running averages of gradients,
            and compute the update for this step."""
        if self.running_g2 is None:
            self.running_g2 = [ np.zeros_like(g) for g in gradient ]

        self.running_g2 = self.running_average_square( self.running_g2, gradient )
        new_weights = []
        for w, g, g2 in zip(weights, gradient, self.running_g2):
            update = np.multiply( np.divide( self.learning_rate, self.sqrt_plus_epsilon(g2) ), g )
            new_weights.append( np.subtract( w, update ) )
        return new_weights

def get_optimizer(name):
    """Get optimizer class by string identifier"""
    lookup = {
            'sgd':      VanillaSGD,
            'adadelta': AdaDelta,
            'rmsprop':  RMSProp,
            'adam':     Adam,
            }
    return lookup[name]

class OptimizerBuilder(object):
    """Builds a new Keras or Torch optimizer and optionally wraps it in horovod DistributedOptimizer."""

    def __init__(self, name, config=None, horovod_wrapper=False):
        self.name = name
        self.config = config
        self.horovod_wrapper = horovod_wrapper

    def build(self):
        from keras.optimizers import deserialize
        if self.config is None:
            self.config = {}
        opt_config = {'class_name': self.name, 'config': self.config}
        opt = deserialize(opt_config)
        if self.horovod_wrapper:
            import horovod.keras as hvd
            if hasattr(opt, 'lr'):
                opt.lr *= hvd.size()
            opt = hvd.DistributedOptimizer(opt)
        return opt

    def build_torch(self, model):
        import torch
        opt = torch.optim.SGD(model.parameters(), 1.)
        if self.horovod_wrapper:
            import horovod.torch as hvd
            opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters())
        return opt
