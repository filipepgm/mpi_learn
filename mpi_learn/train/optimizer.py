### Optimizers used to update master process weights

import numpy as np
import numexpr as ne
import copy
import pickle
import os
import tensorflow as tf
from tensorflow.python.client import timeline

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
        #pickle.dump(self, d) #FIXME re enable
        d.close()

        #tl = timeline.Timeline(self.run_metadata.step_stats)
        #ctf = tl.generate_chrome_trace_format()
        #with open('tf-timeline.json', 'w') as f:
        #    f.write(ctf)

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
        #self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #self.run_metadata = tf.RunMetadata()

        
        self.tf_optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=beta_1,
            beta2=beta_2,
            epsilon=epsilon,
            use_locking=False,
            name='Adam'
        )

        self.init_learning_rate = learning_rate
        self.init_beta_1 = beta_1
        self.reset()

    def reset(self):
        super(Adam, self).reset()
        self.beta_1 = self.init_beta_1
        self.learning_rate = self.init_learning_rate
        self.t = 0
        self.do_reset = True

    def setup_running_average_square_np(self, previous, update):
        """Computes and returns the running average of the square of a numpy array.
            previous (numpy array): value of the running average in the previous step
            update (numpy array): amount of the update"""
        square = tf.square(update)
        new_contribution = tf.scalar_mul(1-self.rho, square)
        old_contribution = tf.scalar_mul(self.rho, previous)

        return new_contribution + old_contribution

    @trace
    def setup_update_graph(self, weights_input):

        print ("AAAA Number of layers: ", len(weights_input))
        self.gradient = [ tf.placeholder(dtype=tf.float32, shape=w.shape, name="gradient") for w in weights_input ]

        self.weights = [ tf.Variable(w, dtype=tf.float32, name="weights") for w in weights_input ]

        var_list = zip(self.gradient, self.weights)

        self.tf_time = tf.Variable(1, dtype=tf.float32, name="time")

        self.adam_op = self.tf_optimizer.apply_gradients(
            grads_and_vars=var_list,
            global_step=self.tf_time,
            name="adam_op"
        )

        #self.running_g2 = [ tf.Variable(np.zeros_like(w), dtype=tf.float32, name="running_g2") for w in weights_input ]
        #self.m = [ tf.Variable(np.zeros_like(w), dtype=tf.float32, name="m") for w in weights_input ]
        #tf_beta1 = tf.constant(self.beta_1, dtype=tf.float32)
        #tf_beta1_inv = tf.constant(1-self.beta_1, dtype=tf.float32)
        #updated_m = [
        #    tf.scalar_mul(tf_beta1_inv, update) + tf.scalar_mul(tf_beta1, previous)
        #    for previous, update in zip(self.m, self.gradient)
        #]
        #self.update_op_m = [
        #    var.assign(updated)  for var, updated in zip(self.m, updated_m)
        #]

        #######################################
        #updated_running_g2 = [
        #    self.setup_running_average_square_np(old, new)
        #    for old, new in zip(self.running_g2, self.gradient)
        #]
#
        #self.update_op_g2 = [
        #    var.assign(updated)  for var, updated in zip(self.running_g2, updated_running_g2)
        #]
        ########################################
#
        ##self.t_ph = tf.placeholder(tf.float32, shape=(), name="time")
        #self.alpga_t_ph = tf.placeholder(tf.float32, shape=(), name="alpha_t")
#
        ##alpha_t = self.learning_rate * (1 - self.rho**self.t_ph)**(0.5) / (1 - self.beta_1**self.t_ph)
        #tf_epsilon = tf.constant(self.epsilon, dtype=tf.float32)
#
        #self.new_weights = [
        #    tf.divide(w - tf.scalar_mul(self.alpga_t_ph, g), ( tf.square(g2) + tf_epsilon ))
        #    for w, g, g2 in zip(self.weights, updated_m, updated_running_g2)
        #]

        #self.apply_weights = [
        #    var.assign(updated)  for var, updated in zip(self.weights, self.new_weights)
        #]

        writer = tf.summary.FileWriter("graph_log", self.sess.graph)

    @trace
    def apply_update(self, weights, gradient):
        if self.do_reset:
            self.setup_update_graph(weights)
            #self.sess.run([v.initializer for v in self.running_g2]+[v.initializer for v in self.m]+[v.initializer for v in self.weights])
            #self.sess.run([v.initializer for v in self.weights] + [self.tf_time.initializer, self.tf_optimizer._beta1_power.initializer, self.tf_optimizer._beta2_power.initializer] )
            self.sess.run(tf.initialize_all_variables())
            self.do_reset = False
        #update vars
        self.t+=1   

        #Trace.begin("feed_dict")
        gradient_dict = {placeholder: value for placeholder, value in zip(self.gradient, gradient)}
        #weights_dict = {placeholder: value for placeholder, value in zip(self.weights,weights)}

        #alpha_t = self.learning_rate * (1 - self.rho**self.t)**(0.5) / (1 - self.beta_1**self.t)
        #overall_dict = {self.alpga_t_ph: alpha_t, **gradient_dict}
        #Trace.end("feed_dict")

        Trace.begin("update_vars")
        #res = self.sess.run(self.update_op_g2+self.update_op_m + self.apply_weights, feed_dict=overall_dict,
        #    options=self.run_options, run_metadata=self.run_metadata)[-len(weights):]
        self.sess.run([self.adam_op], feed_dict=gradient_dict)
        Trace.end("update_vars")
        Trace.begin("get_weights")
        res = self.sess.run(self.weights)
        Trace.begin("get_weights")
        
        #Trace.begin("get new weights")
        #res = self.sess.run(self.new_weights, feed_dict=overall_dict)
        #Trace.end("get new weights")
        return res #TODO FIXME


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
