import tensorflow as tf
from tensorflow.python.framework import ops


class ZeroGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x):
        grad_name = "ZeroGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _zero_gradients(op, grad):
            return [grad * 0]
        
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
            
        self.num_calls += 1
        return y
    
zero_gradient = ZeroGradientBuilder()
