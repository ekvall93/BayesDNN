import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli

class StochasticLayer:
    """
    StochasticLayer Dense Layer.
    Credit: Viking Penguin


    Parameters
    ----------
    BN : bool
        Batchnormalization
    n_in : int
        Input nodes
    n_out: int
        Output nodes
    model_prob: float
        Dropout probability
    model_lam: float
        regualarization term
    is_training: bool
        traing-phase/test-phase        
    
    Returns
    -------
    array
        Outputlayer
    """
    

    def __init__(self, n_in, n_out, model_prob, model_lam, is_training, BN=False, decay=0.99):
        
        
    
        self.is_training = is_training    
        self.model_prob = model_prob
        self.model_lam = model_lam
        self.model_bern = Bernoulli(probs=self.model_prob, dtype=tf.float32)
        self.model_M = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01))
        self.model_m = tf.Variable(tf.zeros([n_out]))
        self.model_W = tf.matmul(
            tf.diag(self.model_bern.sample((n_in, ))), self.model_M
        )
        #Parameters for BatchNormalization
        self.BN = BN
        self.scale = tf.Variable(tf.ones([n_out]))
        self.beta = tf.Variable(tf.zeros([n_out]))
        self.epsilon = 1e-2
        self.mean = tf.Variable(tf.zeros([n_out]), trainable=False)
        self.var = tf.Variable(tf.ones([n_out]), trainable=False)
        self.decay = decay

    def __call__(self, X, activation=tf.identity):
        output = activation(tf.matmul(X, self.model_W) + self.model_m)
        
        if self.BN:
            output = self.batch_norm_wrapper(output, decay = 0.5)
                
        if self.model_M.shape[1] == 1:
            output = tf.squeeze(output)
        return output

    @property
    def regularization(self):
        """regularization"""

        return self.model_lam * (
            self.model_prob * tf.reduce_sum(tf.square(self.model_M)) +
            tf.reduce_sum(tf.square(self.model_m))
        )

    
    
    def batch_norm_wrapper(self, inputs, decay = 0.99):
        """
        Batchnormalization

        Parameters
        ----------
        inputs : array
            Batch input
        is_training : bool
            training phase/test phase
        decay: float
            Decrease training of popoulation mean and variance.    
        Returns
        -------
        array
            Normalized batched
        """
        
        
        
        def BN_train():
            """
            Batchnormalization training
 
            Returns
            -------
            array
                Normalized train batch
            """
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
            train_mean = tf.assign(self.mean, self.mean * self.decay + batch_mean * (1 - self.decay))
            train_var = tf.assign(self.var, self.var * self.decay + batch_var * (1 - self.decay))
            #train_mean = tf.assign(self.mean, self.mean + batch_mean)
            #train_var = tf.assign(self.var, self.var + batch_var)
            
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, self.beta, self.scale, self.epsilon)
            
        def BN_test():
            """
            Batchnormalization test
 
            Returns
            -------
            array
                Normalized test batched
            """
            return tf.nn.batch_normalization(inputs,
                self.mean, self.var, self.beta, self.scale, self.epsilon)
        
        
        return tf.cond(self.is_training, BN_train, BN_test)

       
        
