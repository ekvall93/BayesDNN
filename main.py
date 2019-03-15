import numpy as np
import tensorflow as tf
from StochasticLayer import StochasticLayer
from utils import plot

np.random.seed(15)
#Created synthetic sample data
n_samples = 100
X = np.random.normal(size=(n_samples, 1))
y = np.random.normal(np.sin(6. * X) / (np.abs(X) + 1.5), 0.1).ravel() #Ravel make it to a vector
X_pred = np.atleast_2d(np.linspace(-3, 3, num=n_samples)).T #Make sure data is 2D
X = np.hstack((X, X**2, X**3, X**4)) #Featurestacking
X_pred = np.hstack((X_pred, X_pred **2, X_pred **3, X_pred**4))

mc_samples = 800
batch_size = 5

def build_graph(dp_prob=1.0, BN=False, decay=0.99):
    """
    Building training graph

    Parameters
    ----------
    dp_prob : float
        Dropout prob
    BN : bool
        Batchnorm
    Returns
    -------
    array
        graph outputs
    """
    
    n_feats = X.shape[1]
    n_hidden = 100
    model_prob = dp_prob
    model_lam = 1e-2
    BN=BN
    
    X_input = tf.placeholder(tf.float32, [None, n_feats])
    y_input = tf.placeholder(tf.float32, [None])
    
    is_training = tf.placeholder(tf.bool)
    
    Layer_1 = StochasticLayer(n_feats, n_hidden, model_prob, model_lam, is_training ,BN=BN, decay=decay)
    Layer_2 = StochasticLayer(n_hidden, n_hidden, model_prob, model_lam, is_training,BN=BN, decay=decay)
    Layer_3 = StochasticLayer(n_hidden, 1, model_prob, model_lam, is_training, BN=BN, decay=decay)
    
    z_1 = Layer_1(X_input, tf.nn.relu)
    z_2 = Layer_2(z_1, tf.nn.relu)
    y_pred = Layer_3(z_2)
    
    model_sse = tf.reduce_sum(tf.square(y_input - y_pred))
    model_mse = model_sse / n_samples
    model_loss = (
        # Negative log-likelihood.
        model_sse +
        # Regularization.
        Layer_1.regularization +
        Layer_2.regularization +
        Layer_3.regularization
                ) / n_samples
    
    train_step = tf.train.AdamOptimizer(1e-3).minimize(model_loss)
        
    return (X_input, y_input), train_step, model_mse, y_pred, tf.train.Saver(), is_training
    
    

    

def train(iterations=2000, verbose =True, bs=10):
    """
    Train model

    Parameters
    ----------
    iterations : int
        Number of iterations
    verbose : bool
        Verbose
    bs: int
        batch-size
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            #Run it in batches.
            test_idx = np.arange(0 , len(X))
            np.random.shuffle(test_idx)
            X_batch = X[test_idx[:bs]]
            y_batch = y[test_idx[:bs]]
        
        
            train_step.run(feed_dict={X_input: X_batch, y_input: y_batch, is_training:True})
            if verbose:
                if i % 100 == 0:
                    print(i)
                    mse = sess.run(model_mse, {X_input: X_batch, y_input: y_batch, is_training:True})
                    print("Iteration {}, Mean squared errror: {:.4f}".format(i, mse))
        saved_model = saver.save(sess, 'temp/temp-bn-save')
        sess.close()

def test(samples=1000, bs=10,BN=False):
    """
    Test model

    Parameters
    ----------
    samples : int
        Monte Carlo Samples
    bs : int
        Batchnorm samples
    """
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'temp/temp-bn-save')
    

        Y_sample = np.zeros((samples, X_pred.shape[0]))
        for i in range(samples):       
            if BN:
                test_idx = np.arange(0 , len(X_pred))
                np.random.shuffle(test_idx)
                model_pred.eval({X_input: X[test_idx[:bs]],  is_training: True})
            Y_sample[i] = sess.run(model_pred, {X_input: X_pred, is_training: False})
        return Y_sample


if __name__ == "__main__":
   print("Generate point estimate plot")
   tf.reset_default_graph()
   (X_input, y_input), train_step, model_mse, _, saver, is_training = build_graph(BN=False, dp_prob=1.0)
   train(verbose = True)

   tf.reset_default_graph()
   (X_input, y_input), train_step, model_mse, model_pred, saver, is_training = build_graph(BN=False, dp_prob=1.0)
   Y_sample = test(mc_samples)
   plot(Y_sample, X_pred,X,y, mc_samples, path="./assets/point_estimate")


   print("Generate Droput-BDNN")
   tf.reset_default_graph()
   (X_input, y_input), train_step, model_mse, _, saver, is_training = build_graph(BN=False, dp_prob=0.9)
   train(verbose = False, bs=n_samples)

   tf.reset_default_graph()
   (X_input, y_input), train_step, model_mse, model_pred, saver, is_training = build_graph(BN=False, dp_prob=0.9)
   Y_sample_DP = test(mc_samples, bs=n_samples)
   plot(Y_sample_DP, X_pred, X,y,mc_samples,color_line="b-", path="./assets/non_DO_BN")
   print("Generate BN-BDNN")
   tf.reset_default_graph()
   (X_input, y_input), train_step, model_mse, _, saver, is_training = build_graph(BN=True, dp_prob=1.0, decay=0.6)
   train(verbose = False , bs=15)

   tf.reset_default_graph()
   (X_input, y_input), train_step, model_mse, model_pred, saver, is_training = build_graph(BN=True, dp_prob=1.0, decay=0.6)
   Y_sample_BN = test(mc_samples, bs=15,BN=True)
   plot(Y_sample_BN, X_pred,X,y,mc_samples, color_line="g-",path="./assets/BN")

   print("Generate Dropout-BN-BDNN")
   tf.reset_default_graph()
   (X_input, y_input), train_step, model_mse, _, saver, is_training = build_graph(BN=True, dp_prob=0.85, decay=0.5)
   train(verbose = False, bs=20)

   tf.reset_default_graph()
   (X_input, y_input), train_step, model_mse, model_pred, saver, is_training = build_graph(BN=True, dp_prob=0.85, decay=0.5)
   Y_sample = test(mc_samples, bs=20,BN=True)
   plot(Y_sample, X_pred,X,y,mc_samples, color_line="b-",path="./assets/BN_DO")
