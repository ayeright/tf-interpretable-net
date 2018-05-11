# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:42:44 2018

@author: scott
"""

import tensorflow as tf
import numpy as np

class BinaryClassificationNet(object):
    
    def __init__(self, 
                 layers, 
                 initialiser=tf.keras.initializers.glorot_uniform(), 
                 l2_reg=0.0, 
                 optimiser=tf.train.AdamOptimizer(),
                 lambda_u=1.0,
                 checkpoint_path='/tmp/my_model_final.ckpt'):
        '''
        Class initialiser
        
        Inputs:
            layers - (dict) Dictionary of dictionaries specifying properties of each hidden layer
            e.g. {1: {'dim': 20, 'activation': tf.nn.relu, 'dropout_rate': 0.5, 'batch_norm': True},
                  2: {'dim': 20, 'activation': tf.nn.relu, 'dropout_rate': 0.5, 'batch_norm': True}}
            initialiser - (tf op) tensorflow initialiser
                          default=tf.keras.initializers.glorot_uniform()
            l2_reg - (float >= 0) l2 regularisation parameter
                     default=0.0
            optimiser - (tf op) tensorflow optimiser
                        default=tf.train.AdamOptimizer()
            lambda_u - (float >= 0) weight of unlabelled examples in cost function 
                       default=1.0
            checkpoint_path - (string) path for saving model checkpoint
                              default='/tmp/my_model_final.ckpt'
        '''
        
        # add linear output layer with a single neuron to layer list
        layers.append({'dim': 1})
        
        # convert to dictionary
        layer_dict = {}
        for i, layer in enumerate(layers):
            layer_dict[i] = layer
        
        self.layers = layer_dict
        self.initialiser = initialiser
        self.l2_reg = l2_reg
        self.optimiser = optimiser
        self.lambda_u = lambda_u 
        self.checkpoint_path = checkpoint_path
        self.weights = {}
        self.biases = {}
        self.tf_ops = {}
        self.first_fit = True
        
    
    def fit(self, 
            X_train, 
            y_train, 
            X_val=None,
            y_val=None,
            batch_size=32, 
            num_epochs=100, 
            patience=10,
            verbose=True):
        '''
        Trains the network
        
        Inputs:
            X_train - (np array) labelled training features
            y_train - (np vector) training labels (0/1)
            X_val - (np array) validation features
                      default=None
            y_val - (np vector) validation labels (0/1) 
                      default=None
            batch_size - (int > 0) training batch size
                         default=32
            num_epochs - (int > 0) number of training epochs
                         default=100
            patience - (int > 0) if validation data provided, 
                       number of epochs without improvement to wait before early stopping
                       default=10
            verbose - (bool) whether or not to print updates after every epoch
                      default=True
        '''
        
        # get data dimensions
        num_x, input_dim = X_train.shape
        
        # build computational graph
        self.build_graph(input_dim)
                    
        # train network               
        with tf.Session() as sess:
            
            if self.first_fit:                
                # initialise variables
                sess.run(self.tf_ops['init'])
                self.first_fit = False                                                
            else:                
                # restore variables
                self.tf_ops['saver'].restore(sess, self.checkpoint_path)
            
            if (X_val is not None) & (y_val is not None):
                # compute initial validation loss
                best_loss = self.tf_ops['data_loss'].eval(feed_dict={self.tf_ops['X']: X_val,
                                                                    self.tf_ops['y']: y_val})
            
            # train for num_epochs
            num_batches = int(np.ceil(float(num_x) / batch_size))
            for epoch in np.arange(1, num_epochs + 1):

                if verbose:
                    print('+-----------------------------------------------------------+')
                    print('Running epoch', epoch, 'of', num_epochs)

                # shuffle examples
                if epoch == 1:
                    shuffle = np.random.choice(num_x, num_x, replace=False)
                else:
                    shuffle = shuffle[np.random.choice(num_x, num_x, replace=False)]
                X_train = X_train[shuffle]
                y_train = y_train[shuffle]
                
                # train in batches
                for batch in np.arange(num_batches):

                    # get labelled data in this batch
                    i_first = batch * batch_size
                    i_last = (1 + batch) * batch_size
                    i_last = min(num_x, i_last)
                    X_batch = X_train[i_first:i_last]
                    y_batch = y_train[i_first:i_last]
                    
                    # run training operation
                    sess.run(self.tf_ops['training_op'], 
                             feed_dict={self.tf_ops['X']: X_batch,
                                        self.tf_ops['y']: y_batch,
                                        self.tf_ops['is_training']: True})
                    
                # compute cross entropy of training examples in batches
                _, xentropy_train = self.apply_in_batches(sess, 
                                                          X_train, 
                                                          y_train, 
                                                          batch_size=batch_size)
                
                # compute mean cross entropy
                loss_train = xentropy_train.mean()
                
                if verbose:
                    print('Training loss =', loss_train)
                
                if (X_val is not None) & (y_val is not None): 
                    
                    # compute cross entropy of validation examples in batches
                    _, xentropy_val = self.apply_in_batches(sess, 
                                                            X_val, 
                                                            y_val, 
                                                            batch_size=batch_size)

                    # compute mean cross entropy
                    loss_val = xentropy_val.mean()                  
                
                    if verbose:
                        print('Validation loss =', loss_val)
                    
                    # loss improved?
                    if loss_val < best_loss:                        
                        best_loss -= best_loss - loss_val
                        epochs_since_improvement = 0
                        
                        # save model checkpoint
                        self.tf_ops['saver'].save(sess, self.checkpoint_path)
                        
                    else:                        
                        epochs_since_improvement += 1
                        
                        # early stopping?
                        if epochs_since_improvement >= patience:                               
                            if verbose:
                                print('Early stopping. Best epoch =', epoch - patience)                            
                            break
                            
            if (X_val is None) or (y_val is None):                
                # save final model
                self.tf_ops['saver'].save(sess, self.checkpoint_path)
        
    
    def build_graph(self, input_dim):
        '''
        Builds the tensorflow computational graph
        
        Inputs:
            input_dim - (int > 0) number of input features        
        '''
        
        tf.reset_default_graph()
        
        # create placeholder to specify whether we are training or predicting
        is_training = tf.placeholder_with_default(False, shape=(), name='training')

        # define the network weights and biases
        self.init_weights(input_dim)

        if self.l2_reg > 0:
            # add regularisation loss
            reg_loss = tf.add_n([tf.nn.l2_loss(W) 
                                      for _, W in self.weights.items()]) * self.l2_reg
        else:
            reg_loss = 0.0

        # define placeholders for input data
        X = tf.placeholder(tf.float32, shape=(None, input_dim), name='X')
        y = tf.placeholder(tf.float32, shape=(None), name='y')

        # forward pass
        logits = self.forward_pass(X, is_training)
        p = tf.sigmoid(logits)
        
        # partial derivatives of output probability wrt inputs
        dp_dX = tf.gradients(p, X)[0]

        with tf.name_scope('loss'):

            # labelled loss
            xentropy = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)) 
            data_loss = tf.reduce_mean(xentropy, name='labelled_loss')

            # sum data loss and regularisation loss
            loss = data_loss + reg_loss

        # optimise
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            training_op = self.optimiser.minimize(loss)

        # define initialiser and saver nodes
        init = tf.global_variables_initializer() 
        saver = tf.train.Saver()
        
        # store tf operations we will need to access from other methods
        self.tf_ops['is_training'] = is_training
        self.tf_ops['X'] = X
        self.tf_ops['y'] = y
        self.tf_ops['logits'] = logits
        self.tf_ops['p'] = p
        self.tf_ops['dp_dX'] = dp_dX
        self.tf_ops['xentropy'] = xentropy
        self.tf_ops['data_loss'] = data_loss
        self.tf_ops['loss'] = loss
        self.tf_ops['training_op'] = training_op
        self.tf_ops['init'] = init
        self.tf_ops['saver'] = saver
         
       
    def init_weights(self, input_dim):
        '''
        Defines the network weights and biases
        
        Inputs:
            input_dim - (int > 1) number of input features                      
        '''
         
        # loop through layers
        for i, layer in self.layers.items():
                                       
            # define weights
            output_dim = layer['dim']
            self.weights[i] = tf.Variable(self.initialiser((input_dim, output_dim)), 
                                          name='kernel')  
                                     
            # define biases
            self.biases[i] = tf.Variable(tf.zeros([output_dim]), 
                                         name='bias')
            
            # output dimension becomes new input dimension
            input_dim = int(self.weights[i].get_shape()[1])
                    
            
    def forward_pass(self, X, training=False):
        '''
        Performs forward pass through network to compute logits
        
        Inputs:
            X - (np array) features
            training - (bool) whether or not we are in training mode
                       default=False
        
        Outputs:
            Z - (tensor) logits
        '''
        
        # loop through layers
        for i, layer in self.layers.items():
            
            # get weights and biases for this layer
            W = self.weights[i]
            b = self.biases[i]
        
            # compute linear transformation
            Z = tf.matmul(X, W) + b
            
            if 'batch_norm' in layer:
                if layer['batch_norm'] == True:
                    # perfom batch normalisation
                    Z = tf.layers.batch_normalization(Z, training=training)
                                       
            if 'activation' in layer:
                # compute activations 
                Z = layer['activation'](Z)

            if 'dropout_rate' in layer:
                # dropout
                Z = tf.layers.dropout(Z, rate=layer['dropout_rate'], training=training)
            
            # layer output becomes input for next layer
            X = Z
        
        return tf.squeeze(Z)
    
    
    def apply_in_batches(self, sess, X, y=None, batch_size=32, return_dp_dX=False):
        '''
        Applys the network to the input data in batches.
        If y is provided returns the probabilities of the poitives class
        as well as the cross entropy of each examples.
        If y is not provided returns only the probabilities.
        
        Inputs:
            sess - (tf Session) current tensorflow session
            X - (np array) features
            y - (np vector) class labels (0/1)
                default=None
            batch_size - (int > 0) batch size
                         default = 32
            return_dp_dX - (bool) whether or not to return partial derivatives of probability wrt inputs
                         
        Outputs:
            p - (np vector) probability of positive class for each input
            dp_dX - (np array) partial derivatives of probability wrt inputs
            xentropy - (np vector) cross entropy for each input (only return if y provided)            
        '''
        
        # define variables for storing outputs
        num_x, input_dim = X.shape
        p = np.zeros(num_x)
        if y is not None:        
            xentropy = np.zeros(num_x)
        if return_dp_dX:
            dp_dX = np.zeros((num_x, input_dim))
        
        # loop through batches
        num_batches = int(np.ceil(float(num_x) / batch_size))
        for batch in np.arange(num_batches):

            # get data in this batch
            i_first = batch * batch_size
            i_last = (1 + batch) * batch_size
            i_last = min(num_x, i_last)
            X_batch = X[i_first:i_last]
            
            if y is None:
                
                if return_dp_dX:
                    p[i_first:i_last], dp_dX[i_first:i_last] = sess.run([self.tf_ops['p'], 
                                                                        self.tf_ops['dp_dX']],
                                                                        feed_dict={self.tf_ops['X']: X_batch}) 
                else:
                    p[i_first:i_last] = sess.run(self.tf_ops['p'], feed_dict={self.tf_ops['X']: X_batch}) 
                
               
            else:
                y_batch = y[i_first:i_last]
                
                if return_dp_dX:
                    p[i_first:i_last], dp_dX[i_first:i_last], xentropy[i_first:i_last] = sess.run([self.tf_ops['p'], 
                                                                                                self.tf_ops['dp_dX'],
                                                                                                self.tf_ops['xentropy']],
                                                                                                feed_dict={
                                                                                                self.tf_ops['X']: X_batch,
                                                                                                self.tf_ops['y']: y_batch})
                else:
                    p[i_first:i_last], xentropy[i_first:i_last] = sess.run([self.tf_ops['p'], 
                                                                            self.tf_ops['xentropy']],
                                                                            feed_dict={
                                                                            self.tf_ops['X']: X_batch,
                                                                            self.tf_ops['y']: y_batch})
                    
                
        if y is not None:
            if return_dp_dX:
                return p, dp_dX, xentropy
            else:
                return p, xentropy
        else:
            if return_dp_dX:
                return p, dp_dX
            else:
                return p
                                                                   
    def predict(self, X, batch_size=32):
        '''
        Predicts probability of positive class for each input
        
        Inputs:
            X - (np array) features
            batch_size - (int > 0) batch size for computing probabilities
                         default = 32
            
        Outputs:
            p - (np vector) probability of positive class for each input
        
        '''
        
        # build computational graph
        num_x, input_dim = X.shape
        self.build_graph(input_dim)
        
        with tf.Session() as sess:
        
            # restore variables
            self.tf_ops['saver'].restore(sess, self.checkpoint_path)
            
            # compute probabilities in batches
            p = self.apply_in_batches(sess, X, batch_size=batch_size)
              
        return p 


    def get_dp_dX(self, X, batch_size=32):
        '''
        Gets partial derivatives of probability of positive class wrt each input
        
        Inputs:
            X - (np array) features
            batch_size - (int > 0) batch size for computing gradients
                         default = 32
            
        Outputs:
            dp_dX - (np array) partial derivatives of probability wrt inputs
        
        '''
        
        # build computational graph
        num_x, input_dim = X.shape
        self.build_graph(input_dim)
        
        with tf.Session() as sess:
        
            # restore variables
            self.tf_ops['saver'].restore(sess, self.checkpoint_path)
            
            # compute probabilities in batches
            _, dp_dX = self.apply_in_batches(sess, X, batch_size=batch_size, return_dp_dX=True)
              
        return dp_dX                     