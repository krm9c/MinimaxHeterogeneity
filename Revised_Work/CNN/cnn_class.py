# Class for Generalized Convolutional Neural Network
import tensorflow as tf
import numpy as np
import random
import operator

# Function for defining weights
def new_weights(shape, trainable, name):
    initial = tf.Variable(tf.truncated_normal(shape, stddev=0.1), trainable=trainable, name=name)
    return initial

# Function for defining biases
def new_bias(length, trainable, name):
    initial = tf.Variable(tf.constant(0.1, shape=[length]), trainable=trainable, name=name)
    return initial

# The main Class
class learners():
    def __init__(self):
        self.classifier = {}
        self.Deep = {}
        self.Trainer = {}
        self.Evaluation = {}
        self.keys = []
        self.sess = tf.Session()

        # Extra Parameters
        self.Weights =[]

    def new_conv_layer(self,input, num_input_channels, filter_size, num_filters,trainability, key, pool_shape, use_pooling=True):
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        with tf.name_scope(key):
            with tf.name_scope('weights' + key):
                self.classifier['Weight' + key] = new_weights(
                    shape = shape, trainable=trainability, name='Weight'+key)
            with tf.name_scope('bias' + key):
                self.classifier['Bias' + key] = new_bias(
                    length = num_filters, trainable=trainability, name='Bias'+key)
            # tf.nn.conv2d needs a 4D input
            layer = tf.nn.conv2d(input = input, filter= self.classifier['Weight' + key], strides=[1,1,1,1], padding='SAME')
            layer += self.classifier['Bias' + key]
            if use_pooling:
                ksize = [1, pool_shape[0], pool_shape[1], 1]
                layer = tf.nn.max_pool(value = layer, ksize=ksize, strides=[1,2,2,1], padding='SAME')
            # relu activation function converts all negatives to zero
            layer = tf.nn.relu(layer)
            self.Weights.append((self.classifier['Weight'+key], self.classifier['Bias'+key]))
            return layer

    # After all convolutions, we need to flatten the layer
    def flatten_layer(self,layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features

    # Fully connected layer
    def new_fc_layer(self, input, num_inputs, num_outputs, trainability, key, use_relu=True, dropout=False, keep_prob=1.0):
        with tf.name_scope(key):
            with tf.name_scope('weights' + key):
                self.classifier['Weight' + key] = new_weights(
                    shape=[num_inputs, num_outputs], trainable=trainability, name='Weight'+key)
            with tf.name_scope('bias' + key):
                self.classifier['Bias' + key] = new_bias(
                    length= num_outputs, trainable=trainability, name='Bias'+key)
            layer = tf.matmul(input, self.classifier['Weight' + key]) + self.classifier['Bias' + key]
            if use_relu:
                layer = tf.nn.relu(layer)
            if dropout:
                layer = tf.nn.dropout(layer, keep_prob)
            self.Weights.append((self.classifier['Weight'+key], self.classifier['Bias'+key]))
            return layer

    def Custom_Optimizer(self, lr, optimizer):
        if optimizer == 'Adam':
            opt = tf.train.AdamOptimizer
        elif optimizer == 'Adadelta':
            opt = tf.train.AdadeltaOptimizer
        elif optimizer == 'Adagrad':
            opt = tf.train.AdagradOptimizer
        elif optimizer == 'GradientDescent':
            opt = tf.train.GradientDescentOptimizer
        elif optimizer == 'RMSProp':
            opt = tf.train.RMSPropOptimizer
        #print opt
        train_list = []
        for i in xrange(len(self.Weights)):
            weight, bias = self.Weights[i]
            ## Gradient Descent update
            weight_update = tf.gradients(self.classifier['Overall cost'], weight)[0]   + 0.0001*weight
            bias_update   = tf.gradients(self.classifier['Overall cost'],   bias)[0]   + 0.0001*bias

            # Generate the updated variables
            train_list.append((weight_update, weight))
            train_list.append((bias_update, bias))

        return opt(lr).apply_gradients(train_list)


    def Def_Network(self, classes, num_conv_layers, num_fc_layers, img_size, num_channels, \
        filter_size, num_of_filters, pool_shape, fc_size):
        with tf.name_scope("Trainer_Network"):
            #inp_fc_layer = []
            self.Deep['Conv_Layer_0'] = tf.reshape(self.Deep['Conv_Layer_00'], [-1, img_size[0], img_size[1], num_channels])
            for i in range(1, num_conv_layers+1):
                key = 'Conv_Layer_'+ str(i)
                self.Deep['Conv_Layer_'+ str(i)] = self.new_conv_layer(input = self.Deep['Conv_Layer_'+str(i-1)], num_input_channels= num_of_filters[i-1],\
                    filter_size = filter_size[i-1], num_filters = num_of_filters[i], trainability=False, \
                    key=key, pool_shape=pool_shape, use_pooling=True)
                #print key
            
            i = i +1
            key = 'Conv_Layer_'+ str(i)
            self.Deep['Conv_Layer_'+ str(i)], num_features = self.flatten_layer(self.Deep['Conv_Layer_'+ str(i-1)])
            #print key

            i = i +1

            ''' inp_fc_layer[0] = num_features
            for k in range(1, len(num_fc_layers)):
                inp_fc_layer[k] = fc_size
            inp_fc_layer.append(classes)

            temp = i
            for i in range(temp, (len(num_fc_layers)+temp)):
                key = 'Conv_Layer_'+ str(i)
                self.Deep['Conv_Layer_'+ str(i)] = model.new_fc_layer(self.Deep['Conv_Layer_'+ str(i-1)], inp_fc_layer.append[i-temp], inp_fc_layer.append[i-temp+1], trainability=False, \
                key=key, use_relu=True, dropout=True, keep_prob=0.5) '''
            
            key = 'Conv_Layer_'+ str(i)
            self.Deep['Conv_Layer_'+str(i)] = self.new_fc_layer(self.Deep['Conv_Layer_'+str(i-1)], num_features, fc_size, trainability=False, \
                key=key,use_relu=True, dropout=True, keep_prob=0.5)
            #print key
            i = i+1
            key = 'Conv_Layer_'+ str(i)
            self.Deep['Conv_Layer_'+str(i)] = self.new_fc_layer(self.Deep['Conv_Layer_'+str(i-1)], fc_size, classes, trainability=False, \
                key=key, use_relu=False, dropout=False)
            #print key
                
        with tf.name_scope("Classifier"):
            self.classifier['class'] = self.Deep['Conv_Layer_'+str(i)]

        
    def init_NN_custom(self, classes, num_conv_layers, num_fc_layers, img_size_flat, img_size, num_channels, \
        filter_size, num_of_filters, pool_shape, fc_size, lr, optimizer="Adam"):
            with tf.name_scope("PlaceHolders"):  
                # Setup the placeholders
                self.classifier['Target'] = tf.placeholder(
                    tf.float32, shape=[None, classes])
                self.classifier["learning_rate"] = tf.placeholder(
                    tf.float32, [], name='learning_rate')
                self.Deep['Conv_Layer_00'] = tf.placeholder(
                    tf.float32, shape=[None, img_size_flat])

                # The main network
                self.Def_Network(classes, num_conv_layers, num_fc_layers, img_size, num_channels, \
                filter_size, num_of_filters, pool_shape, fc_size)

                with tf.name_scope("Trainer"):
                    # The overall cost function
                    self.classifier["Overall cost"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.classifier['class'], labels=self.classifier['Target'], name='Error_Cost'))

                # The final optimization
                with tf.name_scope("Trainers"): 
                    # Call the other optimizer
                    self.Trainer["Weight_op"] = self.Custom_Optimizer(self.classifier["learning_rate"], optimizer)
                    
                # The evaluation section of the methodology
                with tf.name_scope('Evaluation'):
                    self.Evaluation['correct_prediction'] = \
                            tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class']),1 ) , 
                                    tf.argmax(self.classifier['Target'], 1))
                    self.Evaluation['accuracy'] = tf.reduce_mean(
                            tf.cast(self.Evaluation['correct_prediction'], tf.float32))

            self.sess.run(tf.global_variables_initializer())
            return self