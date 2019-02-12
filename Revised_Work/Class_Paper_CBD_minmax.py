# The distributed learning paradigm
# Author : Krishnan Raghavan
# Date: June 22nd, 2018
#######################################################################################
# Define all the libraries
import random
import numpy as np
import tensorflow as tf
import operator
from functools import reduce

####################################################################################
# Helper Function for the weight and the bias variable initializations
# Weight
####################################################################################
def sample_Z(m, n, kappa):
    # return(np.random.uniform(-kappa, kappa, size=[m, n]))
    return(np.random.normal(kappa, kappa, size=[m, n]))


############################################################################################
def xavier(fan_in, fan_out):
   # use 4 for sigmoid, 1 for tanh activation
   low = -1 * np.sqrt(1.0 / (fan_in + fan_out))
   high = 1 * np.sqrt(1.0 / (fan_in + fan_out))
   return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)


############################################################################################
def weight_variable(shape, trainable, name):
   initial = xavier(shape[0], shape[1])
   return tf.Variable(initial, trainable=trainable, name=name)


#############################################################################################
# Bias function
def bias_variable(shape, trainable, name):
   initial = tf.random_normal(shape, trainable, stddev=1)
   return tf.Variable(initial, trainable=trainable, name=name)


#############################################################################################
#  Summaries for the variables
def variable_summaries(var, key):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries' + key):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev' + key):
           stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))


#############################################################################################
def init_ftn(name, num_input, num_output, runiform_range):
    if(name == "normal"):
        return(tf.truncated_normal([num_input, num_output]))
    elif(name == "uniform"):
        return(tf.random_uniform([num_input, num_output], minval=-runiform_range, maxval=runiform_range))
    else:
        print("not normal or uniform")


#############################################################################################
# The main Class
class learners():
    def __init__(self, gamma = 0.1, alpha=0.1, K=5):
        self.classifier = {}
        self.Deep = {}
        self.Trainer = {}
        self.Evaluation = {}
        self.keys = []
        self.sess = tf.Session()

        # Extra Parameters
        self.Layer =[]
        self.Weights =[]
        self.Cost =[]
        self.Layer_cost = []
        self.Noise_List =[]

        # Hyper-parameters
        self.alpha = alpha
        self.gamma = gamma
        self.K = K

#############################################################################################
    # Function for defining every NN
    def nn_layer(self, input_tensor, input_dim, output_dim, act,
                 trainability, key, act_par="tf.nn.tanh"):
        with tf.name_scope(key):
            with tf.name_scope('weights' + key):
                self.classifier['Weight' + key] = weight_variable(
                    [input_dim, output_dim], trainable=trainability, name='Weight' + key)
            with tf.name_scope('bias' + key):
                self.classifier['Bias' + key] = bias_variable(
                    [output_dim], trainable=trainability, name='Bias' + key)
            with tf.name_scope('Wx_plus_b' + key):
                preactivate = tf.matmul(
                    input_tensor, self.classifier['Weight' + key]) + self.classifier['Bias' + key]
            return act(preactivate, name='activation' + key)


#############################################################################################
    def Grad_Descent(self, lr):
        train_list = []
        for i in xrange(len(self.Weights)):
            weight, bias = self.Weights[i]
            ## Gradient Descent update
            weight_update = tf.gradients(self.classifier['Overall_cost'], weight)[0]   + 0.0001*weight
            bias_update   = tf.gradients(self.classifier['Overall_cost'],   bias)[0]   + 0.0001*bias
            # Generate the updated variables
            train_list.append((weight_update, weight))
            train_list.append((bias_update, bias))
        return tf.train.AdamOptimizer(lr).apply_gradients(train_list)


##################################################################################
    def Optimizer(self, lr, dError_dy):
        train_list = []
        a = []
        var_list = []
        self.classifier['real_cost'] = []
        self.classifier['updates'] = []
        for i in xrange(len(self.Layer_cost)):
            cost, weight, bias, fan_in, batch_size, layer_in,\
                backward = self.Layer_cost[i]
            if i < (len(self.Layer_cost) - 1):
                cost = self.Deep['FL_layer_1' + str(i+1)]
                dError_dhidden = tf.matmul(dError_dy, backward)
                reshaped_layer_in = tf.reshape(layer_in, [batch_size, fan_in, 1])
                upd_weight = tf.reduce_mean(tf.matmul(reshaped_layer_in, dError_dhidden), 0)
                upd_bias = tf.reduce_mean(dError_dhidden, 0)
                # Reshape back to output dimensions and then get the gradients.
                weight_update = 0.5*upd_weight  + 0.5*tf.gradients(self.classifier['Overall_cost'], weight)[0]+ 0.0001*weight
                bias_update   = 0.5*upd_bias[0] + 0.5*tf.gradients(self.classifier['Overall_cost'], bias)[0]  + 0.0001*bias
            else:
                reshaped_layer_in = tf.reshape(layer_in, [batch_size, fan_in, 1])
                dError_dhidden = dError_dy
                upd_weight = tf.reduce_mean(tf.matmul(reshaped_layer_in, dError_dhidden), 0)
                upd_bias = tf.reduce_mean(dError_dy, 0)
                weight_update = 0.5*upd_weight   + 0.5*tf.gradients(self.classifier['Overall_cost'], weight)[0] + 0.0001*weight
                bias_update   = 0.5*upd_bias[0]  + 0.5*tf.gradients(self.classifier['Overall_cost'],  bias )[0] + 0.0001*bias
           # Generate the updated variables
            train_list.append((weight_update, weight))
            train_list.append((bias_update, bias))
        return tf.train.AdamOptimizer(lr).apply_gradients(train_list)


#############################################################################################
    # Function for optimizing the noise
    def Noise_EDL_Optimizer(self, lr, dError_dy):
        train_list =[]
        _, weight, bias, fan_in, batch_size, layer_in, backward = self.Noise_List[0]
        dError_dhidden = tf.matmul(dError_dy, backward)
        reshaped_layer_in = tf.reshape(layer_in, [batch_size, fan_in, 1])
        upd_weight = tf.reduce_mean(tf.matmul(reshaped_layer_in, dError_dhidden), 0)
        upd_bias   = tf.reduce_mean(dError_dhidden, 0)

        ## updates
        weight_update = -0.5*upd_weight  -0.5*tf.gradients(self.classifier["Overall_cost"], weight)[0] + 0.0001*weight
        bias_update   = -0.5*upd_bias[0] -0.5*tf.gradients(self.classifier["Overall_cost"], bias)[0]   + 0.0001*bias

        ## Generate the updated variables
        train_list.append((weight_update, weight))
        train_list.append((bias_update, bias))
        return tf.train.AdamOptimizer(lr).apply_gradients(train_list)

        



#############################################################################################
    def Def_Network(self, classes, Layers, act_function, batch_size, back_range = 1):
        with tf.name_scope("Trainer_Network"):
            # The noise transfoirmations and the setup.
            # Start with defining the affine transformation
            self.classifier['A'] = weight_variable([Layers[0], Layers[0]], trainable=False, name='A')
            self.classifier['b'] = bias_variable([Layers[0]], trainable=False, name='b')

            # The first layer stuff
            # Defining the weight and bias variable for the first layer
            self.classifier['WeightFL_layer_11'] = weight_variable([Layers[0], Layers[1]], trainable=False, name='WeightFL_layer_11')
            self.classifier['BiasFL_layer_11']   = bias_variable([Layers[1]], trainable=False, name='BiasFL_layer_11')


            # The input noise model
            input_model              = tf.nn.sigmoid(tf.matmul(self.Deep['FL_layer_10'], self.classifier['A']) + self.classifier['b'])
            preactivate              = tf.matmul(input_model, self.classifier['WeightFL_layer_11']) + self.classifier['BiasFL_layer_11']
            self.Deep['FL_layer_11'] = act_function(preactivate)
            self.Weights.append((self.classifier['WeightFL_layer_11'], self.classifier['BiasFL_layer_11']))


            # weights for each layer is append to the training list
            fan_in = Layers[0]
            fan_out = Layers[0]
            num_final = classes
            cost_layer = self.Deep['FL_layer_11']
            weight_temp = self.classifier['A']
            bias_temp = self.classifier['b']
            layer_in = input_model
            backward_t = tf.Variable(init_ftn("uniform", num_final, fan_out, back_range))
            backward = tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [batch_size, num_final, fan_out])
            self.Noise_List.append((cost_layer, weight_temp, bias_temp, fan_in, batch_size, layer_in, backward))


            # The rest of the layers.
            for i in range(2, len(Layers)):
                key = 'FL_layer_1' + str(i)
                self.Deep['FL_layer_1' + str(i)]\
                = self.nn_layer (self.Deep['FL_layer_1' + str(i-1)], Layers[i-1],
                Layers[i], act=act_function, trainability=False, key=key)

                # weights for each layer is append to the training list
                self.Weights.append( (self.classifier['Weight'+key], self.classifier['Bias'+key]) )
                fan_in = Layers[i-2]
                fan_out = Layers[i-1]
                num_final = classes
                cost_layer = self.Deep['FL_layer_1' + str(i)]
                weight_temp = self.classifier['Weight' + 'FL_layer_1' + str(i-1)]
                bias_temp = self.classifier['Bias' + 'FL_layer_1' + str(i - 1)]
                layer_in = self.Deep['FL_layer_1' + str(i - 2)]

                backward_t = tf.Variable(init_ftn("uniform", num_final, fan_out, back_range))
                backward = tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [batch_size, num_final, fan_out])
                
                self.Layer_cost.append((cost_layer, weight_temp, bias_temp, fan_in, batch_size, layer_in, backward))



        # The final classification layer
        with tf.name_scope("Classifier"):
            self.classifier['class_Noise']=self.nn_layer(self.Deep['FL_layer_1' + str(len(Layers) - 1)],
            Layers[len(Layers)-1], classes, act=tf.identity, trainability=False, key='class')
            
            fan_in      =   Layers[len(Layers) - 2]
            fan_out     =   Layers[len(Layers) - 1]
            num_final   =   classes
            cost_layer  =   self.classifier['class_Noise']
            weight_temp =   self.classifier['Weight' + 'FL_layer_1' + str(len(Layers) - 1)]
            bias_temp   =   self.classifier['Bias'   + 'FL_layer_1' + str(len(Layers) - 1)]
            layer_in    =   self.Deep['FL_layer_1' + str(len(Layers) - 2)]
            backward_t  =   tf.Variable(init_ftn("uniform", num_final, fan_out, back_range))
            backward    =   tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [
                            batch_size, num_final, fan_out ])
            self.Layer_cost.append((cost_layer, weight_temp, bias_temp,
            fan_in, batch_size, layer_in, backward))
            
            # weights for the final layer is appended on to the training list
            self.Weights.append((self.classifier['Weight'+'class'], self.classifier['Bias'+'class']))



        # Finally, we define an additional network for producing noise free outputs
        ############ The network without the noise, primarily for output estimation
        self.Deep['FL_layer_3' + str(0)] = self.Deep['FL_layer_10']
        for i in range(1, len(Layers)):
            key = 'FL_layer_1' + str(i)
            preactivate = tf.matmul(self.Deep['FL_layer_3'+str(i-1)], self.classifier['Weight' + key]) + self.classifier['Bias' + key]
            self.Deep['FL_layer_3' +str(i)] = act_function(preactivate, name='activation_3' + key)


        self.classifier['class_NoNoise'] = tf.identity(tf.matmul(self.Deep['FL_layer_3' + str(len(Layers) - 1)],\
        self.classifier['Weightclass']) + self.classifier['Biasclass'])


 ############################################################################################################################
    def init_NN_custom(self, classes, lr, Layers, act_function, batch_size,
                        back_range,  par='GDR', act_par="tf.nn.tanh"):


        ########################### Initial Definitions
            with tf.name_scope("PlaceHolders"):  
                #### Setup the placeholders        
                # Label placeholder
                self.classifier['Target'] = tf.placeholder(tf.float32, shape=[None, classes])
                # Place holder for the learning rate
                self.classifier["learning_rate"] = tf.placeholder(tf.float32, [], name='learning_rate') 
                # Input placeholder
                self.Deep['FL_layer_10'] = tf.placeholder(tf.float32, shape=[None, Layers[0]])


        ######################################################################## The network
                # The main network
                self.Def_Network(classes, Layers, act_function, batch_size, back_range)        

        ################################################# Design the trainer for the  network
            with tf.name_scope("Trainer"):
                ## Model
                distance = tf.reduce_sum(tf.abs(tf.subtract(self.classifier['class_Noise'],\
                tf.expand_dims(self.classifier['class_NoNoise'], 1))), axis=2)

                # nearest k points
                _, top_k_indices = tf.nn.top_k(tf.negative(distance), k=self.K)
                top_k_label = tf.gather(self.classifier['Target'], top_k_indices)
                sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
                pred= tf.argmax(sum_up_predictions, axis=1)
                one_hot_a = tf.one_hot(pred, classes)

                ## Direct Costs
                self.classifier["cost_NN"]         = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                logits=self.classifier['class_Noise'],\
                labels=one_hot_a, name='Error_Cost')) 
                self.classifier["cost_NN_nonoise"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                logits=self.classifier['class_NoNoise'],\
                labels=self.classifier['Target'], name='Error_Cost_Nonoise')) 


                self.classifier["Cost_M1"] = (self.alpha)*self.classifier["cost_NN"]\
                +(1-self.alpha)*self.classifier["cost_NN_nonoise"]
                self.classifier["L2Cost"]  = tf.nn.l2_loss(self.classifier['A'])\
                +tf.nn.l2_loss(self.classifier['b'])


                ## KL divergence
                Dist_1 = tf.nn.softmax(self.classifier['class_NoNoise'])
                Dist_2 = tf.nn.softmax(self.classifier['class_Noise'])
                self.classifier["Cross_Entropy"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                logits=Dist_1, labels=Dist_2, name='CrossEntropy')) 
                self.classifier["KL"] = (np.square(self.gamma))*(self.classifier["Cross_Entropy"])


                ## The final cost function
                self.classifier["Overall_cost"] = self.classifier["Cost_M1"] - self.classifier["KL"]


                fan_in      =   Layers[len(Layers) - 1]
                fan_out     =   classes
                num_final   =   classes
                cost_layer  =   self.classifier["Overall_cost"]
                weight_temp =   self.classifier['Weightclass']
                bias_temp   =   self.classifier['Biasclass']
                layer_in    =   self.Deep['FL_layer_1' + str(len(Layers) - 1)]
                backward_t  =   tf.Variable(tf.eye(classes))
                backward_t  =   tf.Variable(init_ftn("uniform", num_final, fan_out, back_range))
                backward    =   tf.reshape(tf.stack([backward_t for _ in range(batch_size)]), [
                                batch_size, num_final, fan_out ])
                self.Layer_cost.append((cost_layer, weight_temp, bias_temp,
                fan_in, batch_size, layer_in, backward))

        ##########################################################################################################
        # The final optimization
            with tf.name_scope("Trainers"): 
                # Call the other optimizer
                self.Trainer["Grad_op"]  = self. Grad_Descent(self.classifier["learning_rate"])
                # self.Trainer["Noise_op"] = self.Noise_optimizer(self.classifier["learning_rate"])


                # The general optimizer
                dError_dy = 0.5*tf.reshape(tf.gradients(self.classifier["Overall_cost"], self.classifier['class_Noise'])[0], [batch_size, 1, num_final])\
                + 0.5*tf.reshape(tf.gradients(self.classifier["Overall_cost"], self.classifier['class_NoNoise'])[0], [batch_size, 1, num_final])


                # the optimizers
                self.Trainer['EDL_op'] = self.Optimizer(self.classifier["learning_rate"],  dError_dy,)
                
                # The general optimizer
                dError_dy_Noise = tf.reshape(tf.gradients(self.classifier["Overall_cost"], self.classifier['class_Noise'])[0], [batch_size, 1, num_final])
                
                #  the optimizers
                self.Trainer['EDL_Noise_op'] = self.Noise_EDL_Optimizer(self.classifier["learning_rate"],  dError_dy_Noise)

                
                
        ############## The evaluation section of the methodology
            with tf.name_scope('Evaluation'):
                self.Evaluation['correct_prediction'] = \
                        tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class_NoNoise']),1 ) , 
                                tf.argmax(self.classifier['Target'], 1))
                self.Evaluation['accuracy'] = tf.reduce_mean(
                        tf.cast(self.Evaluation['correct_prediction'], tf.float32))
            self.sess.run(tf.global_variables_initializer())


            return self




