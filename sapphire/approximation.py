
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time


class LinearSplines():
    
    def __init__(self, func, x_interval, num_lin_pieces, cond_func):
        """
        Computes a piece-wise approximation of the passed function 'func' and
        stores it as a nested expression of conditions. The evaluatable 
        activation-string has a form like the following:
            cond_func(x<0, m*x +1, cond_func(x<5, m*x +2, ...)...)
        
        :usage 
            1) Create an instance and pass the function to be approximated, 
            the interval over which it should be approximated, the number of 
            linear pieces and the condition function to it:
            activation = LinearSplines(
                func=np.log, x_interval=(-5, 5), 
                num_lin_pieces=10,
                cond_func=tf.where)
            2) activation.as_string returns the nested expression as a string,
            like to one displayed above.
            2) (Optional) When used as an activation function for a keras neural network:
            Pass 'activation.call' to a keras-layer instance just like every 
            other activation function
        
        :param func, Python or numpy function; the function to be linearly 
            approximated, for example np.log, np.exp, etc.
        :param x_interval, tuple; x-range over which the function should be 
            approximated
        :param num_lin_pieces, int; number of linear function used to 
            approximate function 'func'
        :param cond_func, Python, Numpy, Tensorflow, Z3, ... function; the 
            function for the nested expression, for example tf.where, z3.If, 
            etc., of the form cond_func(<condition>, <true>, <false>)
        """
        
        (self.point_2_x_ls, self.f_x_ls, self.xs, self.xs_approx, self.ys, 
         self.y_approx_ls) = LinearSplines.approximate(
            func=func, x_interval=x_interval, 
            num_lin_pieces=num_lin_pieces)  
             
        self.activation_str = LinearSplines.to_string(
            point_2_x_ls=self.point_2_x_ls, f_x_ls=self.f_x_ls)
        
        self.cond_func = cond_func
    
    
    @staticmethod
    def approximate(func, x_interval, num_lin_pieces):
        """
        Approximates the given function 'func' over the interval 
        [x_interval[0], x_interval[1]] by 'num_lin_pieces' linear functions.
        
        :param func, Python or numpy function; the function to be linearly 
            approximated, for example np.log, np.exp, etc.
        :param x_interval, tuple; x-range over which the function should be 
            approximated
        :param num_lin_pieces, int; number of linear function used to 
            approximate function 'func'
        
        :return point_2_x_ls, list of strings; the x-coordinate of the second 
            point of a line equation
        :return f_x_ls, list of strings; the line equation between two points
        :return xs, numpy array; x-coordinates of the precise activation
        :return xs_approx, numpy array; x-coordinates of the piece-wise 
            approximation
        :return ys, numpy array; y-coordinates of the precise activation
        :return y_approx_ls, list; y-coordinates of the piece-wise 
            approximation
        """
    
        xs = np.linspace(
            start=x_interval[0], stop=x_interval[1], num=1000)
        xs_approx = np.linspace(
            start=x_interval[0], stop=x_interval[1], num=num_lin_pieces+1)
        ys = func(xs)
    
        y_approx_ls = []
        point_2_x_ls = []
        f_x_ls = []
    
        for index in range(len(xs_approx[:-1])):
            
            point_1 = (xs_approx[index], func(xs_approx[index]))
            point_2 = (xs_approx[index+1], func(xs_approx[index+1]))
            
            m = LinearSplines.slope(
                point_1=point_1, point_2=point_2)
            n = LinearSplines.y_axis_cut(point=point_2, slope=m)
        
            xs_line_seg = np.linspace(
                start=point_1[0], stop=point_2[0], num=100)
            
            f_x = (m*xs_line_seg + n).tolist()[0]
            y_approx_ls.append(f_x)
            
            point_2_x_ls.append(str(point_2[0]))
            f_x_ls.append('{m}*x + {n}'.format(
                m=m, n=n))
            
        y_approx_ls.append(ys[-1])
        
        return point_2_x_ls, f_x_ls, xs, xs_approx, ys, y_approx_ls
    
    
    def call(self, x):
        """
        Evaluates the computed actvation string for a passed x. This method 
        can be directly passed to a keras layer as the variable for the 
        'activation' parameter
    
        :param x, numpy array; input
        
        :return (y), numpy; output
        """
        
        return eval(
            self.activation_str, 
            {'condition_func':self.cond_func, 'x':x})

    
    def get_computations(self):
        """
        Returns all computed quantities.
        """
        
        return (
            self.point_2_x_ls, self.f_x_ls, self.xs, self.xs_approx, self.ys, self.y_approx_ls)

    
    @staticmethod
    def plot(point_2_x_ls, f_x_ls, xs, xs_approx, ys, y_approx_ls, num_lin_pieces):
        """
        Plot the prcise afction function and its linear approximation.
        
        :param point_2_x_ls, list of strings; the x-coordinate of the second point
            of a line equation
        :param f_x_ls, list of strings; the line equation between two points
        :param xs, numpy array; x-coordinates of the precise activation
        :param xs_approx, numpy array; x-coordinates of the piece-wise 
            approximation
        :param ys, numpy array; y-coordinates of the precise activation
        :param y_approx_ls, list; y-coordinates of the piece-wise 
            approximation
        :param, num_lin_pieces, int; number of linear pieces to approximate
            the function with
        """
        
        plt.figure()
        plt.title(
            'Precise function and its approximation by {num_lin_pieces} line segments'.format(
                num_lin_pieces=num_lin_pieces))
        plt.plot(
            xs, ys, color='green', label='precise',
            alpha=0.75)
        plt.plot(
            xs_approx, y_approx_ls, color='red', label='approx.',
            alpha=0.75)
        plt.scatter(xs_approx, y_approx_ls, label='separators')
        plt.legend(loc='upper left')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.show()
        
        
    @staticmethod
    def slope(point_1, point_2):
        """
        Computes the slope 'm' for a linear function connecting 'point_1' and 'point_2' according
        to the formula
            m = (y_2 - y_1) / (x_2 - x_1)
        where 'y_i' is the y-axis value for point 'i' and 'x_i' is the x-axis coordinate for point 'i'.
        
        :param point_1, tuple; point 1 of form (x_1, y_1)
        :param point_1, tuple; point 2 of form (x_2, y_2)
        
        :return (slope), float; slope of line segement through points one and two
        """
    
        return (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
    
    
    def string(self):
        """
        Returns the activation function as an evaluable string.
        """
        
        return self.activation_str
    
    
    @staticmethod
    def to_string(point_2_x_ls, f_x_ls):
        """
        Returs the function call for the computed piece-wise apprximation
        as a string to be evaluated.
        
                
        :param point_2_x_ls, list of strings; the x-coordinate of the second point
            of a line equation
        :param f_x_ls, list of strings; the line equation between two points
        
        return cond_str, string; the piece-wise linear approximation of the 
            function as a nested conditional expression
        """
        
        cond_str = 'condition_func(x <= {x}, {f_x}, '.format(
            x=point_2_x_ls[0], f_x=f_x_ls[0])
         
        for index in range(1, len(point_2_x_ls)-1):
            cond_str += 'condition_func(x <= {x}, {f_x}, '.format(
                x=point_2_x_ls[index], f_x=f_x_ls[index])
            
        cond_str += '{f_x})'.format(f_x=f_x_ls[-1])
        cond_str += ')'*(len(point_2_x_ls)-2)
        
        return cond_str


    @staticmethod
    def y_axis_cut(point, slope):
        """
        Compute the point where a linear function of the form f(x) = mx + n cuts the y-axis, so value 'n', 
        where 'm' is the 'slope' of the function.
        
        :param point, tuple; arbitray point on the linear function
        :param slope, float; slope of linear function
        
        :return (n), float; n = f(x) - m*x, point where the lienar function cuts the y-axis
        """
    
        return point[1] - slope*point[0]


# #%% Demo

# #%%% Define activation functions     
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


# #%%% Define neural network
    

# def train_model(activation_hidden, activation_top, x_train, y_train, x_test, y_test):
#     """
#     """
    
#     model = Sequential()
#     model.add(Dense(
#         512, activation=activation_hidden, input_shape=(784,)))
#     model.add(Dense(
#         512, activation=activation_hidden))
#     model.add(Dense(10, activation=activation_top))
    
#     model.compile(
#         loss='categorical_crossentropy', optimizer='adam',
#         metrics=['accuracy'])
    
#     model.fit(
#         x_train, y_train, batch_size=128, epochs=10)
#     train_acc = model.evaluate(x_train, y_train, verbose=0)[1]
#     test_acc = model.evaluate(x_test, y_test, verbose=0)[1]
    
#     return train_acc, test_acc


# #%%% Load data
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train.reshape(60000, 784).astype('float32') / 255
# x_test = x_test.reshape(10000, 784).astype('float32') / 255

# y_train = tf.keras.utils.to_categorical(y=y_train, num_classes=10)
# y_test = tf.keras.utils.to_categorical(y=y_test, num_classes=10)


# #%%% Benchmark

# #%%%% Example plots

# x_interval = (-5, 5)
# num_lin_pieces = 3
# func = sigmoid

# activation = LinearSplines(
#     func=func, x_interval=(-5, 5), 
#     num_lin_pieces=num_lin_pieces,
#     cond_func=tf.where)

# activation_str = activation.activation_str
# print('activation_str: ', activation_str, '\n')

# point_2_x_ls, f_x_ls, xs, xs_approx, ys, y_approx_ls = activation.get_computations()
# activation.plot(
#     point_2_x_ls=point_2_x_ls, f_x_ls=f_x_ls, xs=xs, 
#     xs_approx=xs_approx, ys=ys, 
#     y_approx_ls=y_approx_ls, 
#     num_lin_pieces=num_lin_pieces)


# #%%%% 1) sigmoid_approx + softmax_precise

# x_interval = (-5, 5)
# num_lin_pieces = 10
# activation_hidden = sigmoid
# activation_top = 'softmax'

# activation = LinearSplines(
#     func=sigmoid, x_interval=(-5, 5), 
#     num_lin_pieces=10,
#     cond_func=tf.where)

# activation_str = activation.activation_str
# print('activation_str: ', activation_str, '\n')

# point_2_x_ls, f_x_ls, xs, xs_approx, ys, y_approx_ls = activation.get_computations()
# activation.plot(
#     point_2_x_ls=point_2_x_ls, f_x_ls=f_x_ls, xs=xs, 
#     xs_approx=xs_approx, ys=ys, 
#     y_approx_ls=y_approx_ls, 
#     num_lin_pieces=num_lin_pieces)

# time_beg = time.time()
# train_acc_1, test_acc_1 = train_model(
#     activation_hidden=activation.call, 
#     activation_top=activation_top, x_train=x_train, 
#     y_train=y_train, x_test=x_test, y_test=y_test)
# time_end = time.time()
# time_1 = round(time_end - time_beg, 4)


# #%%%% 2) sigmoid_approx + softmax_approx

# x_interval = (-5, 5)
# num_lin_pieces = 10
# activation_hidden = sigmoid
# activation_top = np.exp


# # Hidden layer activation:
# activation_hidden = LinearSplines(
#     func=sigmoid, x_interval=(-5, 5), 
#     num_lin_pieces=10,
#     cond_func=tf.where)

# activation_hidden_str = activation_hidden.activation_str
# print('activation_hidden_str: ', activation_hidden_str, '\n')

# point_2_x_ls, f_x_ls, xs, xs_approx, ys, y_approx_ls = activation_hidden.get_computations()
# activation_hidden.plot(
#     point_2_x_ls=point_2_x_ls, f_x_ls=f_x_ls, xs=xs, 
#     xs_approx=xs_approx, ys=ys, 
#     y_approx_ls=y_approx_ls, 
#     num_lin_pieces=num_lin_pieces)


# # Top layer activation:
# activation_top = LinearSplines(
#     func=activation_top, x_interval=(-10, 10), 
#     num_lin_pieces=20,
#     cond_func=tf.where)

# activation_top_str = activation_top.activation_str
# print('activation_top_str: ', activation_top_str, '\n')

# point_2_x_ls, f_x_ls, xs, xs_approx, ys, y_approx_ls = activation_top.get_computations()
# activation_top.plot(
#     point_2_x_ls=point_2_x_ls, f_x_ls=f_x_ls, xs=xs, 
#     xs_approx=xs_approx, ys=ys, 
#     y_approx_ls=y_approx_ls, 
#     num_lin_pieces=num_lin_pieces)

# def softmax(x):
#     summed = tf.reshape(
#         tensor=tf.reduce_sum(activation_top.call(x), axis=1), 
#         shape=(tf.shape(x)[0], 1))
        
#     summed_tile = tf.tile(
#         input=summed, 
#         multiples=(1, tf.shape(x)[1]))
                    
#     return activation_top.call(x) / summed_tile
    
# time_beg = time.time()
# train_acc_2, test_acc_2 = train_model(
#     activation_hidden=activation.call, 
#     activation_top=softmax, x_train=x_train, 
#     y_train=y_train, x_test=x_test, y_test=y_test)
# time_end = time.time()
# time_2 = round(time_end - time_beg, 4)


# #%%%% 3) sigmoid_precise + softmax_precise

# x_interval = (-5, 5)
# num_lin_pieces = 10
# activation_hidden = 'sigmoid'
# activation_top = 'softmax'

# time_beg = time.time()
# train_acc_3, test_acc_3 = train_model(
#     activation_hidden=activation_hidden, 
#     activation_top=activation_top, x_train=x_train, 
#     y_train=y_train, x_test=x_test, y_test=y_test)
# time_end = time.time()
# time_3 = round(time_end - time_beg, 4)


# df = pd.DataFrame(
#     {'Configuration':['sigmoid_approx + softmax_precise', 'sigmoid_approx + softmax_approx', 'sigmoid_precise + softmax_precise'],
#      'Train acc. (%)':[round(train_acc_1, 4), round(train_acc_2, 4), round(train_acc_3, 4)], 
#      'Test acc. (%)':[round(test_acc_1, 4), round(test_acc_2, 4), round(test_acc_3, 4)],
#      'Run time (sec.)': [time_1, time_2, time_3]})
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df.to_latex())
