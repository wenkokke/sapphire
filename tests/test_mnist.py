
"""
test_mnist.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import trange
from z3 import *
from sapphire import *

#%% Set hyperparameters
latent_dim = 64
num_jacobians = 25
top_r = 5

#%% Functions

#%% Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype(float) / 255
x_test = x_test.astype(float) / 255

# TODO: For testing only a smaller binary classifier
x_train = x_train[np.isin(y_train, [0, 1])]
x_test = x_test[np.isin(y_test, [0, 1])]
y_train = y_train[np.isin(y_train, [0, 1])]
y_test = y_test[np.isin(y_test, [0, 1])]
x_train, x_test, y_train, y_test = train_test_split(
    x_test, y_test, test_size=1/7)

#%% Train autoencoder
class Autoencoder(tf.keras.Model):
    
  def __init__(self, latent_dim, output_shape):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            input_shape=(784, ), units=128, activation='tanh', use_bias=True),
        tf.keras.layers.Dense(
            units=latent_dim, activation='tanh', use_bias=True)])
    self.decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(
            units=128, activation='tanh', use_bias=True),
        tf.keras.layers.Dense(
            units=784, activation='sigmoid', 
            use_bias=True),
        tf.keras.layers.Reshape((output_shape[0], output_shape[1]))])
 
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim=latent_dim, output_shape=(28, 28))
autoencoder.compile(
    optimizer='adam', loss='mean_squared_error')
autoencoder.fit(x_train, x_train, epochs=1, batch_size=32)

rand_ind = np.random.randint(
    low=0, high=x_train.shape[0], size=1)
x_sample = x_train[rand_ind]
x_sample_recon = autoencoder.predict(x_sample)

fig, (axis_1, axis_2) = plt.subplots(nrows=1, ncols=2)
axis_1.imshow(x_sample.reshape((28, 28)))
axis_2.imshow(x_sample_recon.reshape((28, 28)))

#%% 
data_ls = []
top_sing_vecs_ls = []
# for i in trange(x_train.shape[0]):
for i in trange(num_jacobians):
    i = 0
    x_sample = x_train[i]
    
    with tf.GradientTape() as g:
        x = tf.constant(x_sample.reshape((1, 784)))
        g.watch(x)
        y = autoencoder.encoder(x)
    jacobian = g.batch_jacobian(y, x).numpy()
    jacobian = jacobian.reshape((latent_dim, x_train.shape[-1]*x_train.shape[-2]))
    
    # _, singluar values, singluar vectors (rows)
    u, s, vh = np.linalg.svd(
        a=jacobian, full_matrices=True, compute_uv=True)
    
    top_sing_vecs_ls.append(vh[:top_r])
    data_ls.append(pd.DataFrame(
        {'IMAGE_INDEX':[str(i)]*len(s),
         'SING_VALS':s.tolist(),
         'SING_VAL_ORD':[i for i in range(len(s))]}))
    
tangent_basis = np.vstack(top_sing_vecs_ls)
sing_val_df = pd.concat(data_ls)
sing_val_df_descr = sing_val_df.groupby('SING_VAL_ORD').describe()

fig, (axis_1, axis_2) = plt.subplots(nrows=1, ncols=2)
plt.suptitle(
    "Ordered sing. val. distr. of encoder's Jacobian averaged over {} images".format(
        num_jacobians))
axis_1.set(
    ylabel='Singular value', xlabel='Singular value order',
    title='Average sing. values')
axis_1.plot(
    sing_val_df['SING_VAL_ORD'].unique().tolist(), 
    sing_val_df_descr['SING_VALS']['mean'].tolist())
axis_1.scatter(
    sing_val_df['SING_VAL_ORD'].unique().tolist(), 
    sing_val_df_descr['SING_VALS']['mean'].tolist(),
    s=1.5)
axis_1.fill_between(
    x=sing_val_df['SING_VAL_ORD'].unique().tolist(), 
    y1=sing_val_df_descr['SING_VALS']['mean']+sing_val_df_descr['SING_VALS']['std'],
    y2=sing_val_df_descr['SING_VALS']['mean']-sing_val_df_descr['SING_VALS']['std'],
    alpha=0.2)
axis_2.set(
    xlabel='Singular value order', ylabel='',
    title='Top-{} sing. values'.format(top_r))
axis_2.plot(
    sing_val_df['SING_VAL_ORD'].unique().tolist()[:top_r], 
    sing_val_df_descr['SING_VALS']['mean'].tolist()[:top_r])
# plt.savefig(
#     image_dir + '/singval_dist_{}.png'.format(
#         num_jacobians))

#%% Train evaluation net
eval_net = keras.Sequential()
eval_net.add(keras.layers.Flatten(input_shape=(28, 28)))
eval_net.add(keras.layers.Dense(64, activation='relu'))
eval_net.add(keras.layers.Dense(2, activation='softmax'))

eval_net.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])
eval_net.fit(x_train, y_train, epochs=1)
test_loss, test_acc = eval_net.evaluate(x_test, y_test, verbose=2)

#%% Verify network
# def InTangentSpace(x, t_Bx):
#     """
#     x and x_Bx are assumed to be column vectors.
#     """
#     variable_names = ['c_{}'.format(i) for i in range(1, t_Bx.shape[1]+1)]
#     print(variable_names)
#     # TODO: Manually change the  number of return variables:
#     # variables = Reals(' '.join(variable_names))
#     c_1, c_2, c_3 = Reals(' '.join(variable_names))

#     solver = Solver()
#     for i in range(t_Bx.shape[0]):
#         vars_and_coeffs = list(zip(t_Bx[i].tolist(), variable_names))
#         row = '+'.join([str(expr[0]) + '*' + (expr[1]) for expr in vars_and_coeffs])
#         row = row + '==' + str(x[i][0])
#         solver.add(eval(row))
    
#     print(solver)
#     result = str(solver.check())
#     print(result)
#     if result == 'sat':
#         return True
#     else:
#         return False
    
def InTangentSpace(X, T, salt=""):
    # X is a "vector" of symbolic reals (using Python lists)
    # T is a "matrix" of symbolic reals (using Python lists)
    C = [Real('c_{}_{}'.format(i, salt)) for i in range(1, len(T[0]))]
    return And([x == product([ci + ti for ci, ti in zip(C, t)]) for x, t in zip(X, T)])

def sample_dataset(label):
    """Select a random sample with a particular label."""
    label_mask = (y_train == label).flatten()
    label_count = np.count_nonzero(label_mask)
    return x_train[label_mask][
        np.random.randint(low=0, high=label_count, size=1)][0]

x_sample = sample_dataset(label=0)
y_sample = eval_net.predict(np.array([x_sample]))[0]

X, Y = NN(eval_net)

# BUG: b'parser error'
x_sample = list(RealVal(x) for x in x_sample.flatten())
y_sample = list(RealVal(y) for y in y_sample.flatten())

def Eq(X1, X2):
    return And([x1 == x2 for x1, x2 in zip(X1, X2)])

s = SolverFor('NRA')
s.add(ForAll(And(X, InTangentSpace(x=, t_Bx=)), 
              Implies(Eq(X, X_sample), Y[0]>Y[1])))
print(s.check())

