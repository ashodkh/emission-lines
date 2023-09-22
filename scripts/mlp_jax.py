from jax.nn import selu
from jax import numpy as jnp
from jax.nn import selu
from copy import deepcopy
from jax.nn import selu
from jax import vmap as vmap
from jax import grad
from jax import jit as jjit
from jax import random as jran

# module that contains functions to run a JAX simple neural network

def get_network_layer_sizes(n_features, n_targets, dense_layer_sizes):
    '''
        n_features (int) is the dimensionality of the feature space
        n_targets (int) is the dimensionality of the output space
        dense_layer_sizes (list) is a list of hidden layer sizes
    '''
    layer_sizes = [n_features, *dense_layer_sizes, n_targets]
    return layer_sizes

def get_init_network_params(sizes, ran_key):
    """Initialize all layers for a fully-connected neural network."""
    keys = jran.split(ran_key, len(sizes))
    return [get_random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def get_random_layer_params(m,n, ran_key, scale=0.1):
    w_key, b_key = jran.split(ran_key)
    ran_weights = scale*jran.normal(w_key,(n,m))
    ran_biases = scale*jran.normal(b_key, (n,))
    return ran_weights, ran_biases

@jjit
def update(params, x, y, w, learning_rate):
    grads = grad(mse_loss)(params, x, y, w)
    return [(w-learning_rate*dw, b-learning_rate*db) for (w, b), (dw, db) in zip(params,grads)]


@jjit
def adam_update(params, x, y, w, m0, v0, beta1, beta2, lr, eps):
    grads = grad(mse_loss)(params, x, y, w)
    m0 = [(beta1*m0_w + (1-beta1)*dw, beta1*m0_b + (1-beta1)*db) for (m0_w,m0_b), (dw,db) in zip(m0,grads)]
    v0 = [(beta2*v0_w + (1-beta2)*dw**2, beta2*v0_b + (1-beta2)*db**2) for (v0_w,v0_b), (dw,db) in zip(v0,grads)]
    params = [(w-lr*m0_w/(jnp.sqrt(v0_w)+eps), b-lr*m0_b/(jnp.sqrt(v0_b)+eps))\
                 for (w,b), (m0_w, m0_b), (v0_w, v0_b) in zip(params, m0, v0)]
    return params, m0, v0
    
    
def mse_loss(params, abscissa, targets, weights):
    preds = batched_prediction(params, abscissa)
    return jnp.sum((preds-targets)**2*weights)/preds.shape[0]

def feedforward_prediction(params, abscissa):
    activations = abscissa
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = selu(outputs)

    w_final, b_final = params[-1]
    final_outputs = jnp.dot(w_final, activations) + b_final
    return final_outputs[0]

batched_prediction = vmap(feedforward_prediction, in_axes=(None, 0))

