#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import random
from utils.Mish import Mish

class Agent():

    class ActorNetwork(tf.keras.Model):

        def __init__(self):

            super().__init__()
            # Shared layers for policy and value function networks
            self.layer1 = tf.keras.layers.Dense(256, activation = Mish())
            self.layer2 = tf.keras.layers.Dense(256, activation = Mish())
            self.mout = tf.keras.layers.Dense(3, activation = 'softmax') # state to action probabilities

        def call(self, state):

            x = tf.convert_to_tensor(state)
            x = self.layer1(x)
            x = self.layer2(x)
            move_out = self.mout(x)

            return move_out


    class CriticNetwork(tf.keras.Model):

        def __init__(self):

            super().__init__()
            # Shared layers for policy and value function networks
            self.layer1 = tf.keras.layers.Dense(256, activation = Mish())
            self.layer2 = tf.keras.layers.Dense(256, activation = Mish())
            self.vout = tf.keras.layers.Dense(1, activation = None) # ... from state to value scalar

        def call(self, state):

            x = tf.convert_to_tensor(state)
            x = self.layer1(x)
            x = self.layer2(x)
            value = self.vout(x)

            return value

    def __init__(self, action_size_comm, alr, vlr, ecoef, num_arms):
        self.aModel = self.ActorNetwork()
        self.vModel = self.CriticNetwork()
        self.gamma = 0.99
        self.ent_coef = ecoef

        self.alr = alr
        self.vlr = vlr

        self.aopt = tf.keras.optimizers.Adam(learning_rate=self.alr)
        self.vopt = tf.keras.optimizers.Adam(learning_rate=self.vlr)

        self.num_arms = num_arms
        self.prior_mean_mu = 0.0
        self.prior_mean_sigma = 1.0
        self.prior_var_alpha = 1.0
        self.prior_var_beta = 1.0

        # Initialize parameters for each arm using a NumPy array
        initial_means = np.random.normal(self.prior_mean_mu, self.prior_mean_sigma, num_arms)
        initial_variances = 1.0 / np.random.gamma(self.prior_var_alpha, 1.0 / self.prior_var_beta, num_arms)

        # Create a NumPy array to store the parameters for each arm
        self.prior_parameters = np.column_stack((initial_means, initial_variances))

    def select_arm(self):
        # Use NumPy operations for sampling from the posterior
        sampled_means = np.random.normal(self.prior_parameters[:, 0], np.sqrt(self.prior_parameters[:, 1]))
        sampled_variances = np.random.gamma(self.prior_var_alpha, 1.0 / self.prior_var_beta, self.num_arms)
        sampled_parameters = np.column_stack((sampled_means, sampled_variances))
        return np.argmax(sampled_parameters[:, 0])

    def update_comm(self, arm, reward):
        # Update the posterior distribution based on the observed reward
        # For simplicity, let's assume a conjugate update for Gaussian distribution
        updated_mean = (self.prior_parameters[arm, 1] * reward +
                        self.prior_mean_sigma**2 * self.prior_parameters[arm, 0]) / \
                       (self.prior_parameters[arm, 1] + self.prior_mean_sigma**2)

        updated_variance = 1.0 / (1.0 / self.prior_parameters[arm, 1] + 1.0 / self.prior_mean_sigma**2)

        # Update the parameters for the selected arm
        self.prior_parameters[arm, 0] = updated_mean
        self.prior_parameters[arm, 1] = updated_variance

    def choose_action_move(self, state):
        move_out= self.aModel(np.array([state]))

        dist_move = tfp.distributions.Categorical(probs=move_out, dtype=tf.float32) # categorical dist
        action_move = dist_move.sample() # ... sampled to get movement action

        return int(action_move.numpy()[0])

    @tf.function
    def train(self, state, action, reward, next_state): # train from an episode of experience

        state = tf.reshape(state, (1, -1))
        next_state = tf.reshape(next_state, (1, -1))
        reward = tf.cast(reward, tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            # critic loss calculation
            v = self.vModel(state, training=True)
            v_p = self.vModel(next_state, training=True)
            td = reward + self.gamma * v_p - v
            loss_critic = tf.reduce_mean(td**2)  # Mean squared error

            # Calculate the log probabilities and losses for both action types
            action_move, action_comm = action[0], action[1]

            # move loss
            move_out = self.aModel(state, training=True)
            dist_move = tfp.distributions.Categorical(probs=move_out, dtype=tf.float32)
            entropy = dist_move.entropy()
            log_prob_move = dist_move.log_prob(action_move)
            loss_actor = -log_prob_move * td - entropy * self.ent_coef

        grads_actor = tape.gradient(loss_actor, self.aModel.trainable_variables)
        grads_critic = tape.gradient(loss_critic, self.vModel.trainable_variables)

        self.aopt.apply_gradients(zip(grads_actor, self.aModel.trainable_variables))
        self.vopt.apply_gradients(zip(grads_critic, self.vModel.trainable_variables))