#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class Agent():

    class ActorNetwork(tf.keras.Model):

        def __init__(self, action_size):

            super().__init__()
            # Shared layers for policy and value function networks
            self.layer1 = tf.keras.layers.Dense(256, activation = 'relu')
            self.layer2 = tf.keras.layers.Dense(256, activation = 'relu')
            self.pout = tf.keras.layers.Dense(3, activation = 'softmax') # state to action probabilities

        def call(self, state):

            x = tf.convert_to_tensor(state)
            x = self.layer1(x)
            x = self.layer2(x)
            probs = self.pout(x)

            return probs

    class CriticNetwork(tf.keras.Model):

        def __init__(self):

            super().__init__()
            # Shared layers for policy and value function networks
            self.layer1 = tf.keras.layers.Dense(256, activation = 'relu')
            self.layer2 = tf.keras.layers.Dense(256, activation = 'relu')
            self.vout = tf.keras.layers.Dense(1, activation = None) # ... from state to value scalar

        def call(self, state):

            x = tf.convert_to_tensor(state)
            x = self.layer1(x)
            x = self.layer2(x)
            value = self.vout(x)

            return value

    def __init__(self, action_size):
        self.aModel = self.ActorNetwork(action_size)
        self.vModel = self.CriticNetwork()
        self.gamma = 0.99
        self.alr = 1e-4
        self.clr = 5e-4
        self.aopt = tf.keras.optimizers.Adam(learning_rate=self.alr)
        self.copt = tf.keras.optimizers.Adam(learning_rate=self.clr)

    def choose_action(self, state):
        a_probs = self.aModel(np.array([state]))
        dist = tfp.distributions.Categorical(probs = a_probs, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def calculate_loss(self, a_probs, action, delta):
        dist = tfp.distributions.Categorical(probs = a_probs, dtype=tf.float32)
        log_prob = dist.log_prob(action) # get log(pi(a\s))
        return -log_prob*delta

    def train(self, state, action, reward, next_state, terminated): # train from an episode of experience

        state = np.array([state])
        next_state = np.array([next_state])

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

            a =  self.aModel(state,training=True) # action probabilities and value of state S_t
            v = self.vModel(state, training=True) # action probabilities and value of state S_t+1
            v_p = self.vModel(next_state, training=True)
            td = reward + self.gamma*v_p - v # calculate TD error

            a_update = self.calculate_loss(a, action, td)
            c_update = td**2

        self.aopt.minimize(a_update, self.aModel.trainable_variables, tape=tape1)
        self.copt.minimize(c_update, self.vModel.trainable_variables, tape=tape2)

        return a_update, c_update
