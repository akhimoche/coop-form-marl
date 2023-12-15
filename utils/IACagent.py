#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class Agent():

    class ActorNetwork(tf.keras.Model):

        def __init__(self):

            super().__init__()
            # Shared layers for policy and value function networks
            self.layer1 = tf.keras.layers.Dense(256, activation = 'relu')
            self.layer2 = tf.keras.layers.Dense(256, activation = 'relu')
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
            self.layer1 = tf.keras.layers.Dense(256, activation = 'relu')
            self.layer2 = tf.keras.layers.Dense(256, activation = 'relu')
            self.vout = tf.keras.layers.Dense(1, activation = None) # ... from state to value scalar

        def call(self, state):

            x = tf.convert_to_tensor(state)
            x = self.layer1(x)
            x = self.layer2(x)
            value = self.vout(x)

            return value

    class CommNetwork(tf.keras.Model):

        def __init__(self, action_size_comm):

            super().__init__()
            # Shared layers for policy and value function networks
            self.layer1 = tf.keras.layers.Dense(256, activation = 'relu')
            self.cout = tf.keras.layers.Dense(action_size_comm, activation = 'softmax') # state to mean, stddev for gaussian p.d. dist.

        def call(self, state):

            x = tf.convert_to_tensor(state)
            x = self.layer1(x)
            value = self.cout(x)

            return value


    def __init__(self, action_size_comm, alr, vlr):
        self.aModel = self.ActorNetwork()
        self.vModel = self.CriticNetwork()
        self.cModel = self.CommNetwork(action_size_comm)
        self.gamma = 0.99

        self.alr = alr
        self.clr = alr
        self.vlr = vlr

        self.aopt = tf.keras.optimizers.Adam(learning_rate=self.alr)
        self.copt = tf.keras.optimizers.Adam(learning_rate=self.clr)
        self.vopt = tf.keras.optimizers.Adam(learning_rate=self.vlr)

    def choose_action_move(self, state):
        move_out= self.aModel(np.array([state]))

        dist_move = tfp.distributions.Categorical(probs=move_out, dtype=tf.float32) # categorical dist
        action_move = dist_move.sample() # ... sampled to get movement action

        return int(action_move.numpy()[0])

    def choose_action_comm(self, context):
        comm_out= self.cModel(np.array([context]))

        dist_comm = tfp.distributions.Categorical(probs=comm_out, dtype=tf.float32) # categorical dist
        action_comm = dist_comm.sample() # ... sampled to get movement action

        return int(action_comm.numpy()[0])

    def train(self, state, action, reward, next_state, context, singleton_val): # train from an episode of experience

        state = np.array([state])
        next_state = np.array([next_state])
        context = np.array([context])

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
            log_prob_move = dist_move.log_prob(action_move)
            loss_actor = -log_prob_move * td

            # comm loss
            comm_out = self.cModel(context, training=True)
            dist_comm = tfp.distributions.Categorical(probs=comm_out, dtype=tf.float32)
            log_prob_comm = dist_comm.log_prob(action_comm)
            loss_comm = -log_prob_comm * (reward-singleton_val)

        grads_actor = tape.gradient(loss_actor, self.aModel.trainable_variables)
        grads_critic = tape.gradient(loss_critic, self.vModel.trainable_variables)
        grads_comm = tape.gradient(loss_comm, self.cModel.trainable_variables)

        # numerical stability check
        for grad in grads_actor:
            if tf.math.reduce_any(tf.math.is_nan(grad)) or tf.math.reduce_any(tf.math.is_inf(grad)):
                raise ValueError(f"NaNs or infs in actor gradient. Stopping training.")
        for grad in grads_critic:
            if tf.math.reduce_any(tf.math.is_nan(grad)) or tf.math.reduce_any(tf.math.is_inf(grad)):
                raise ValueError(f"NaNs or infs in critic gradient. Stopping training.")
        for grad in grads_comm:
            if tf.math.reduce_any(tf.math.is_nan(grad)) or tf.math.reduce_any(tf.math.is_inf(grad)):
                raise ValueError(f"NaNs or infs in comm gradient. Stopping training.")

        self.aopt.apply_gradients(zip(grads_actor, self.aModel.trainable_variables))
        self.copt.apply_gradients(zip(grads_comm, self.cModel.trainable_variables))
        self.vopt.apply_gradients(zip(grads_critic, self.vModel.trainable_variables))