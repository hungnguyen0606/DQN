import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
import numpy as np


class QEstimator:
    """
    Estimator for Q value.
        sample_action: function that generate a random action
    """
    def __init__(self, sample_action, state_size, num_actions, learning_rate, scope='Estimator'):
        self.scope = scope
        self.state_size = state_size
        self.num_actions = num_actions
        self.lr = learning_rate
        # function that return ONE random action
        self.sample_action = sample_action

        with tf.variable_scope(self.scope):
            self.build_model()

    def build_model(self):
        self.target = tf.placeholder(tf.float32, [None, 1], 'target')

        self.input = tf.placeholder(tf.float32, [None, self.state_size])
        self.hidden1 = layers.fully_connected(self.input, 256)
        self.hidden2 = layers.fully_connected(self.hidden1, 128)
        self.q_value = layers.fully_connected(self.hidden2, self.num_actions)

        self.prediction = tf.argmax(self.q_value, axis=1)[0]
        self.epsilon_prediction = tf.where(tf.placeholder(tf.bool, name='action_condition'),
                                        self.prediction,
                                        self.sample_action())

        self.action_taken = tf.placeholder(tf.int32, name='action_taken')
        self.action_mask = tf.one_hot(self.action_taken, self.num_actions)
        self.loss = tf.reduce_mean((tf.reduce_sum(self.q_value*self.action_mask, axis=1, keep_dims=True) - self.target)**2)
        self.train_ops = tf.train.RMSPropOptimizer(self.lr, 0.99, 0.0, 1e-6).minimize(self.loss)


    def Q_Value(self, sess, states):
        """
        Return a list containing q-values for list of states
        :param sess: tensorflow session instance
        :param states: (BATCH_SIZE * STATE_SIZE) numpy array containing states used to compute q-value
        :return: (BATCH_SIZE * NUM_ACTIONS) list of q-values
        """
        assert states.shape[1] == self.state_size, 'Shape of states doesn\'t match STATE_SIZE: %d != %d'%(states.shape[1], self.state_size)
        return sess.run(self.q_value, {self.input: states})

    def predict(self, sess, state):
        """
        Return ONE action according state, using greedy policy
        :param sess: tensorflow session instance
        :param state: state used to predict action
        :return: ONE action corresponding to the STATE
        """
        return sess.run(self.prediction, {self.input: np.array([state]).reshape(1, self.state_size)})

    def epsilon_predict(self, sess, state):
        """
        Return ONE action according state, using epsilon-greedy policy
        :param sess: tensorflow session instance
        :param state: state used to predict action
        :return:
        """
        return sess.run(self.epsilon_prediction, {self.input: np.array([state]).reshape(1, self.state_size)})

    def train(self, sess, states, actions, target):
        """
        Update gradient
        :param sess: tensorflow session instance
        :param states: (BATCH_SIZE * STATE_SIZE) np array list of states
        :param actions: 1-D array contains (BATCH_SIZE) actions (represented by integer)
        :param target: (BATCH_SIZE * 1) np array target values - reward + gamma*max_qvalue(target_net, next_state)
        :return: average loss value
        """
        assert np.array(states).shape[1] == self.state_size, 'Shape of states doesn\'t match STATE_SIZE: %d != %d' % (states.shape[1], self.state_size)

        loss, _ = sess.run([self.loss, self.train_ops], {self.input: states, self.target: target, self.action_taken: actions})
        return loss
