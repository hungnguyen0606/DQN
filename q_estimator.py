import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
import numpy as np


class QEstimator:
    """
    Estimator for Q value.
        sample_action: function that generate a random action
    """

    def __init__(self, sample_action, state_size, num_actions, scope='Estimator'):
        self.scope = scope
        self.state_size = state_size
        self.num_actions = num_actions
        # function that return ONE random action
        self.sample_action = sample_action

        with tf.variable_scope(self.scope):
            self.build_model()

    def build_model(self):
        self.target = tf.placeholder(tf.float32, [None, 1], 'target')

        self.input = tf.placeholder(tf.float32, [None, self.state_size], name='observation')
        self.hidden1 = layers.fully_connected(self.input, 512)
        self.hidden2 = layers.fully_connected(self.hidden1, 256)
        self.q_value = layers.fully_connected(self.hidden1, self.num_actions, activation_fn=None)

        self.prediction = tf.gather(tf.argmax(self.q_value, axis=1, name='greedy_action_array'), 0,
                                    name='greedy_action_scalar')


        self.action_taken = tf.placeholder(tf.int32, name='action_taken')
        self.action_mask = tf.one_hot(self.action_taken, self.num_actions, name='one_hot_actionmask')
        self.loss = tf.reduce_mean(
            (self.target - tf.reduce_sum(self.q_value * self.action_mask, axis=1, keep_dims=True)) ** 2, name='loss')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.train_ops = tf.train.RMSPropOptimizer(self.lr, 0.99, 0.0, 1e-6).minimize(self.loss)

    def Q_Value(self, sess, states):
        """
        Return a list containing q-values for list of states
        :param sess: tensorflow session instance
        :param states: (BATCH_SIZE * STATE_SIZE) numpy array containing states used to compute q-value
        :return: (BATCH_SIZE * NUM_ACTIONS) list of q-values
        """
        assert states.shape[1] == self.state_size, 'Shape of states doesn\'t match STATE_SIZE: %d != %d' % (
        states.shape[1], self.state_size)
        return sess.run(self.q_value, {self.input: states})

    def predict(self, sess, state):
        """
        Return ONE action according state, using greedy policy
        :param sess: tensorflow session instance
        :param state: state used to predict action
        :return: ONE action corresponding to the STATE
        """
        return sess.run(self.prediction, {self.input: np.array([state]).reshape(1, self.state_size)})

    def epsilon_predict(self, sess, state, eps):
        """
        Return ONE action according state, using epsilon-greedy policy

        :param sess: tensorflow session instance
        :param state: state used to predict action
        :param eps: epsilon
        :return:
        """
        if np.random.rand() < eps:
            return self.sample_action()
        return self.predict(sess, state)

    def train(self, sess, states, actions, target, lr):
        """
        Update gradient

        :param sess: tensorflow session instance
        :param states: (BATCH_SIZE * STATE_SIZE) np array list of states
        :param actions: 1-D array contains (BATCH_SIZE) actions (represented by integer)
        :param target: (BATCH_SIZE * 1) np array target values - reward + gamma*max_qvalue(target_net, next_state)
        :param lr: learning rate
        :return: average loss value
        """
        assert np.array(states).shape[1] == self.state_size, 'Shape of states doesn\'t match STATE_SIZE: %d != %d' % (
            states.shape[1], self.state_size)
        loss, _ = sess.run([self.loss, self.train_ops],
                           {self.input: states,
                            self.target: target,
                            self.action_taken: actions,
                            self.lr: lr})
        return loss
