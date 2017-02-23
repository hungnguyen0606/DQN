import gym
import tensorflow as tf
import numpy as np

from tensorflow.contrib import layers
from PolicyGradient.problem import Problem
from PolicyGradient.memory import replay_memory

num_actions = 6
state_size = 128
action_list = list(range(6))


class PolicyPong(Problem):
    def __init__(self):
        pass



class Actor():
    def __init__(self, scope):
        with tf.variable_scope(scope):
            self.build_network()

    def build_network(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self.inputs = tf.placeholder(tf.float32, [None, state_size], 'input')
        self.hidden1 = layers.fully_connected(self.inputs, 256)
        self.hidden2 = layers.fully_connected(self.hidden1, 128)
        self.policy = tf.nn.softmax(self.hidden2)
        self.q_value = layers.fully_connected(self.hidden2, activation_fn=None)

        self.value = tf.placeholder(tf.float32, name='episode_return')
        self.actions = tf.one_hot(tf.placeholder(tf.uint8, name='action_indices'), name='action_mask')
        self.policy_loss = tf.reduce_mean(tf.log(tf.reduce_sum(self.policy*self.actions, axis=1))*self.value, name='policy_loss')

        # use gradient ascent to maximize the expected return
        self.train_ops = tf.train.AdamOptimizer(self.lr).minimize(-self.policy_loss)

    def train(self, sess, lr, inputs, actions, values):
        sess.run(self.train_ops, {self.inputs: inputs, self.actions: actions, self.value: values})

    def predict(self, sess, state):
        states = [state]
        policy = sess.run(self.policy, {self.inputs: states})[0]
        return np.random.choice(action_list)


class agent():
    def __init__(self, prob, setting):
        self.sess = tf.Session()
        self.actor = Actor('MyActor')
        self.prob = prob
        self.setting = setting
        self.average_reward = tf.placeholder(tf.float32)

    def generate_batch(self, batch_size=10):
        lstate = []
        lreward = []
        laction = []
        ldone = []
        for _ in range(batch_size):
            state = self.prob.reset()
            while True:
                action = self.actor.predict(self.sess, state)
                next_state, reward, done = self.prob.step(action)
                lstate.append(state)
                laction.append(action)
                lreward.append(reward)
                ldone.append(done)
                if done:
                    break
        average_reward = np.sum(lreward)*1.0/batch_size
        # calculate return for each episode
        for i in range(len(lstate)-2, -1, -1):
            if (not ldone[i]) or (lreward[i] < 1.0):
                lreward[i] += self.setting.gamma*lreward[i+1]

        return lstate, laction, lreward, average_reward

    def get_lr(self):
        return 0.1
        pass

    def train(self, max_step = 5000, batch_size=10):
        for _ in range(max_step):
            b_inputs, b_actions, b_values, average_reward = self.generate_batch(batch_size)
            self.actor.train(self.sess, self.get_lr(), b_inputs, b_actions, b_values)




        pass

if __name__ == '__main__':
    sess = tf.Session()
    input = tf.placeholder(tf.float32)
    print(sess.run(tf.nn.softmax(input), {input: [[1,1],[2,2]]})[0])

