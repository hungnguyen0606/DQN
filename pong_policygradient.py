import gym
import tensorflow as tf
import numpy as np
import argparse

from gym.wrappers import Monitor
from tensorflow.contrib import layers
from PolicyGradient.problem import Problem
from PolicyGradient.memory import replay_memory
from PolicyGradient.agent import ModelParam
import time
num_actions = 6
state_size = 128
action_list = list(range(6))


parser = argparse.ArgumentParser(description='MountainCar-v0')
parser.add_argument('--n_eps', metavar='episode', type=int, nargs=1, default=[100],
                    help='Total number of episodes used to train.')
parser.add_argument('--batch_size', metavar='bsize', type=int, nargs=1, default=[10],
                    help='Total number of episodes used to train.')
parser.add_argument('--gamma', type=float, nargs=1, default=[0.99], help='Discount factor of Q-learning')
parser.add_argument('--lr', type=float, nargs=1, default=[0.1], help='Learning rate for gradient update')
parser.add_argument('--lr_decay', metavar='decay', type=float, nargs=1, default=[0],
                    help='Decay ratio for learning rate')
# parser.add_argument('--lr_decay_step', metavar='decay', type=int, nargs=1, default=[5000],
#                     help='Decay ratio for learning rate')

parser.add_argument('--ftest', type=int, nargs=1, default=[1],
                    help='0: if you don\'t want to run a final test.\n1 (default): otherwise')
parser.add_argument('--freeze_time', metavar='freeze interval', type=int, nargs=1, default=[1],
                    help='Number of iteration used for frequent test. If you don\' want to use frequent test, set this value to -1 (default value)')

parser.add_argument('--eps', metavar='epsilon_greedy', type=float, nargs=1, default=[0.1], help='Exploration rate.')
parser.add_argument('--eps_decay', metavar='epsilon_decay', type=float, nargs=1, default=[0],
                    help='Decay value for epsilon.')
# parser.add_argument('--eps_decay_step', metavar='epsilon_greedy', type=int, nargs=1, default=[5000],
#                     help='# episodes before update epsilon.')

parser.add_argument('--stime', metavar='save_time', type=int, nargs=1, default=[5000],
                    help='Save the model after "stime" episodes.')

parser.add_argument('--save_path', type=str, nargs=1, default=['./MountainCar-v0'],
                    help='folder to save the result video & models.')
parser.add_argument('--load_path', type=str, nargs=1, default=['./MountainCar-v0'], help='folder to load old models.')

parser.add_argument('--force-save', dest='force', action='store_true',
                    help='Force the program to overwrite old results.')
parser.set_defaults(force=False)
parser.add_argument('--test', dest='test_model', action='store_true', help='Test model')
parser.set_defaults(test_model=False)
args = parser.parse_args()


class PolicyPong(Problem):
    def __init__(self, save_path, is_test):
        self.env = gym.make('Pong-ram-v0')
        if is_test:
            self.env = Monitor(self.env, save_path)

    def get_state_size(self):
        return 128

    def get_num_actions(self):
        return 6

    def sample_action(self):
        return self.env.action_space.sample()

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        return self.state_normalize(state), reward, done

    def reset_environment(self):
        state = self.env.reset()
        return self.state_normalize(state)

    def render(self):
        self.env.render()

    def state_normalize(self, state):
        return state/255.0

class Actor():
    def __init__(self, scope, state_size, num_action):
        self.state_size = state_size
        self.num_action = num_action
        with tf.variable_scope(scope):
            self.build_network()

    def build_network(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self.inputs = tf.placeholder(tf.float32, [None, self.state_size], 'input')
        self.hidden1 = layers.fully_connected(self.inputs, 256)
        self.hidden2 = layers.fully_connected(self.hidden1, 128)
        self.log_prob = layers.fully_connected(self.hidden2, self.num_action)
        self.policy = tf.nn.softmax(self.log_prob)


        self.value = tf.placeholder(tf.float32, name='episode_return')
        self.action_indices = tf.placeholder(tf.int32, name='action_indices')
        self.actions = tf.one_hot(self.action_indices, self.num_action, name='action_mask')
        self.policy_loss = tf.reduce_mean(tf.log(tf.reduce_sum(self.policy*self.actions, axis=1, name='sum'), name='log')*self.value, name='policy_loss')

        # use gradient ascent to maximize the expected return
        self.train_ops = tf.train.RMSPropOptimizer(self.lr).minimize(-self.policy_loss)

    def train(self, sess, lr, inputs, actions, values):
        sess.run(self.train_ops, {self.inputs: inputs, self.action_indices: actions, self.value: values, self.lr: lr})

    def predict(self, sess, state):
        states = [state]
        policy = sess.run(self.policy, {self.inputs: states})[0]
        return np.random.choice(action_list)


class PolicyAgent():
    def __init__(self, prob, setting):
        self.sess = tf.Session()
        self.prob = prob
        self.actor = Actor('MyActor', self.prob.get_state_size(), self.prob.get_num_actions())
        self.setting = setting
        self.average_reward = tf.placeholder(tf.float32)
        tf.summary.scalar('average_reward', self.average_reward, collections=['reward'])
        self.merge_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(os.path.join(self.setting.save_path, 'models', ''), self.sess.graph)

        self.global_step = tf.Variable(0, name='global_step')
        self.saver = tf.train.Saver(max_to_keep=100000)

        self.sess.run(tf.global_variables_initializer())

        checkpoint = tf.train.get_checkpoint_state(os.path.join(self.setting.load_path, 'models', ''))

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded old agent at: ", checkpoint.model_checkpoint_path)
        else:
            print("Unable to find the old agent.")

    def generate_batch(self, batch_size):
        lstate = []
        lreward = []
        laction = []
        ldone = []
        for _ in range(batch_size):
            state = self.prob.reset_environment()
            while True:
                action = self.actor.predict(self.sess, state)
                # self.prob.render()
                next_state, reward, done = self.prob.step(action)
                lstate.append(state)
                laction.append(action)
                lreward.append(reward)
                ldone.append(done)
                state = next_state
                if done:
                    break
        average_reward = np.sum(lreward)*1.0/batch_size
        # calculate return for each episode
        for i in range(len(lstate)-2, -1, -1):
            if ldone[i] or abs(lreward[i]) > 0:
                pass
            else:
                lreward[i] += self.setting.gamma * lreward[i + 1]

        return lstate, laction, lreward, average_reward

    def get_lr(self):
        return self.setting.lr

    def run_for_fun(self):
        print("Playing for fun")
        state = self.prob.reset_environment()
        while True:
            action = self.actor.predict(self.sess, state)
            _, _, done = self.prob.step(action)
            self.prob.render()
            if done:
                break

    def train(self, max_step=5000, batch_size=10):
        gb = self.sess.run(self.global_step)
        for _ in range(max_step):
            start = time.time()
            b_inputs, b_actions, b_values, average_reward = self.generate_batch(batch_size)
            #print(b_values)

            self.actor.train(self.sess, self.get_lr(), b_inputs, b_actions, b_values)
            print("Step {} - Elapsed time: {}s\nLearning rate {} - Batch size {}\nAverage reward {}".
                  format(_, time.time()-start, self.get_lr(), batch_size, average_reward))
            summary = self.sess.run(tf.summary.merge_all(key='reward'), {self.average_reward: average_reward})
            self.train_writer.add_summary(summary, gb)

            if gb % self.setting.save_time == 0:
                self.saver.save(self.sess, os.path.join(self.setting.save_path, 'models/', ''), gb)

            gb += 1
            self.sess.run(self.global_step.assign(gb))

            if _ % 100 == 0:
                self.run_for_fun()


if __name__ == '__main__':
    setting = ModelParam(args.lr[0], args.lr_decay[0],
                         args.eps[0], args.eps_decay[0],
                         args.gamma[0], args.freeze_time[0], args.load_path[0], args.save_path[0], args.stime[0])
    import os
    if not os.path.exists(setting.save_path):
        os.mkdir(setting.save_path)
        os.mkdir(os.path.join(setting.save_path, 'models'))

    prob = PolicyPong(setting.save_path, args.test_model)
    agent = PolicyAgent(prob, setting)
    agent.train(args.n_eps[0], args.batch_size[0])

