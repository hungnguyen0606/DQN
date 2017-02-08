import argparse
import numpy as np
import gym
from gym import wrappers
from problem import Problem
from agent import Agent, ModelParam
import tensorflow as tf

parser = argparse.ArgumentParser(description='MountainCar-v0')
parser.add_argument('--n_eps', metavar='episode', type=int, nargs=1, default=[100],
                    help='Total number of episodes used to train.')
parser.add_argument('--gamma', type=float, nargs=1, default=[0.99], help='Discount factor of Q-learning')
parser.add_argument('--lr', type=float, nargs=1, default=[0.1], help='Learning rate for gradient update')
parser.add_argument('--lr_decay', metavar='decay', type=float, nargs=1, default=[0.99],
                    help='Decay ratio for learning rate')
parser.add_argument('--lr_decay_step', metavar='decay', type=int, nargs=1, default=[5000],
                    help='Decay ratio for learning rate')

parser.add_argument('--ftest', type=int, nargs=1, default=[1],
                    help='0: if you don\'t want to run a final test.\n1 (default): otherwise')
parser.add_argument('--freeze_time', metavar='freeze interval', type=int, nargs=1, default=[1],
                    help='Number of iteration used for frequent test. If you don\' want to use frequent test, set this value to -1 (default value)')

parser.add_argument('--eps', metavar='epsilon_greedy', type=float, nargs=1, default=[0.1], help='Exploration rate.')
parser.add_argument('--eps_decay', metavar='epsilon_decay', type=float, nargs=1, default=[0.9],
                    help='Decay value for epsilon.')
parser.add_argument('--eps_decay_step', metavar='epsilon_greedy', type=int, nargs=1, default=[5000],
                    help='# episodes before update epsilon.')

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


# --------------------------------------------------------------------------------


class MountainProblem(Problem):
    def __init__(self, save_path):
        self.env = gym.make('MountainCar-v0')
        self.env = wrappers.Monitor(self.env, save_path, force=True)
        self.low = np.array(self.env.observation_space.low)
        self.high = np.array(self.env.observation_space.high)
        self.range = (self.high - self.low)

    def get_state_size(self):
        return 2

    def get_num_actions(self):
        return 3

    def sample_action(self):
        return self.env.action_space.sample()

    def sample_observation(self):
        state = self.env.observation_space.sample()
        state = self.state_normalize(state)
        return state

    def reset_environment(self):
        state = self.env.reset()
        state = self.state_normalize(state)
        return state

    def state_normalize(self, state):
        state = np.array(state)
        return (state - self.low) / self.range

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        next_state = self.state_normalize(next_state)
        return next_state, reward, done

    def render(self):
        self.env.render()


class MyAgent(Agent):
    def __init__(self, environment, model_setting):
        super().__init__(environment, model_setting)
        self.summary_loss = tf.placeholder(tf.float32)
        self.summary_total_reward = tf.placeholder(tf.float32)
        tf.summary.scalar("loss", tf.reduce_mean(self.summary_loss))
        tf.summary.scalar("loss", tf.reduce_sum(self.summary_total_reward))
        self.merge_summary = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(os.path.join(self.setting.save_path, 'models', ''), self.sess.graph)
        print("using my agent")

    def update_target_by_source(self, local_step, global_step=0):
        if local_step % self.setting.freeze_time == 0:
            print("updating target network...")
            self.sess.run(self.update_ops)

    def train(self, max_episode):
        self.init_memory()

        for episode in range(max_episode):
            # last = self.source_net.Q_Value(self.sess, states)
            lreward, lloss = self.run_episode(episode)

            print("Episode %d\n\tLoss %f - Total Reward %f\n\tLearning rate %f - Epsilon %f\n/------------------------/"
                  % (episode, np.mean(lloss), np.sum(lreward), self.get_lr(episode), self.get_eps(episode)))

            self.update_target_by_source(episode)

            summary = self.sess.run(self.merge_summary, {self.summary_loss: lloss, self.summary_total_reward: lreward})
            self.train_writer.add_summary(summary, self.sess.run(self.global_step))
            if episode % self.setting.save_time == 0 or episode == max_episode - 1:
                self.saver.save(self.sess, os.path.join(self.setting.save_path, 'models/', ''),
                                global_step=episode)
            self.sess.run(self.global_step.assign(tf.add(self.global_step, 1)))
        pass


if __name__ == '__main__':

    setting = ModelParam(args.lr[0], args.lr_decay[0], args.lr_decay_step[0],
                         args.eps[0], args.eps_decay[0], args.eps_decay_step[0],
                         args.gamma[0], args.freeze_time[0], args.load_path[0], args.save_path[0], args.stime[0])
    import os

    if not os.path.exists(setting.save_path):
        os.mkdir(setting.save_path)
        os.mkdir(os.path.join(setting.save_path, 'models'))

    prob = MountainProblem(setting.save_path)
    my_agent = MyAgent(prob, setting)
    my_agent.train(args.n_eps[0])
