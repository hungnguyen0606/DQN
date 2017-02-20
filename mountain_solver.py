import argparse
import numpy as np
import gym
from gym import wrappers
from Basic_Qlearning.problem import Problem
from Basic_Qlearning.agent import Agent, ModelParam
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

parser = argparse.ArgumentParser(description='MountainCar-v0')
parser.add_argument('--n_eps', metavar='episode', type=int, nargs=1, default=[100],
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


# --------------------------------------------------------------------------------


class MountainProblem(Problem):
    def __init__(self, save_path, is_test=False):
        self.env = gym.make('MountainCar-v0')
        if is_test:
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
        tf.summary.scalar("loss", tf.reduce_mean(self.summary_loss), collections=['loss'])
        tf.summary.scalar("reward", tf.reduce_sum(self.summary_total_reward), collections=['reward'])
        self.img = tf.placeholder(dtype=tf.uint8, name='qimage')
        tf.summary.image("max_Qvalue", self.img, max_outputs=100, collections=['image'])
        self.merge_summary = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(os.path.join(self.setting.save_path, 'models', ''), self.sess.graph)
        print("using my agent")

    def update_target_by_source(self, global_step=0):
        if global_step % self.setting.freeze_time == 0:
            print("updating target network...")
            self.sess.run(self.update_ops)

    def get_lr(self, global_step):
        temp = super().get_lr(global_step)
        if temp > 0.0001:
            return  temp
        return 0.0001

    def get_eps(self, global_step):
        temp = super().get_eps(global_step)
        # print('eps ', temp)
        if temp > 0.0001:
            return temp
        return 0.0001

    def run_step(self, state, global_step=0):
        """
        Take action, add transition to memory
        :param state: current state of the agent
        :param episode: numerical order of episode - for calculate epsilon
        :return: next state, reward, terminal signal
        """
        # choose action using epsilon-greedy policy
        action = self.source_net.epsilon_predict(self.sess, state, self.get_eps(global_step))
        # take action & observe the environment
        next_state, reward, done = self.env.step(action)
        # add transition to replay memory
        # reward /= 200.0

        self.memory.add(state, action, reward, next_state, done)
        return next_state, reward, done

    def full_update(self, gb_step, **kwargs):
        self.env.render()

        if gb_step % self.setting.freeze_time == 0:
            self.saver.save(self.sess, os.path.join(self.setting.save_path, 'models/', ''),
                            global_step=gb_step)
        self.update_target_by_source(gb_step)

        if gb_step % 200 == 0:
            summary = self.sess.run(tf.summary.merge_all(key='loss'), {self.summary_loss: kwargs['loss']})
            self.train_writer.add_summary(summary, gb_step)


    def run_episode(self, episode, max_step = 15000):
        """

        :param episode: number of current episode
        :return: list of reward, list of loss
        """
        state = self.env.reset_environment()
        lreward = []
        lloss = []
        total_reward = 0
        msg = "\nLearning rate {} - Epsilon {}\nGlobal step {} - Episode {} - Step {}/{} - Loss {} - Current Reward {} "
        gb_step = self.sess.run(self.global_step)
        for _ in range(max_step):

            next_state, reward, done = self.run_step(state, gb_step)
            total_reward += reward
            state = next_state

            # prepare data, compute & apply gradient
            data = self.prepare_batch()

            target = self.get_target(data.lnext_state, data.lreward, data.lterminal)
            # print(self.source_net.train())
            _loss = self.source_net.train(self.sess, data.lstate, data.laction, target, self.get_lr(gb_step))

            self.full_update(gb_step, loss=_loss)
            if _ % 1000 == 0:
                print(msg.format(self.get_lr(gb_step), self.get_eps(gb_step), gb_step, episode, _, max_step, _loss, total_reward))
            gb_step += 1
            if done:
                break
        self.sess.run(self.global_step.assign(gb_step))
        return total_reward

    def train(self, max_episode):
        # print("Helllo train")
        self.init_memory()
        # print("end train")
        combine_img = []
        lreward = []
        for episode in range(max_episode):
            # last = self.source_net.Q_Value(self.sess, states)
            reward = self.run_episode(episode=episode, max_step=10000)
            summary = self.sess.run(tf.summary.merge_all(key='reward'), {self.summary_total_reward: reward})
            self.train_writer.add_summary(summary, self.sess.run(self.global_step))
            print("Episode Reward {}".format(reward))
            lreward.append(reward)
            # summary = self.sess.run(self.merge_summary, {self.img: self.visualize()})
            # summary = self.sess.run(tf.summary.merge_all(key='statistics'), { self.summary_loss: lloss, self.summary_total_reward: lreward})
            # self.train_writer.add_summary(summary, self.sess.run(self.global_step))

            # if episode % 10 == 0:
            #     print("Addingggggggggggggggg img")
            #     combine_img.append(self.visualize())
            #     summary = self.sess.run(tf.summary.merge_all(key='image'), {self.img:combine_img})
            #     self.train_writer.add_summary(summary, self.sess.run(self.global_step))


    def test(self, max_episode):
        for episode in range(max_episode):
            state = self.env.reset_environment()
            while True:
                action = self.source_net.predict(self.sess, state)
                state, reward, done = self.env.step(action)
                self.env.render()
                if done:
                    break

    def visualize(self):
        x = np.linspace(0, 1, 1000)
        y = np.linspace(0, 1, 1000)
        data = [[u, v] for u in x for v in y]
        z_ = np.max(self.source_net.Q_Value(self.sess, np.array(data)), axis = 1)
        z = np.zeros((len(x), len(y)))
        nx = len(x)
        ny = len(y)
        for i in range(len(x)):
            for j in range(len(y)):
                z[i, j] = z_[i*ny+j]#np.max(self.source_net.Q_Value(self.sess, np.array([[x[i], y[j]]])), axis = 1)
        x, y = np.meshgrid(x, y)
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.imshow(z, vmin=np.min(z), vmax=np.max(z), origin='lower', extent=[0, 1, 0, 1])
        # plt.scatter(x, y, c=z)
        plt.colorbar()
        plt.show()


        return data.astype(np.uint8)

    def show_weight(self):
        w1, b1, w2, b2, w3, b3 = self.sess.run([t for t in tf.trainable_variables() if t.name.startswith('SourceNet')])
        relu = np.vectorize(lambda x: max(0,x))
        data = np.array([[0.8, 0.3], [0.1, 0.6]])
        logit1 = relu(data.dot(w1)+b1)
        logit2 = relu(logit1.dot(w2)+b2)
        logit3 = logit2.dot(w3)+b3
        print(logit1)
        print(logit2)
        print(logit3)
        print(self.source_net.Q_Value(self.sess, np.array(data)))
        # for t in tf.trainable_variables():
        #     if t.name.startswith('SourceNet'):
        #         tmp = self.sess.run(t)
        #         print(t.name)
        #         print(tmp)

if __name__ == '__main__':
    setting = ModelParam(args.lr[0], args.lr_decay[0],
                         args.eps[0], args.eps_decay[0],
                         args.gamma[0], args.freeze_time[0], args.load_path[0], args.save_path[0], args.stime[0])
    import os

    if not os.path.exists(setting.save_path):
        os.mkdir(setting.save_path)
        os.mkdir(os.path.join(setting.save_path, 'models'))

    prob = MountainProblem(setting.save_path, args.test_model)
    my_agent = MyAgent(prob, setting)
    # if args.test_model:
    #     print("Testing model")
    #     my_agent.test(args.n_eps[0])
    # else:
    #     print("Training model")
    #     my_agent.train(args.n_eps[0])

    # my_agent.test(args.n_eps[0])
    my_agent.visualize()
    # my_agent.show_weight()
