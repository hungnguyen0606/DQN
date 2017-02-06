import gym
from gym import wrappers

class Problem:
    def __init__(self):
        assert 1 == 2, 'Override init'

    def get_state_size(self):
        """

        :return: # of dimensions of state
        """
        assert 1 == 2, 'Need to implement get_state_size'

    def get_num_actions(self):
        """

        :return: number of actions
        """
        assert 1 == 2, 'Need to implement get_num_actions'

    def sample_action(self):
        """

        :return: a random action
        """
        assert 1 == 2, 'Need to implement sample_action'

    def reset_environment(self):
        """

        :return: return the first state
        """
        assert 1 == 2, 'Need to implement reset_environment'

    def step(self, action):
        """
        Take this action and return the observation = (next_state, reward, done)
        :param action: take this action
        :return: tuple (next_state, reward, done)
        """
        assert 1 == 2, 'Need to implement step'


class MountainProblem(Problem):
    def __init__(self, save_path):
        self.env = gym.make('MountainCar-v0')
        self.env = wrappers.Monitor(self.env, save_path, force=True)

    def get_state_size(self):
        return 2

    def get_num_actions(self):
        return 3

    def sample_action(self):
        return self.env.action_space.sample()

    def reset_environment(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done
