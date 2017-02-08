

class Problem:
    def __init__(self):
        raise NotImplementedError

    def get_state_size(self):
        """

        :return: # of dimensions of state
        """
        raise NotImplementedError

    def get_num_actions(self):
        """

        :return: number of actions
        """
        raise NotImplementedError

    def sample_action(self):
        """

        :return: a random action
        """
        raise NotImplementedError

    def reset_environment(self):
        """

        :return: return the first state
        """
        raise NotImplementedError

    def step(self, action):
        """
        Take this action and return the observation = (next_state, reward, done)
        :param action: take this action
        :return: tuple (next_state, reward, done)
        """
        raise NotImplementedError

