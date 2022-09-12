import gym
import numpy as np
from zhikong.tasks import Shaping, AircombatTask
from zhikong.simulation import Simulation
from zhikong.aircraft import Aircraft, f16
from typing import Type, Tuple, Dict
from zhikong.properties import Property, BoundedProperty


class JsbSimEnv(gym.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    An JsbSimEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.

    ATTRIBUTION: this class implements the OpenAI Gym Env API. Method
    docstrings have been adapted or copied from the OpenAI Gym source code.
    """
    JSBSIM_DT_HZ: int = 60  # JSBSim integration frequency
    metadata = {'render.modes': ['human', 'flightgear']}

    def __init__(self, task_type: Type[AircombatTask], aircraft: Aircraft = f16,
                 agent_interaction_freq: int = 5, shaping: Shaping=Shaping.STANDARD):
        """
        Constructor. Inits some internal state, but JsbSimEnv.reset() must be
        called first before interacting with environment.

        :param task_type: the Task subclass for the task agent is to perform
        :param aircraft: the JSBSim aircraft to be used
        :param agent_interaction_freq: int, how many times per second the agent
            should interact with environment.
        :param shaping: a HeadingControlTask.Shaping enum, what type of agent_reward
            shaping to use (see HeadingControlTask for options)
        """
        if agent_interaction_freq > self.JSBSIM_DT_HZ:
            raise ValueError('agent interaction frequency must be less than '
                             'or equal to JSBSim integration frequency of '
                             f'{self.JSBSIM_DT_HZ} Hz.')
        self.sim: Simulation = None
        self.sim2: Simulation = None
        self.sim_steps_per_agent_step: int = self.JSBSIM_DT_HZ // agent_interaction_freq
        self.aircraft = aircraft
        self.task = task_type(shaping, agent_interaction_freq, aircraft)
        # set Space objects
        self.observation_space: gym.spaces.Box = self.task.get_state_space()
        self.action_space: gym.spaces.Box = self.task.get_action_space()
        # set visualisation objects
        pass



    def step(self, action: np.ndarray):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        :param action: the agent's action, with same length as action variables.
        :return:
            state: agent's observation of the current environment
            reward: amount of reward returned after previous action
            done: whether the episode has ended, in which case further step() calls are undefined
            info: auxiliary information, e.g. full reward shaping data
        """
        # if not (action.shape == self.action_space.shape):
        #     raise ValueError('mismatch between action and action space size')

        state, state1, reward, done, info = self.task.task_step(self.sim, self.sim2, action, self.sim_steps_per_agent_step)
        return np.array([state,state1]).flatten(), np.array(state), np.array(state1), reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        :return: array, the initial observation of the space.
        """
        init_conditions = self.task.get_initial_conditions()
        init_conditions_2 = self.task.get_initial_conditions_2()
        send_init_conditions_1 = {}
        send_init_conditions_2 = {}

        for key in init_conditions.keys():
            if isinstance(key, (BoundedProperty, Property)):
                send_init_conditions_1[key.name] = init_conditions[key]
            else:
                send_init_conditions_1[key.name] = init_conditions[key]

        for key in init_conditions_2.keys():
            if isinstance(key, (BoundedProperty, Property)):
                send_init_conditions_2[key.name] = init_conditions_2[key]
            else:
                send_init_conditions_2[key.name] = init_conditions_2[key]

        if self.sim:

            init_info={'flag':{},'red':{},'blue':{}}
            init_info['flag']={'init':{'render':0}}
            init_info['red']={'red_0':send_init_conditions_1}
            init_info['blue']={'blue_0':send_init_conditions_2}
            msg_recieve=self.task.communication.reset(1,1,init_info)

        else:
            self.sim = self._init_new_sim(self.aircraft)
            self.sim2 = self._init_new_sim(self.aircraft)
            init_info = {'flag': {}, 'red': {}, 'blue': {}}
            init_info['flag'] = {'reset': {}}
            init_info['red'] = {'red_0': send_init_conditions_1}
            init_info['blue'] = {'blue_0': send_init_conditions_2}
            msg_recieve=self.task.communication.reset(1,1,init_info)

        self.task._accept_from_socket(msg_recieve,self.sim, self.sim2)
        global_state, state1, state2 = self.task.observe_first_state(self.sim,self.sim2)


        return global_state, state1, state2

    def _init_new_sim(self,  aircraft):
        return Simulation(aircraft=aircraft)


    def close(self):
        """ Cleans up this environment's objects

        Environments automatically close() when garbage collected or when the
        program exits.
        """
        if self.sim:
            self.sim.close()

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        gym.logger.warn("Could not seed environment %s", self)
        return

    def render(self):
        self.task.communication.RENDER=1

