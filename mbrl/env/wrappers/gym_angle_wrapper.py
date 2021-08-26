"""Reward wrapper for the Ant-v3 environment that adjusts the forward axis to
   run on an angle.
   To activate it, put in the name the substring 'angle_xyz', with xyz beginning
   the digits of the angle in degrees (potentially with zeros in the start.)"""

from gym import Wrapper
import numpy as np


class AngleWrapper(Wrapper):
    PREFIX = '-angle_'

    def __init__(self, env, angle_in_radians: float):
        super(AngleWrapper, self).__init__(env)
        self._angle_in_radians = angle_in_radians
        self._angle_unit_vector = np.array(
            [np.cos(angle_in_radians),
             np.sin(angle_in_radians)])
        self._i = 1

    def step(self, action):
        observation, original_reward, done, info = self.env.step(action)
        x_velocity = info['x_velocity']
        y_velocity = info['y_velocity']
        velocity_vector = np.array([x_velocity, y_velocity])
        # Get velocity at the angle
        angle_velocity = velocity_vector.dot(self._angle_unit_vector)
        reward = (angle_velocity + info['reward_survive'] +
                  info['reward_ctrl'] + info['reward_contact'])
        #print('[gym_angle_wrapper] reward: ', reward)
        #print('[gym_angle_wrapper] info: ', info)
        if self._i % 100 == 0:
            print('[gym_angle_wrapper] angle_velocity: ', angle_velocity)
            print('[gym_angle_wrapper] reward_survive: ',
                  info['reward_survive'])
            print('[gym_angle_wrapper] reward_ctrl: ', info['reward_ctrl'])
            self._i = 1
            print('[gym_angle_wrapper] reward_contact: ',
                  info['reward_contact'])
        self._i += 1
        return observation, reward, done, info
