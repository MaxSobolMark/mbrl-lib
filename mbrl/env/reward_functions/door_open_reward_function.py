from .base_reward_function import BaseRewardFunction
import tensorflow as tf


class DoorOpenRewardFunction(BaseRewardFunction):
    OBS_DIM = 39

    def __call__(self, raw_observations: tf.Tensor, actions: tf.Tensor):
        
        TODO: do drawer, faucet opening and closing rewards func.
        To get the tcp thing, use the hand position instead.
        
