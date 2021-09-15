from typing import Callable, Dict, List, Tuple
from .base_reward_function import BaseRewardFunction
from .drawer_close_reward_function import DrawerCloseRewardFunction
from .drawer_open_reward_function import DrawerOpenRewardFunction
# from .faucet_close_reward_function import FaucetCloseRewardFunction
# from .faucet_open_reward_function import FaucetOpenRewardFunction
# from .reacher_3d_reward_function import Reacher3DRewardFunction
from .half_cheetah_reward_function import HalfCheetahRewardFunction
from .half_cheetah_backwards_reward_function import (
    HalfCheetahBackwardsRewardFunction)
from .half_cheetah_jump_reward_function import HalfCheetahJumpRewardFunction
from .pusher_reward_function import PusherRewardFunction

from mbrl.env.reward_fns import reacher
# from .reach_reward_function import ReachRewardFunction
# import tensorflow as tf


class PrefixDict(dict):
    def __getitem__(self, arg):
        matches = []
        for key, value in dict.items(self):
            if arg.startswith(key):
                matches.append((key, value))
        if len(matches) == 0:
            raise ValueError('Key not found.')
        longest_match_key, longest_match_value = matches[0]
        for key, value in matches:
            if len(key) > len(longest_match_key):
                longest_match_key, longest_match_value = key, value
        return longest_match_value


# DOMAIN_TO_REWARD_FUNCTION = PrefixDict({
DOMAIN_TO_REWARD_FUNCTION = {  # PrefixDict({
    'drawer-close-v2':
    DrawerCloseRewardFunction,
    'drawer-close-v2-max':
    DrawerCloseRewardFunction,
    'drawer-open-v2':
    DrawerOpenRewardFunction,
    'drawer-open-v2-max':
    DrawerOpenRewardFunction,
    # 'faucet-close-v2':
    # FaucetCloseRewardFunction,
    # 'faucet-open-v2':
    # FaucetOpenRewardFunction,
    'pets_reacher':
    reacher,
    #Reacher3DRewardFunction,
    # 'MBRLReacher5Goals3D':
    # Reacher3DRewardFunction,
    # 'reach-v2':
    # ReachRewardFunction,
    'pets_halfcheetah':
    HalfCheetahRewardFunction,
    'pets_halfcheetah_backwards':
    HalfCheetahBackwardsRewardFunction,
    'pets_halfcheetah_jump':
    HalfCheetahJumpRewardFunction,
    'pets_pusher': PusherRewardFunction,
    }
# })
