# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict
import hydra
import numpy as np
import omegaconf
from omegaconf import OmegaConf
import torch
import wandb

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.util.mujoco as mujoco_util


@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    wandb.init(project='fsrl-mbrl', config=OmegaConf.to_container(cfg))
    env, term_fn, reward_fn = mujoco_util.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "pets":
        return pets.train(env,
                          term_fn,
                          reward_fn,
                          cfg,
                          forward_postprocess_fn=env.forward_postprocess_fn)
    if cfg.algorithm.name == "mbpo":
        test_env, *_ = mujoco_util.make_env(cfg)
        return mbpo.train(env, test_env, term_fn, cfg)


if __name__ == "__main__":
    run()
