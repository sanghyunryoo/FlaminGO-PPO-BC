#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .ppo import PPO
from .ppo_bc import PPO_BC
from .srmppo import SRMPPO
from .sac import SAC
from .tqc import TQC
from .taco import TACO

__all__ = ["PPO", "PPO_BC", "SRMPPO", "SAC", "TQC", "TACO"]
