# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Wrapper for creating the ant environment using Gymnasium's MujocoEnv."""

import os
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
  FILE = "ant.xml"

  def __init__(self, file_path=None, expose_all_qpos=True,
               expose_body_coms=None, expose_body_comvels=None):
    self._expose_all_qpos = expose_all_qpos
    self._expose_body_coms = expose_body_coms
    self._expose_body_comvels = expose_body_comvels
    self._body_com_indices = {}
    self._body_comvel_indices = {}

    # Resolve default model path if none provided
    if file_path is None:
      file_path = os.path.join(os.path.dirname(__file__), "assets", self.FILE)

    # Gymnasium's MujocoEnv requires an observation_space argument; we
    # provide None to let the env infer it from _get_obs during setup.
    mujoco_env.MujocoEnv.__init__(self, file_path, 5, observation_space=None)
    utils.EzPickle.__init__(self)

  @property
  def physics(self):
    return self.model

  def _step(self, a):
    # Backwards-compat: alias to step for old callers
    return self.step(a)

  def step(self, a):
    xposbefore = self.get_body_com("torso")[0]
    self.do_simulation(a, self.frame_skip)
    xposafter = self.get_body_com("torso")[0]
    forward_reward = (xposafter - xposbefore) / self.dt
    ctrl_cost = .5 * np.square(a).sum()
    survive_reward = 1.0
    reward = forward_reward - ctrl_cost + survive_reward
    state = self.state_vector()
    done = False
    ob = self._get_obs()
    return ob, reward, done, dict(
        reward_forward=forward_reward,
        reward_ctrl=-ctrl_cost,
        reward_survive=survive_reward)

  def _get_obs(self):
    # No cfrc observation
    if self._expose_all_qpos:
      obs = np.concatenate([
          self.data.qpos.flat[:15],  # Ensures only ant obs.
          self.data.qvel.flat[:14],
      ])
    else:
      obs = np.concatenate([
          self.data.qpos.flat[2:15],
          self.data.qvel.flat[:14],
      ])

    if self._expose_body_coms is not None:
      for name in self._expose_body_coms:
        com = self.get_body_com(name)
        if name not in self._body_com_indices:
          indices = range(len(obs), len(obs) + len(com))
          self._body_com_indices[name] = indices
        obs = np.concatenate([obs, com])

    if self._expose_body_comvels is not None:
      for name in self._expose_body_comvels:
        # Gymnasium MujocoEnv may not expose get_body_comvel; fall back gracefully
        comvel_fn = getattr(self, "get_body_comvel", None)
        if comvel_fn is None:
          continue
        comvel = comvel_fn(name)
        if name not in self._body_comvel_indices:
          indices = range(len(obs), len(obs) + len(comvel))
          self._body_comvel_indices[name] = indices
        obs = np.concatenate([obs, comvel])
    return obs

  def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(
        size=self.model.nq, low=-.1, high=.1)
    # np_random is a numpy Generator in Gymnasium; use standard_normal
    qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1

    # Set everything other than ant to original position and 0 velocity.
    qpos[15:] = self.init_qpos[15:]
    qvel[14:] = 0.
    self.set_state(qpos, qvel)
    return self._get_obs()

  def viewer_setup(self):
    viewer = getattr(self, "viewer", None)
    if viewer is not None and hasattr(viewer, "cam"):
      viewer.cam.trackbodyid = -1
      viewer.cam.distance = 50
      viewer.cam.elevation = -90