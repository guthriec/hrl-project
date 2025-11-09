# Ant Environments for RL

Taken almost entirely from the [Tensorflow Models](https://github.com/tensorflow/models/tree/master/research/efficient-hrl/environments) repository, which is itself inspired by the work done by [RlLab](https://github.com/rll/rllab/blob/master/rllab/envs/mujoco/).

Notes:

- The environments now use Gymnasium's Mujoco backend under the hood (gymnasium>=0.29).
- Public methods keep the legacy Gym API for compatibility with the rest of the codebase/tests:
  - reset() returns only the observation (not (obs, info)).
  - step(action) returns (obs, reward, done, info) where done combines termination/truncation.
