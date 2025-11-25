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

from .ant_maze_env import AntMazeEnv
from .point_maze_env import PointMazeEnv
from .simple_car_maze import simple_car_maze_training, simple_car_maze_eval


def create_maze_env(env_name=None, render_mode=None):
    env_name = env_name or ""
    maze_id = None
    easy = bool(env_name and env_name.endswith("Easy"))
    kwargs = {}
    kwargs.update(dict(
        render_mode=render_mode
    ))

    # Determine if using Point or Ant
    use_point = env_name.startswith("Point")

    if env_name == "SimpleCarMaze":
        return simple_car_maze_training(render_mode=render_mode)

    if env_name.startswith("AntMaze") or env_name.startswith("PointMaze"):
        maze_id = "Maze"
        if easy:
            kwargs.update(
                dict(
                    maze_size_scaling=8,  # smaller cells -> shorter paths
                    maze_height=0.3,  # thinner platforms
                    force_flat=True,  # ignore elevation
                    disable_walls=True,  # open field
                    disable_movable_blocks=True,  # no obstacles
                    # max_episode_steps=200,  # shorter episodes
                )
            )
    elif env_name.startswith("AntPush") or env_name.startswith("PointPush"):
        maze_id = "Push"
        if easy:
            kwargs.update(
                dict(
                    maze_size_scaling=6,
                    maze_height=0.3,
                    max_episode_steps=200,
                )
            )
    elif env_name.startswith("AntFall") or env_name.startswith("PointFall"):
        maze_id = "Fall"
        if easy:
            kwargs.update(
                dict(
                    maze_size_scaling=6,
                    maze_height=0.3,
                    max_episode_steps=300,
                )
            )
    else:
        raise ValueError("Unknown maze environment %s" % env_name)

    if use_point:
        return PointMazeEnv(maze_id=maze_id, **kwargs)
    else:
        return AntMazeEnv(maze_id=maze_id, **kwargs)


# dirty hack.
def create_eval_maze_env(env_name=None, render_mode=None):
    if env_name == "SimpleCarMaze":
        return simple_car_maze_eval(render_mode=render_mode)
    return create_maze_env(env_name=env_name,render_mode=render_mode)