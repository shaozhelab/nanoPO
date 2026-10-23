"""Record one episode of a trained agent as an mp4."""
import os

# MuJoCo draws frames with OpenGL; "egl" renders on the GPU without a window.
# Must be set before MuJoCo is first loaded.
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
from moviepy import ImageSequenceClip


def record_episode(agent, env_id, path, seed=0):
    """Run one full episode with the agent's policy, save it to `path`, and return the episode's score."""
    env = gym.make(env_id, render_mode="rgb_array")
    if isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    score = 0.0
    done = False
    while not done:
        action, _, _ = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        score += reward
        done = terminated or truncated
        frames.append(env.render())
    fps = env.metadata["render_fps"]
    env.close()
    ImageSequenceClip(frames, fps=fps).write_videofile(path, logger=None)
    return score
