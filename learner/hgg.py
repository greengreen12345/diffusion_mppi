from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env

import copy
import numpy as np
# from envs import make_env
from envs.utils import get_goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int
import torch, hydra

# from scripts.reactive_tamp import REACTIVE_TAMP
# from src.m3p2i_aip.config.config_store import ExampleConfig
# import learner.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from src.m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
# from sim1 import run_sim1

import random
import time

import os
import json

from torch.utils.tensorboard import SummaryWriter

class TrajectoryPool:
    def __init__(self, args, pool_length):
        self.args = args
        self.length = pool_length

        self.pool = []
        self.pool_init_state = []
        self.counter = 0

    def insert(self, trajectory, init_state):
        if self.counter < self.length:
            self.pool.append(trajectory.copy())
            self.pool_init_state.append(init_state.copy())
        else:
            self.pool[self.counter % self.length] = trajectory.copy()
            self.pool_init_state[self.counter % self.length] = init_state.copy()
        self.counter += 1

    def pad(self):
        if self.counter >= self.length:
            return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
        pool = copy.deepcopy(self.pool)
        pool_init_state = copy.deepcopy(self.pool_init_state)
        while len(pool) < self.length:
            pool += copy.deepcopy(self.pool)
            pool_init_state += copy.deepcopy(self.pool_init_state)
        return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])

    def clear(self):
        """Clear all trajectories and states"""
        self.pool.clear()
        self.pool_init_state.clear()
        self.counter = 0

class HGGLearner:
    def __init__(self, args):
        self.args = args
        self.goal_distance = get_goal_distance(args)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)
        self.reactive_tamp = None

        # Track training status
        self.training_state = {"total_episodes": 0}
        self.env = None
        self.env_test = None
        self.env_type = 'simpler'
        self.planner = None
        self.agent = None
        self.buffer = None

        # Historical trajectory logging
        self.all_trajectories = []
        self.all_trajectories_capacity = 100
        self.all_episode_trajectories = []

        self.episodes = args.episodes
        self.cycles = 0

    def learn(self, args, env, env_test, agent, bc_diffusion_agent, ql_diffusion_agent, buffer, planner, training_state=None, sampler=None):
        """Main training loop

        Args:
            args: Configuration arguments
            env: Training environment
            env_test: Test environment
            agent: The agent to be trained
            buffer: Replay buffer
            planner: MPPI planner
            training_state: Optional dictionary to track training state

        """
        self.initial_goals = []
        self.desired_goals = []
        self.explore_goals = []
        self.achieved_trajectories = []
        self.success_trajectories = []
        self.achieved_rewards = []
        self.episode_return = 0
        self.episode_trajectory = []

        # Set up environment and components
        self.env = env
        self.env_test = env_test
        self.agent = agent
        self.bc_diffusion_agent = bc_diffusion_agent
        self.ql_diffusion_agent = ql_diffusion_agent
        self.buffer = buffer
        self.planner = planner

        # Reset environment and get initial state
        obs = self.env.reset()
        self.prev_position = obs['achieved_goal'].copy()
        goal_a = obs['achieved_goal'].copy()
        goal_d = obs['desired_goal'].copy()
        self.initial_goals.append(goal_a.copy())
        self.desired_goals.append(goal_d.copy())

        self.current = Trajectory(obs)

        if self.buffer.counter > 75:
            self.episodes = 50
        else:
            self.episodes = 75

        for episode in range(self.episodes):
            if self.buffer.counter > 75:
                is_warmup = False
            else:
                is_warmup = True

            if is_warmup:
                cube_state = self.env.get_actor_link_by_name("cubeA", "box")
                base_goal = self.env.get_actor_link_by_name("cubeA", "box")[0, :3].clone().cpu().numpy()
                # Add uniform noise in the range [-0.03, 0.03] to each dimension
                noise = np.random.uniform(-0.07, 0.07, size=3)
                subgoal = base_goal + noise
                subgoal = np.array(subgoal).squeeze()
                subgoal[2] = 1.1

                print("cubeA position", cube_state[0, :3].clone())
                print(f"Warmup phase: Using random subgoal {subgoal}")

            else:
                # generate subgoal by diffusion model
                obs = self.env._get_obs()
                subgoal = self.generate_subgoal(obs, pick_and_place=False)
                subgoal[2] = 1.1

            if is_warmup:
                timesteps = 25
            else:
                timesteps = 40
            current, episode_reward, trajectory, final_distance, success_reached = self.rollout(timesteps, subgoal=subgoal)
            self.explore_goals.append(subgoal)
            self.episode_trajectory.append(trajectory)

            if success_reached:
                self.saver = True
                break

        # Add trajectory to buffer
        self.buffer.store_trajectory(current)

        if self.buffer.counter % 7 == 0:
            self.update_network()

        final_trajectory = np.concatenate(self.episode_trajectory)

        self.achieved_trajectories.append(final_trajectory)
        self.all_episode_trajectories.append(final_trajectory)

        return success_reached

    def update_network(self):

        # Instantiate the TensorBoard writer
        writer = SummaryWriter(log_dir='logs/ql_diffusion')

        for _ in range(3000):
            transitions = self.buffer.sample_batch(self.args.batch_size)

            if len(transitions["obs"]) == 0:
                print("Empty batch detected, skipping training step.")
                continue

            info = self.ql_diffusion_agent.train(transitions, iterations = 100, batch_size = 100, log_writer = writer)

    def generate_subgoal(self, obs, pick_and_place=False):

        if pick_and_place:
            full_observation = np.zeros(13)  # 13-dimensional observation
            full_observation[:7] = obs['observation']  # Position information
            state = np.concatenate((full_observation.reshape(1, -1), obs['desired_goal'].reshape(1, -1)), axis=1)

            optimized_subgoal = self.ql_diffusion_agent.sample_action(np.array(state))
            print("Subgoal generated by ql_diffusion model:", optimized_subgoal)

            return optimized_subgoal

        # QL_diffusion
        full_observation = np.zeros(13)  # 13-dimensional observation
        full_observation[:6] = obs['observation']  # Position information

        state = np.concatenate((full_observation.reshape(1, -1), obs['desired_goal'].reshape(1, -1)), axis=1)
        optimized_subgoal = self.ql_diffusion_agent.sample_action(np.array(state))
        obs_raw = np.array(obs['observation']).squeeze()  # shape (7,)
        object_pos = obs_raw[3:6]
        gripper_pos = obs_raw[0:3]
        print("Subgoal generated by ql_diffusion model:", optimized_subgoal)
        print("Position of cubeA:", object_pos)

        return optimized_subgoal

    def rollout(self, timesteps, subgoal=None):
        """Execute a subgoal-driven trajectory

        Args:
            timesteps: Maximum number of steps
            subgoal: Optional subgoal

        Returns:
            episode_experience: Trajectory experience
            episode_reward: Cumulative reward
        """
        gripper_value = None
        if subgoal is not None and len(subgoal) > 3:
            gripper_value = subgoal[3]
            subgoal = subgoal[:3]

        success_reached = False
        self.env.goal = subgoal
        # Get the current observation from the environment
        obs = self.env._get_obs()
        trajectory = [obs['achieved_goal'].copy()]  # Record position trajectory
        episode_reward = 0  # Cumulative reward

        # If a subgoal is provided, set it in the environment
        if subgoal is not None:
            self.env.subgoal = torch.tensor(subgoal, dtype=torch.float32)

        # Record initial position and target goal
        initial_position = obs['achieved_goal'].copy()
        desired_goal = obs['desired_goal'].copy()

        # Execute the specified number of steps
        for t in range(timesteps):
            # Get current state and goal
            achieved_goal = obs['achieved_goal'].copy()

            # Generate action using MPPI planner
            action_mppi = bytes_to_torch(
                self.planner.run_tamp(
                    torch_to_bytes(self.env._dof_state),
                    torch_to_bytes(self.env._root_state),
                    subgoal.tolist() if subgoal is not None else desired_goal.tolist())
            )

            # Execute the action
            obs, reward, done, info, distance, dis_subgoal = self.env.step(action_mppi)

            # 1. Reward based on goal distance
            current_pos = obs['achieved_goal'].copy()
            curr_distance = np.linalg.norm(current_pos - obs['desired_goal'].copy())

            # 2. Reward based on subgoal distance
            curr_subgoal_distance = np.linalg.norm(subgoal - obs['desired_goal'].copy())
            reward = curr_subgoal_distance * (-1)

            # 4. Success bonus
            if distance < 0.02:
                print(f"Goal reached, terminating at step {t}, distance {distance:.4f}")

                success_bonus = 0.1
                reward += success_bonus
                print(f"Success bonus added: {success_bonus}")
                success_reached = True

                # break

            episode_reward += reward

            if subgoal is not None and isinstance(subgoal, np.ndarray):
                subgoal = torch.tensor(subgoal, dtype=torch.float32)

            # Store current step in trajectory
            self.current.store_step(action_mppi, obs, reward, done, subgoal)
            trajectory.append(current_pos)

            if distance < 0.02:
                break

            if dis_subgoal < 0.005:
                print("----------------------Subgoal reached-----------------", subgoal)
                break

        # Create trajectory data dictionary
        trajectory_data = {
            'obs': self.current.ep['obs'],
            'path': np.array(trajectory),
            'reward': episode_reward,
            'success': False
        }

        # Compute final distance to target
        final_distance = np.linalg.norm(trajectory[-1] - desired_goal)

        # Store in all trajectory history, regardless of success
        self.all_trajectories.append(trajectory_data)
        # Limit history size
        if len(self.all_trajectories) > self.all_trajectories_capacity:
            self.all_trajectories.pop(0)

        return self.current, episode_reward, trajectory, final_distance, success_reached
