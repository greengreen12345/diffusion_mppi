import numpy as np
import copy
from envs.utils import quaternion_to_euler_angle
import torch




def goal_concat(obs, goal):
    obs = np.squeeze(obs)
    goal = np.squeeze(goal)

    return np.concatenate([obs, goal], axis=0)


def goal_based_process(obs):
    return goal_concat(obs['observation'], obs['desired_goal'])


class Trajectory:
    def __init__(self, init_obs):
        self.ep = {
            'obs': [copy.deepcopy(init_obs)],
            'rews': [],
            'acts': [],
            'hgg_acts': [],
            'done': []
        }
        self.length = 0

    def store_step(self, action, obs, reward, done, hgg_action=None):
        if isinstance(hgg_action, (float, np.float32, np.float64)):
            hgg_action = torch.tensor([hgg_action], dtype=torch.float32)
        elif isinstance(hgg_action, np.ndarray):
            hgg_action = torch.tensor(hgg_action, dtype=torch.float32)
        elif not isinstance(hgg_action, torch.Tensor):
            raise TypeError(f"Unsupported type for hgg_action: {type(hgg_action)}")

        self.ep['acts'].append(action.detach().cpu().clone())

        if hgg_action is not None:
            self.ep['hgg_acts'].append(hgg_action.detach().cpu().clone())
        else:

            self.ep['hgg_acts'].append(action.detach().cpu().clone())

        self.ep['obs'].append(copy.deepcopy(obs))
        self.ep['rews'].append(copy.deepcopy([reward]))
        self.ep['done'].append(copy.deepcopy([np.float32(done)]))
        self.length += 1

    def energy(self, env_id, w_potential=1.0, w_linear=1.0, w_rotational=1.0):
        # from "Energy-Based Hindsight Experience Prioritization"
        if env_id[:5] == 'Fetch':
            obj = []
            for i in range(len(self.ep['obs'])):
                obj.append(self.ep['obs'][i]['achieved_goal'])
            # obj = np.array([obj])
            obj = np.array(obj)

            clip_energy = 0.5
            height = obj[:, :, 2]
            height_0 = np.repeat(height[:, 0].reshape(-1, 1), height[:, 1::].shape[1], axis=1)
            height = height[:, 1::] - height_0
            g, m, delta_t = 9.81, 1, 0.04
            potential_energy = g * m * height
            diff = np.diff(obj, axis=1)
            velocity = diff / delta_t
            kinetic_energy = 0.5 * m * np.power(velocity, 2)
            kinetic_energy = np.sum(kinetic_energy, axis=2)
            energy_totoal = w_potential * potential_energy + w_linear * kinetic_energy
            energy_diff = np.diff(energy_totoal, axis=1)
            energy_transition = energy_totoal.copy()
            energy_transition[:, 1::] = energy_diff.copy()
            energy_transition = np.clip(energy_transition, 0, clip_energy)
            energy_transition_total = np.sum(energy_transition, axis=1)
            energy_final = energy_transition_total.reshape(-1, 1)
            return np.sum(energy_final)
        else:
            assert env_id[:4] == 'Hand'
            obj = []
            for i in range(len(self.ep['obs'])):
                obj.append(self.ep['obs'][i]['observation'][-7:])
            obj = np.array([obj])

            clip_energy = 2.5
            g, m, delta_t, inertia = 9.81, 1, 0.04, 1
            quaternion = obj[:, :, 3:].copy()
            angle = np.apply_along_axis(quaternion_to_euler_angle, 2, quaternion)
            diff_angle = np.diff(angle, axis=1)
            angular_velocity = diff_angle / delta_t
            rotational_energy = 0.5 * inertia * np.power(angular_velocity, 2)
            rotational_energy = np.sum(rotational_energy, axis=2)
            obj = obj[:, :, :3]
            height = obj[:, :, 2]
            height_0 = np.repeat(height[:, 0].reshape(-1, 1), height[:, 1::].shape[1], axis=1)
            height = height[:, 1::] - height_0
            potential_energy = g * m * height
            diff = np.diff(obj, axis=1)
            velocity = diff / delta_t
            kinetic_energy = 0.5 * m * np.power(velocity, 2)
            kinetic_energy = np.sum(kinetic_energy, axis=2)
            energy_totoal = w_potential * potential_energy + w_linear * kinetic_energy + w_rotational * rotational_energy
            energy_diff = np.diff(energy_totoal, axis=1)
            energy_transition = energy_totoal.copy()
            energy_transition[:, 1::] = energy_diff.copy()
            energy_transition = np.clip(energy_transition, 0, clip_energy)
            energy_transition_total = np.sum(energy_transition, axis=1)
            energy_final = energy_transition_total.reshape(-1, 1)
            return np.sum(energy_final)


class ReplayBuffer_Episodic:
    def __init__(self, args):
        self.args = args
        if args.buffer_type == 'energy':
            self.energy = True
            self.energy_sum = 0.0
            self.energy_offset = 0.0
            self.energy_max = 1.0
        else:
            self.energy = False



        self.buffer = {}
        self.steps = []
        self.length = 0
        self.counter = 0
        self.steps_counter = 0
        self.sample_methods = {
            'ddpg': self.sample_batch_ddpg
        }
        self.sample_batch = self.sample_methods[args.alg]
        self.args.goal_based = True

    def store_trajectory(self, trajectory):
        if trajectory.length == 0:
            print("Warning: Attempted to store an empty trajectory, skipped.")
            return

        episode = trajectory.ep
        if self.energy:
            energy = trajectory.energy(self.args.env)
            self.energy_sum += energy

        if self.counter == 0:
            for key in episode.keys():
                self.buffer[key] = []
            if self.energy:
                self.buffer_energy = []
                self.buffer_energy_sum = []

        if self.counter < self.args.buffer_size:
            for key in self.buffer.keys():
                self.buffer[key].append(episode[key])
            if self.energy:
                self.buffer_energy.append(copy.deepcopy(energy))
                self.buffer_energy_sum.append(copy.deepcopy(self.energy_sum))
            self.length += 1
            self.steps.append(trajectory.length)
        else:
            idx = self.counter % self.args.buffer_size
            for key in self.buffer.keys():
                self.buffer[key][idx] = episode[key]
            if self.energy:
                self.energy_offset = copy.deepcopy(self.buffer_energy_sum[idx])
                self.buffer_energy[idx] = copy.deepcopy(energy)
                self.buffer_energy_sum[idx] = copy.deepcopy(self.energy_sum)
            self.steps[idx] = trajectory.length
        self.counter += 1
        self.steps_counter += trajectory.length

    def energy_sample(self):
        t = self.energy_offset + np.random.uniform(0, 1) * (self.energy_sum - self.energy_offset)
        if self.counter > self.args.buffer_size:
            if self.buffer_energy_sum[-1] >= t:
                return self.energy_search(t, self.counter % self.length, self.length - 1)
            else:
                return self.energy_search(t, 0, self.counter % self.length - 1)
        else:
            return self.energy_search(t, 0, self.length - 1)

    def energy_search(self, t, l, r):
        if l == r: return l
        mid = (l + r) // 2
        if self.buffer_energy_sum[mid] >= t:
            return self.energy_search(t, l, mid)
        else:
            return self.energy_search(t, mid + 1, r)



    def sample_batch_ddpg(self, batch_size=-1, normalizer=False, plain=True, use_hindsight=False, use_normal=False, pick_and_place=False, push=True):
        batch = {'obs': [], 'obs_next': [], 'acts': [], 'hgg_acts': [], 'rews': [], 'done': [], 'weights': [],
                 'tree_idxs': []}

        # 检查缓冲区是否为空
        if self.counter == 0 or self.length == 0:
            print("警告：缓冲区为空，无法采样")
            return None

        # 确保批次大小至少为1
        if batch_size < 1:
            batch_size = self.args.batch_size if hasattr(self.args, 'batch_size') else 256

        # 确保批次大小不超过可用数据量
        batch_size = min(batch_size, self.length)

        if use_hindsight:

            # 假设 self.counter 当前值 ≥ 10，为防止越界取 max
            start_idx = max(0, self.counter - 20)
            end_idx = self.counter  # 不包含 self.counter 本身

            idx = np.random.randint(start_idx, end_idx)

            # print("**************use_hindsight************")
            # idx = self.counter-1
            traj_length = self.steps[idx]
            # print("**************traj_length************", traj_length)

            random_start = np.random.randint(3)

            for step in range(random_start, traj_length - 1, 3):
                # if len(batch['obs']) >= batch_size:
                # 	break

                desired_goal = self.buffer['obs'][idx][step]['desired_goal']
                # obs = goal_concat(self.buffer['obs'][idx][step]['observation'], desired_goal)
                # obs_next = goal_concat(self.buffer['obs'][idx][step + 1]['observation'], desired_goal)
                full_observation = np.zeros(13)  # 13维observation
                full_observation[:3] = self.buffer['obs'][idx][step]['achieved_goal']
                full_observation_next = np.zeros(13)  # 13维observation
                full_observation_next[:3] = self.buffer['obs'][idx][step + 1]['achieved_goal']
                obs = goal_concat(full_observation, desired_goal)
                obs_next = goal_concat(full_observation_next, desired_goal)
                act = self.buffer['acts'][idx][step]
                rew = self.buffer['rews'][idx][step]
                done = self.buffer['done'][idx][step]

                # 得到hindsight goal
                current_achieved = self.buffer['obs'][idx][step]['achieved_goal']
                current_dist = np.linalg.norm(current_achieved - desired_goal)


                best_future_step = len(self.buffer['hgg_acts'][idx]) + 1
                # # 从 step+1 开始向后找
                for future_step in range(step + 1, self.steps[idx] + 1):
                    # for future_step in range(step + 1, step + 8):
                    future_achieved = self.buffer['obs'][idx][future_step]['achieved_goal']
                    future_dist = np.linalg.norm(future_achieved - desired_goal)

                    # if future_dist < current_dist:
                    if 0.02 < current_dist - future_dist:
                        best_future_step = future_step
                        break  # 找到比当前更接近的就停止

                if best_future_step >= len(self.buffer['hgg_acts'][idx]):
                    # best_future_step = step  # fallback 回当前step
                    hgg_action = self.buffer['hgg_acts'][idx][step]

                # 取该步的 hgg_acts
                # hgg_action = self.buffer['hgg_acts'][idx][best_future_step]
                else:
                    hgg_action = self.buffer['obs'][idx][best_future_step]['achieved_goal']

                hgg_action = np.array(hgg_action).reshape(-1)

                curr_subgoal_distance = np.linalg.norm(hgg_action - desired_goal)
                reward = curr_subgoal_distance * (-1)


                if curr_subgoal_distance < 0.02:
                    success_bonus = 0.1
                    reward += success_bonus
                    #done = True
                # done = np.array(done).reshape(-1)
                rew = reward * 0.1

                batch['obs'].append(copy.deepcopy(obs))
                batch['obs_next'].append(copy.deepcopy(obs_next))
                batch['acts'].append(copy.deepcopy(act))
                # batch['hgg_acts'].append(copy.deepcopy(self.buffer['hgg_acts'][idx][step]))
                batch['hgg_acts'].append(copy.deepcopy(hgg_action))
                batch['rews'].append(copy.deepcopy(rew))
                batch['done'].append(copy.deepcopy(done))

            return batch

        if push:

            # 假设 self.counter 当前值 ≥ 10，为防止越界取 max
            start_idx = max(0, self.counter - 2000)
            end_idx = self.counter  # 不包含 self.counter 本身

            idx = np.random.randint(start_idx, end_idx)
            traj_length = self.steps[idx]
            #random_start = np.random.randint(3)

            #for step in range(random_start, traj_length - 1, 3):
            for step in range(traj_length - 1):

                desired_goal = self.buffer['obs'][idx][step]['desired_goal']
                # obs = goal_concat(self.buffer['obs'][idx][step]['observation'], desired_goal)
                # obs_next = goal_concat(self.buffer['obs'][idx][step + 1]['observation'], desired_goal)
                full_observation = np.zeros(13)  # 13维observation

                #full_observation[:3] = self.buffer['obs'][idx][step]['achieved_goal']
                full_observation[:6] = self.buffer['obs'][idx][step]['observation']
                full_observation_next = np.zeros(13)  # 13维observation
                #full_observation_next[:3] = self.buffer['obs'][idx][step + 1]['achieved_goal']
                full_observation_next[:6] = self.buffer['obs'][idx][step + 1]['observation']
                obs = goal_concat(full_observation, desired_goal)
                obs_next = goal_concat(full_observation_next, desired_goal)
                act = self.buffer['acts'][idx][step]
                rew = self.buffer['rews'][idx][step]
                done = self.buffer['done'][idx][step]

                obs_raw = np.array(self.buffer['obs'][idx][step]['observation']).squeeze()  # shape (7,)
                object_pos = obs_raw[3:6]
                gripper_pos = obs_raw[0:3]
                object_goal_dist = np.linalg.norm(object_pos - desired_goal)
                gripper_object_dist = np.linalg.norm(gripper_pos - object_pos)

                # if gripper_object_dist > 0.05:
                #     desired_goal = [0.2,-0.23,1.06]

                # 得到hindsight goal
                current_achieved = np.array(self.buffer['obs'][idx][step]['observation']).squeeze()[3:6]
                current_dist = np.linalg.norm(current_achieved - desired_goal)

                best_future_step = len(self.buffer['hgg_acts'][idx]) + 1
                # # 从 step+1 开始向后找
                for future_step in range(step + 1, self.steps[idx] + 1):
                    # for future_step in range(step + 1, step + 8):
                    future_achieved = np.array(self.buffer['obs'][idx][future_step]['observation']).squeeze()[3:6]
                    future_dist = np.linalg.norm(future_achieved - desired_goal)

                    # if future_dist < current_dist:
                    # if 0.001 < current_dist - future_dist:
                    # if 0.01 < current_dist - future_dist: #push
                    if 0.01 < current_dist - future_dist and future_achieved[0] > current_achieved[0]:  # push with obstacle
                    # if 0.02 < current_dist - future_dist:
                        best_future_step = future_step
                        break  # 找到比当前更接近的就停止

                if best_future_step >= len(self.buffer['hgg_acts'][idx]):
                    break



                else:
                    hgg_action = self.buffer['obs'][idx][best_future_step]['achieved_goal']


                    obs_raw = np.array(self.buffer['obs'][idx][best_future_step]['observation']).squeeze()  # shape (7,)
                    object_pos = obs_raw[3:6]
                    object_goal_dist = np.linalg.norm(object_pos - desired_goal)


                    # === 新增：obs_next 用 best_future_step 对应的 obs ===
                    full_observation_future = np.zeros(13)
                    full_observation_future[:6] = self.buffer['obs'][idx][best_future_step]['observation']
                    obs_next = goal_concat(full_observation_future, desired_goal)

                if object_goal_dist<0.05:
                    reward = 0.1
                else:
                    reward = -object_goal_dist
                rew = reward * 0.1

                if (object_goal_dist < 0.05) or (step >= traj_length - 2):
                    done = True
                else:
                    done = False

                done = np.array(done, dtype=np.float32).reshape(-1)

                batch['obs'].append(copy.deepcopy(obs))
                batch['obs_next'].append(copy.deepcopy(obs_next))
                batch['acts'].append(copy.deepcopy(act))
                batch['hgg_acts'].append(copy.deepcopy(hgg_action))
                batch['rews'].append(copy.deepcopy(rew))
                batch['done'].append(copy.deepcopy(done))

            # if batch['done']:  # 列表非空才改最后一个
            #     batch['done'][-1] = np.array([1.0], dtype=np.float32)

            return batch




        # 常规采样
        for i in range(batch_size):
            # if self.energy:
            # 	idx = self.energy_sample()
            # 	print("-----------self.energy_buffer-----------", idx)
            # else:
            # 	idx = np.random.randint(self.length)
            # 	print("---------buffer----------", idx)
            idx = np.random.randint(self.length)
            # print("---------buffer----------", idx)

            step = np.random.randint(self.steps[idx])
            # print("----------step------------", step)

            if self.args.goal_based:
                # print("+++++++++++常规采样 self.args.goal_based+++++++++++++")

                if plain:
                    # print("+++++++++++常规采样 plain+++++++++++++")
                    # print("**************replay buffer goal based with plain************")
                    # no additional tricks
                    goal = self.buffer['obs'][idx][step]['desired_goal']
                # goal = self.buffer['hgg_acts'][idx][step]
                elif normalizer:
                    # uniform sampling for normalizer update
                    goal = self.buffer['obs'][idx][step]['achieved_goal']
                else:
                    # print("*****************upsampling by HER trick***********")
                    # upsampling by HER trick
                    if (self.args.her != 'none') and (np.random.uniform() <= self.args.her_ratio):
                        if self.args.her == 'match':
                            goal = self.args.goal_sampler.sample()
                            goal_pool = np.array([obs['achieved_goal'] for obs in self.buffer['obs'][idx][step + 1:]])
                            step_her = (step + 1) + np.argmin(np.sum(np.square(goal_pool - goal), axis=1))
                            goal = self.buffer['obs'][idx][step_her]['achieved_goal']
                        else:
                            step_her = {
                                'final': self.steps[idx],
                                'future': np.random.randint(step + 1, self.steps[idx] + 1)
                            }[self.args.her]
                            goal = self.buffer['obs'][idx][step_her]['achieved_goal']
                    # print("*****************upsampling by HER trick future***********")
                    else:
                        goal = self.buffer['obs'][idx][step]['desired_goal']

                desired_goal = self.buffer['obs'][idx][step]['desired_goal']
                achieved = self.buffer['obs'][idx][step + 1]['achieved_goal']
                achieved_old = self.buffer['obs'][idx][step]['achieved_goal']
                state = goal_concat(self.buffer['obs'][idx][step]['observation'], goal)
                obs = state
                # obs = goal_concat(state, self.buffer['hgg_acts'][idx][step])
                state_next = goal_concat(self.buffer['obs'][idx][step + 1]['observation'], goal)
                obs_next = state_next
                # obs_next = goal_concat(state_next, self.buffer['hgg_acts'][idx][step])
                act = self.buffer['acts'][idx][step]
                rew = self.buffer['rews'][idx][step]
                # rew = self.args.compute_reward((achieved, achieved_old), goal)
                # rew = self.args.compute_reward(achieved, goal)
                # rew = self.args.compute_reward(goal, desired_goal)
                # print("**********rew**********", rew)
                # if 0 < abs(reward) < 0.03:
                # 	reward_goal = 300
                # else:
                # 	reward_goal = 0
                # rew = reward + reward_goal
                done = self.buffer['done'][idx][step]

                batch['obs'].append(copy.deepcopy(obs))
                batch['obs_next'].append(copy.deepcopy(obs_next))
                batch['acts'].append(copy.deepcopy(act))
                batch['hgg_acts'].append(copy.deepcopy(self.buffer['hgg_acts'][idx][step]))
                # batch['rews'].append(copy.deepcopy([rew]))
                batch['rews'].append(copy.deepcopy(rew))
                batch['done'].append(copy.deepcopy(done))
            else:
                print("+++++++++++常规采样 not self.args.goal_based+++++++++++++")
                for key in ['obs', 'acts', 'rews', 'done']:
                    if key == 'obs':
                        batch['obs'].append(copy.deepcopy(self.buffer[key][idx][step]))
                        batch['obs_next'].append(copy.deepcopy(self.buffer[key][idx][step + 1]))
                    else:
                        batch[key].append(copy.deepcopy(self.buffer[key][idx][step]))

        return batch
