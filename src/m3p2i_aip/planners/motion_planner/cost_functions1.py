import torch
from m3p2i_aip.utils import skill_utils
import m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from torch.profiler import record_function

class Objective(object):
    def __init__(self, cfg, nn_model):
        self.cfg = cfg
        self.multi_modal = cfg.multi_modal
        # self.num_samples = cfg.mppi.num_samples
        # self.half_samples = int(cfg.mppi.num_samples / 2)
        self.num_samples = 50
        self.half_samples = int(50 / 2)
        self.device = self.cfg.mppi.device
        self.pre_height_diff = cfg.pre_height_diff
        self.tilt_cos_theta = 0.5

        #self.N_traj = 200
        self.N_traj = 50
        # self.N_traj = 32
        self.step = 0
        self.nn_model = nn_model
        self.n_closest_obs = 2
        self.dst_thr = 0.5
        self.N_KERNEL_MAX = 50

        self.tensor_args = {'device': torch.device('cuda:0'), 'dtype': torch.float32}
        self.nn_model.allocate_gradients(self.N_traj * self.n_closest_obs, self.tensor_args)
        # self.nn_model.allocate_gradients(self.N_traj + self.N_KERNEL_MAX, self.tensor_args)

    def update_objective(self, task, goal):
        self.task = task
        # print("self.task", self.task)
        self.goal = goal if torch.is_tensor(goal) else torch.tensor(goal, device=self.device)

    def compute_cost(self, sim: wrapper):
        print("self.task", self.task)
        print("self.goal", self.goal)

        q0 = sim._dof_state[0]
        #self.n_dof = q0.shape[0]
        self.n_dof = 7

        self.ignored_links = [0, 1, 2]
        if self.n_dof < 7:
            self.ignored_links = []

        task_cost = 0
        if self.task == "navigation":
            # task_cost = self.get_navigation_cost(sim)
            task_cost = self.get_ee_navigation_cost(sim)
        elif self.task == "push":
            return self.get_push_cost(sim, self.goal)
        elif self.task == "pull":
            return self.get_pull_cost(sim, self.goal)
        elif self.task == "push_pull":
            return torch.cat((self.get_push_cost(sim, self.goal)[:self.half_samples],
                              self.get_pull_cost(sim, self.goal)[self.half_samples:]), dim=0)
        elif self.task == "reach":
            return self.get_panda_reach_cost(sim, self.goal)
        elif self.task == "pick":
            # task_cost = self.get_panda_pick_cost(sim, self.goal)
            task_cost = self.get_ee_navigation_cost(sim)
        elif self.task == "place":
            return self.get_panda_place_cost(sim)

        get_motion_cost_1 = self.get_motion_cost_1(sim)
        task_cost = self.get_ee_navigation_cost(sim)

        return 5*task_cost + 5*get_motion_cost_1 + self.get_pick_tilt_cost(sim) #push with obstacle


    def get_navigation_cost(self, sim: wrapper):
        return torch.linalg.norm(sim.robot_pos - self.goal, axis=1)

    def calculate_dist(self, sim: wrapper, block_goal: torch.tensor):
        self.block_pos = sim.get_actor_position_by_name("box")[:, :2]  # x, y position
        robot_to_block = sim.robot_pos - self.block_pos
        block_to_goal = block_goal - self.block_pos

        robot_to_block_dist = torch.linalg.norm(robot_to_block, axis=1)
        block_to_goal_dist = torch.linalg.norm(block_to_goal, axis=1)

        self.dist_cost = robot_to_block_dist + block_to_goal_dist * 10
        self.cos_theta = torch.sum(robot_to_block * block_to_goal, 1) / (robot_to_block_dist * block_to_goal_dist)

    def get_push_cost(self, sim: wrapper, block_goal: torch.tensor):
        # Calculate dist cost
        self.calculate_dist(sim, block_goal)

        # Force the robot behind block and goal, align_cost is actually cos(theta)+1
        align_cost = torch.zeros(self.num_samples, device=self.device)
        align_cost[self.cos_theta > 0] = self.cos_theta[self.cos_theta > 0]

        return 3 * self.dist_cost + 1 * align_cost

    def get_pull_cost(self, sim: wrapper, block_goal: torch.tensor):
        self.calculate_dist(sim, block_goal)
        pos_dir = self.block_pos - sim.robot_pos
        robot_to_block_dist = torch.linalg.norm(pos_dir, axis=1)

        # True means the velocity moves towards block, otherwise means pull direction
        flag_towards_block = torch.sum(sim.robot_vel * pos_dir, 1) > 0

        # simulate a suction to the box
        suction_force = skill_utils.calculate_suction(self.cfg, sim)
        # Set no suction force if robot moves towards the block
        suction_force[flag_towards_block] = 0
        if self.multi_modal:
            suction_force[:self.half_samples] = 0
        sim.apply_rigid_body_force_tensors(suction_force)

        self.calculate_dist(sim, block_goal)

        # Force the robot to be in the middle between block and goal, align_cost is actually 1-cos(theta)
        align_cost = torch.zeros(self.num_samples, device=self.device)
        align_cost[self.cos_theta < 0] = -self.cos_theta[self.cos_theta < 0]  # (1 - cos_theta)

        # Add the cost when the robot is close to the block and moves towards the block
        vel_cost = torch.zeros(self.num_samples, device=self.device)
        robot_block_close = robot_to_block_dist <= 0.5
        vel_cost[flag_towards_block * robot_block_close] = 0.6

        return 3 * self.dist_cost + 3 * vel_cost + 7 * align_cost

    def get_panda_reach_cost(self, sim, pre_pick_goal):
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_r_state = sim.get_actor_link_by_name("panda", "panda_rightfinger")
        ee_state = (ee_l_state + ee_r_state) / 2
        cube_state = sim.get_actor_link_by_name("cubeA", "box")
        if not self.multi_modal:
            pre_pick_goal = cube_state[0, :3].clone()
            pre_pick_goal[2] += self.pre_height_diff
            reach_cost = torch.linalg.norm(ee_state[:, :3] - pre_pick_goal, axis=1)
            # print("pre_pick_goal", pre_pick_goal)
        else:
            pre_pick_goal = cube_state[:, :3].clone()
            pre_pick_goal_1 = cube_state[0, :3].clone()
            pre_pick_goal_2 = cube_state[0, :3].clone()
            pre_pick_goal_1[2] += self.pre_height_diff
            pre_pick_goal_2[0] -= self.pre_height_diff * self.tilt_cos_theta
            pre_pick_goal_2[2] += self.pre_height_diff * (1 - self.tilt_cos_theta ** 2) ** 0.5
            pre_pick_goal[:self.half_samples, :] = pre_pick_goal_1
            pre_pick_goal[self.half_samples:, :] = pre_pick_goal_2
            reach_cost = torch.linalg.norm(ee_state[:, :3] - pre_pick_goal, axis=1)

            # Compute the tilt value between ee and cube
        tilt_cost = self.get_pick_tilt_cost(sim)

        # print("***********reach cost***********", reach_cost)
        return 10 * reach_cost + tilt_cost

    def get_panda_pick_cost(self, sim, pre_place_state):
        cube_state = sim.get_actor_link_by_name("cubeA", "box")

        # Move to pre-place location
        goal_cost = torch.linalg.norm(pre_place_state[:3] - cube_state[:, :3], axis=1)
        cube_quaternion = cube_state[:, 3:7]
        goal_quatenion = pre_place_state[3:7].repeat(self.num_samples).view(self.num_samples, 4)
        ori_cost = skill_utils.get_general_ori_cube2goal(cube_quaternion, goal_quatenion)

        return 10 * goal_cost + 15 * ori_cost

    def get_panda_place_cost(self, sim):
        # task planner will send discrete gripper commands instead of sampling

        # Just to make mppi running! Actually this is not useful!!
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_r_state = sim.get_actor_link_by_name("panda", "panda_rightfinger")
        gripper_dist = torch.linalg.norm(ee_l_state[:, :3] - ee_r_state[:, :3], axis=1)
        gripper_cost = 2 * (1 - gripper_dist)

        return gripper_cost

    def get_pick_tilt_cost(self, sim):
        # This measures the cost of the tilt angle between the end effector and the cube
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_quaternion = ee_l_state[:, 3:7]
        cubeA_ori = sim.get_actor_orientation_by_name("cubeA")
        #cubeA_ori = sim.get_actor_orientation_by_name("cubeB")
        # cube_quaternion = cube_state[:, 3:7]
        if not self.multi_modal:
            # To make the z-axis direction of end effector to be perpendicular to the cube surface
            ori_ee2cube = skill_utils.get_general_ori_ee2cube(ee_quaternion, cubeA_ori, tilt_value=0)
            # ori_ee2cube = skill_utils.get_ori_ee2cube(ee_quaternion, cubeA_ori)
        else:
            # To combine costs of different tilt angles
            cost_1 = skill_utils.get_general_ori_ee2cube(ee_quaternion[:self.half_samples],
                                                         cubeA_ori[:self.half_samples], tilt_value=0)
            cost_2 = skill_utils.get_general_ori_ee2cube(ee_quaternion[self.half_samples:],
                                                         cubeA_ori[self.half_samples:], tilt_value=self.tilt_cos_theta)
            ori_ee2cube = torch.cat((cost_1, cost_2), dim=0)

        return 3 * ori_ee2cube

    def get_motion_cost(self, sim):
        if self.cfg.env_type == 'point_env':
            obs_force = sim.get_actor_contact_forces_by_name("dyn-obs", "box")  # [num_envs, 3]
        elif self.cfg.env_type == 'panda_env':
            obs_force = sim.get_actor_contact_forces_by_name("table", "box")
            obs_force += 4 * sim.get_actor_contact_forces_by_name("shelf_stand", "box")
            obs_force += sim.get_actor_contact_forces_by_name("cubeB", "box")

            table_position = sim.get_actor_position_by_name("table")
            shelf_position = sim.get_actor_position_by_name("shelf_stand")
            cubeB_position = sim.get_actor_position_by_name("cubeB")
            cubeA_position = sim.get_actor_position_by_name("cubeA")

        coll_cost = torch.sum(torch.abs(obs_force[:, :2]), dim=1)  # [num_envs]
        # Binary check for collisions.
        coll_cost[coll_cost > 0.1] = 1
        coll_cost[coll_cost <= 0.1] = 0

        return 1000 * coll_cost

    def collision_cost(self):

        # Binary check for collisions.
        self.closest_dist_all[self.closest_dist_all < 0] = 1
        self.closest_dist_all[self.closest_dist_all > 0] = 0

        return 1000 * self.closest_dist_all

    def get_motion_cost_1(self, sim: wrapper):
        q_prev = sim._dof_state

        distance, self.nn_grad = self.distance_repulsion_nn(sim, q_prev, aot=False)
        self.nn_grad = self.nn_grad[0:self.N_traj, :]
        distance = distance[0:self.N_traj]

        safe_margin = 0.4

        eps = 1e-6

        close_mask = distance < safe_margin
        far_mask = ~close_mask

        cost = torch.zeros_like(distance)

        cost[close_mask] = torch.exp(-distance[close_mask] + safe_margin) * 100

        cost[far_mask] = 0.1 / (distance[far_mask] + eps)

        return cost

    def obs_positions(self, sim):
        device = sim.device
        actor_names = [ "dyn-obs", "shelf_stand"]
        all_spheres = []

        for actor_name in actor_names:
            #size = sim.get_actor_size(actor_name)[0]  # Tensor: (3,) → [length, width, height]
            size_raw = sim.get_actor_size(actor_name)
            if not isinstance(size_raw, torch.Tensor):
                size = torch.tensor(size_raw, device=sim.device)
            else:
                size = size_raw[0] if size_raw.ndim == 2 else size_raw

            position = sim.get_actor_position_by_name(actor_name)[0]  # Tensor: (3,) → [x, y, z]
            position += torch.tensor([0.45, 0., -1.125], device=position.device, dtype=position.dtype)

            shortest_dim = torch.argmin(size).item()
            longest_dims = [i for i in range(3) if i != shortest_dim]
            radius = size[shortest_dim] / 2

            step = 2 * radius
            linspaces = []
            for dim in longest_dims:
                center = position[dim]
                span = size[dim]
                start = center - span / 2
                end = center + span / 2
                lin = torch.arange(start, end + 1e-6, step, device=device)
                linspaces.append(lin)

            mesh1, mesh2 = torch.meshgrid(linspaces[0], linspaces[1], indexing="ij")
            mesh1 = mesh1.flatten()
            mesh2 = mesh2.flatten()

            for i in range(len(mesh1)):
                pos = torch.zeros(3, device=device)
                pos[longest_dims[0]] = mesh1[i]
                pos[longest_dims[1]] = mesh2[i]
                pos[shortest_dim] = position[shortest_dim]

                sphere = torch.cat([pos, radius.unsqueeze(0)])  # [x, y, z, r]
                all_spheres.append(sphere.unsqueeze(0))

        return torch.vstack(all_spheres).to(device)  # shape: [N, 4]

    def build_nn_input(self, q_tens, obs_tens):

        self.nn_input = torch.hstack(
            (q_tens.tile(obs_tens.shape[0], 1), obs_tens.repeat_interleave(q_tens.shape[0], 0)))
        return self.nn_input

    def distance_repulsion_nn(self, sim, q_prev, aot=False):
        device = sim.device
        n_inputs = q_prev.shape[0]
        #print("q_prev.shape", q_prev.shape)
        self.obs = self.obs_positions(sim).to(device)

        # self.obs = torch.tensor([2.0+0.45,0,-1.125,0]).to(device)
        # self.obs = self.obs.unsqueeze(0)
        self.n_obs = self.obs.shape[0]

        with record_function("TAG: evaluate NN_1 (build input)"):
            # building input tensor for NN (N_traj * n_obs, n_dof + 3)
            nn_input = self.build_nn_input(q_prev[:, :7], self.obs).to(device)

        with record_function("TAG: evaluate NN_2 (forward pass)"):

            self.nn_model.model_jit = self.nn_model.model_jit.to(device)

            #padding = torch.zeros(nn_input.size(0), 31 - nn_input.size(1), device=nn_input.device)
            #nn_input_padded = torch.cat((nn_input, padding), dim=1)
            #nn_dist = self.nn_model.model_jit.forward(nn_input_padded[:, 0:-1])
            nn_dist = self.nn_model.model_jit.forward(nn_input[:, 0:-1])
            # nn_dist = torch.cat((x, torch.sin(x), torch.cos(x)), dim=-1)
            print("nn_dist", nn_dist[0])

            if self.nn_model.out_channels == 9:
                nn_dist = nn_dist / 100  # scale down to meters
        with record_function("TAG: evaluate NN_3 (get closest obstacle)"):
            # rebuilding input tensor to only include closest obstacles
            nn_input_resized = nn_input[:nn_dist.shape[0], -1].unsqueeze(1)
            nn_dist -= nn_input_resized

            # nn_dist -= nn_input[:, -1].unsqueeze(1)  # subtract radius
            nn_dist[:, self.ignored_links] = 1e6  # ignore specified links
            mindist, _ = nn_dist.min(1)

            mindist_matrix = mindist.reshape(self.n_obs, n_inputs).transpose(0, 1)
            mindist, sphere_idx = mindist_matrix.min(1)
            sort_dist, sort_idx = mindist_matrix.sort(dim=1)
            mindist_arr = sort_dist[:, 0:self.n_closest_obs]
            sphere_idx_arr = sort_idx[:, 0:self.n_closest_obs]
            # mask_idx = self.traj_range[:n_inputs] + sphere_idx * n_inputs
            # mask_idx = torch.arange(n_inputs) + sphere_idx * n_inputs
            # Ensure all tensors are on the same device
            # sphere_idx = sphere_idx.to(device)
            # sphere_idx_arr = sphere_idx_arr.to(device)

            mask_idx = torch.arange(n_inputs, device=device) + sphere_idx * n_inputs

            # nn_input = nn_input[mask_idx, :]

            # new_mask_idx = torch.arange(n_inputs).unsqueeze(1).repeat(1, self.n_closest_obs) + sphere_idx_arr * n_inputs
            # nn_input = nn_input[new_mask_idx.flatten(), :]

            # Handle new mask_idx for closest obstacles
            new_mask_idx = (
                    torch.arange(n_inputs, device=device)
                    .unsqueeze(1)
                    .repeat(1, self.n_closest_obs)
                    + sphere_idx_arr * n_inputs
            )

            nn_input = nn_input[new_mask_idx.flatten(), :]

        with record_function("TAG: evaluate NN_4 (forward+backward pass)"):

            if aot:
                nn_dist, nn_grad, nn_minidx = self.nn_model.dist_grad_closest_aot(nn_input)
            else:

                nn_dist, nn_grad, nn_minidx = self.nn_model.dist_grad_closest(nn_input[:, 0:-1])
                nn_grad = nn_grad.squeeze(2)

            self.nn_grad = nn_grad[:nn_input.shape[0], 0:self.n_dof]
            if self.nn_model.out_channels == 9:
                nn_dist = nn_dist / 100  # scale down to meters

        with record_function("TAG: evaluate NN_5 (process outputs)"):

            # cleaning up to get distances and gradients for closest obstacles
            nn_dist -= nn_input[:, -1].unsqueeze(1)  # subtract radius and some threshold
            # extract closest link distance
            nn_dist = nn_dist[torch.arange(self.n_closest_obs * n_inputs).unsqueeze(1), nn_minidx.unsqueeze(1)]
            # reshape to match n_traj x n_closest_obs
            nn_dist = nn_dist.reshape(n_inputs, self.n_closest_obs)
            self.nn_grad = self.nn_grad.reshape(n_inputs, self.n_closest_obs, self.n_dof)
            # weight gradients according to closest distance
            weighting = (-10 * nn_dist).softmax(dim=-1)
            self.nn_grad = torch.sum(self.nn_grad * weighting.unsqueeze(2), dim=1)
            # distance - mindist
            distance = nn_dist[:, 0]
        return distance, self.nn_grad

    def update_obstacles(self, obs):
        self.obs = obs
        self.n_obs = obs.shape[0]
        return 0

    def get_ee_navigation_cost(self, sim):
        # Get end-effector states (left finger + right finger)/ 2
        left_finger = sim.get_actor_link_by_name("panda", "panda_leftfinger")[:, :3]  # position
        right_finger = sim.get_actor_link_by_name("panda", "panda_rightfinger")[:, :3]
        ee_pos = (left_finger + right_finger) / 2.0  # shape: [num_envs, 3]

        cost = torch.linalg.norm(ee_pos - self.goal, dim=1)  # shape: [num_envs]

        return cost