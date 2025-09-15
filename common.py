from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env

import numpy as np
import copy
from envs import make_env, clip_return_range, Robotics_envs_id
from utils.os_utils import get_arg_parser, get_logger, str2bool
from algorithm import create_agent, create_bc_diffusion_agent, create_ql_diffusion_agent

from learner import create_learner, learner_collection
from test import Tester
from algorithm.replay_buffer import ReplayBuffer_Episodic, goal_based_process

import  learner.utils.isaacgym_utils.isaacgym_wrapper as wrapper

import time
import uuid
import os

hyperparameters = {
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 1},
    'hopper-medium-v2':              {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 2},
    'walker2d-medium-v2':            {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 1.0,  'top_k': 1},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 2.0,  'top_k': 0},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 2},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 1},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 7.0,  'top_k': 0},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 2},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 1},
    'antmaze-umaze-v0':              {'lr': 3e-4, 'eta': 0.5,   'max_q_backup': False,  'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 2},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-4, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 3.0,  'top_k': 2},
    'antmaze-medium-play-v0':        {'lr': 1e-3, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 1},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'eta': 3.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 1.0,  'top_k': 1},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'eta': 4.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'eta': 3.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 1},
    'pen-human-v1':                  {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'pen-cloned-v1':                 {'lr': 3e-5, 'eta': 0.1,   'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 8.0,  'top_k': 2},
    'kitchen-complete-v0':           {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 250 , 'gn': 9.0,  'top_k': 2},
    'kitchen-partial-v0':            {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'kitchen-mixed-v0':              {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 0},
}


def get_args():
	parser = get_arg_parser()

	parser.add_argument('--tag', help='terminal tag in logger', type=str, default='')
	parser.add_argument('--alg', help='backend algorithm', type=str, default='ddpg', choices=['ddpg', 'ddpg2'])
	parser.add_argument('--learn', help='type of training method', type=str, default='hgg', choices=learner_collection.keys())

	#parser.add_argument('--env', help='gym env id', type=str, default='FetchPush-v1')
	args, _ = parser.parse_known_args()

	parser.add_argument("--subgoal_data_path", default="subgoal_data.npz", type=str)

	parser.add_argument('--gamma', help='discount factor', type=np.float32, default=0.98)
	#parser.add_argument('--clip_return', help='whether to clip return value', type=str2bool, default=True)
	parser.add_argument('--eps_act', help='percentage of epsilon greedy explorarion', type=np.float32, default=0.3)
	parser.add_argument('--std_act', help='standard deviation of uncorrelated gaussian explorarion', type=np.float32, default=0.2)

	parser.add_argument('--pi_lr', help='learning rate of policy network', type=np.float32, default=1e-3)
	parser.add_argument('--q_lr', help='learning rate of value network', type=np.float32, default=1e-3)
	parser.add_argument('--act_l2', help='quadratic penalty on actions', type=np.float32, default=1.0)
	parser.add_argument('--polyak', help='interpolation factor in polyak averaging for DDPG', type=np.float32, default=0.95)

	parser.add_argument('--epochs', help='number of epochs', type=np.int32, default=20)
	parser.add_argument('--cycles', help='number of cycles per epoch', type=np.int32, default=20)
	parser.add_argument('--episodes', help='number of episodes per cycle', type=np.int32, default=50)
	parser.add_argument('--timesteps', help='number of timesteps per episode', type=np.int32, default=50)
	parser.add_argument('--train_batches', help='number of batches to train per episode', type=np.int32, default=20)


	#parser.add_argument('--her', help='type of hindsight experience replay', type=str, default='future', choices=['none', 'final', 'future'])
	parser.add_argument('--her_ratio', help='ratio of hindsight experience replay', type=np.float32, default=0.8)
	parser.add_argument('--pool_rule', help='rule of collecting achieved states', type=str, default='full', choices=['full', 'final'])

	#parser.add_argument('--hgg_c', help='weight of initial distribution in flow learner', type=np.float32, default=3.0)
	#parser.add_argument('--hgg_L', help='Lipschitz constant', type=np.float32, default=5.0)
	parser.add_argument('--hgg_pool_size', help='size of achieved trajectories pool', type=np.int32, default=1000)

	#parser.add_argument('--save_acc', help='save successful rate', type=str2bool, default=True)

	# DDPG parameters
	parser.add_argument('--clip_return', type=float, default=50., help='clip return in critic update')

	# training parameters
	parser.add_argument('--n_test_rollouts', type=int, default=1, help='number of test rollouts')
	parser.add_argument('--evaluate_episodes', type=int, default=10, help='max number of episodes')

	# setting for different environments
	parser.add_argument('--env', type=str, default='FetchReach-v1', help='env name')
	parser.add_argument('--env_type', type=str, default='gym', choices=['gym', 'isaac'], help='environment type')
	#parser.add_argument('--learn', type=str, default='hgg', help='learn type')
	parser.add_argument('--goal_based', type=str2bool, default=True, help='whether use goal-based RL method')

	# reward type
	parser.add_argument('--sparse_reward', type=str2bool, default=True, help='whether use sparse reward')
	parser.add_argument('--reward_type', type=str, default='sparse', help='reward type')

	# hyper parameters
	parser.add_argument('--buffer_size', type=int, default=100000, help='replay buffer size')
	parser.add_argument('--dynamics_buffer_size', type=int, default=100000, help='hyper params')
	parser.add_argument('--fake_buffer_size', type=int, default=10000, help='hyper params')
	parser.add_argument('--gen_buffer_size', type=int, default=10000, help='hyper params')
	parser.add_argument('--dynamic_batchsize', type=int, default=16, help='hyper params')
	parser.add_argument('--gen_batchsize', type=int, default=16, help='hyper params')
	parser.add_argument('--warmup', type=int, default=2000, help='warm up steps')
	parser.add_argument('--coll_r', type=float, default=0.1, help='hgg collision_threshold')
	parser.add_argument('--inner_r', type=float, default=0.8, help='hgg inner radius')
	parser.add_argument('--outer_r', type=float, default=1.0, help='hgg outer radius')
	parser.add_argument('--buffer_type', type=str, default='energy', help='replay buffer type')

	# HER parameters
	parser.add_argument('--hgg_L', type=int, default=10, help='hyper params')
	parser.add_argument('--hgg_c', type=float, default=3.0, help='hyper params')

	# RIS parameters
	parser.add_argument('--her', type=str, default='future', help='her strategy during training')
	parser.add_argument('--her_k', type=int, default=4, help='use k experiences for each transition')

	# model save and load
	parser.add_argument('--save_acc', type=float, default=0.0, help='save exp when acc greater than this threshold')
	parser.add_argument('--save_episodes', type=int, default=10,
						help='save models when acc greater than this threshold')

	'''
	diffusion model parameter setup
	'''
	### Experimental Setups ###
	parser.add_argument("--exp", default='exp_1', type=str)  # Experiment ID
	parser.add_argument('--device', default=0, type=int)  # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
	parser.add_argument("--env_name", default="walker2d-medium-expert-v2", type=str)  # OpenAI gym environment name
	parser.add_argument("--dir", default="results", type=str)  # Logging directory
	parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--num_steps_per_epoch", default=1000, type=int)

	### Optimization Setups ###
	parser.add_argument("--batch_size", default=256, type=int)
	parser.add_argument("--lr_decay", action='store_true')
	parser.add_argument('--early_stop', action='store_true')
	parser.add_argument('--save_best_model', action='store_true')

	### RL Parameters ###
	parser.add_argument("--discount", default=0.99, type=float)
	parser.add_argument("--tau", default=0.005, type=float)

	### Diffusion Setting ###
	parser.add_argument("--T", default=5, type=int)
	parser.add_argument("--beta_schedule", default='vp', type=str)
	### Algo Choice ###
	parser.add_argument("--algo", default="ql", type=str)  # ['bc', 'ql']
	parser.add_argument("--ms", default='offline', type=str, help="['online', 'offline']")
	# parser.add_argument("--top_k", default=1, type=int)

	args = parser.parse_args()
	#args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
	args.output_dir = f'{args.dir}'

	args.num_epochs = hyperparameters[args.env_name]['num_epochs']
	args.eval_freq = hyperparameters[args.env_name]['eval_freq']
	args.eval_episodes = 10 if 'v2' in args.env_name else 100

	args.lr = hyperparameters[args.env_name]['lr']
	args.eta = hyperparameters[args.env_name]['eta']
	args.max_q_backup = hyperparameters[args.env_name]['max_q_backup']
	args.reward_tune = hyperparameters[args.env_name]['reward_tune']
	args.gn = hyperparameters[args.env_name]['gn']
	args.top_k = hyperparameters[args.env_name]['top_k']

	args, _ = parser.parse_known_args()

	logger_name = args.alg+'-'+args.env+'-'+args.learn
	if args.tag!='': logger_name = args.tag+'-'+logger_name
	args.logger = get_logger(logger_name)

	for key, value in args.__dict__.items():
		if key!='logger':
			args.logger.info('{}: {}'.format(key,value))

	return args

def experiment_setup(args):

	env = wrapper.IsaacGymWrapper(
		"panda",
		"panda_env",
		num_envs=1,
		viewer=False,
		device="cuda:0",
		cube_on_shelf=False,
	)
	env_test = env

	# Handling the observation space and action space
	if args.goal_based:
		if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'spaces'):
			args.obs_dims = list(env.observation_space.spaces['observation'].shape)
			args.acts_dims = list(env.action_space.shape)
			args.goal_dims = list(env.observation_space.spaces['desired_goal'].shape)

			args.obs_dims[0] += args.goal_dims[0]
			if hasattr(env, 'compute_reward'):
				args.compute_reward = env.compute_reward
			if hasattr(env, 'compute_distance'):
				args.compute_distance = env.compute_distance
		else:
			# specifying the dimensions for the IsaacGym environment
			args.obs_dims = [9]  # Modify based on the IsaacGym environment
			args.acts_dims = [3]  # Modify based on the IsaacGym environment
			args.goal_dims = [3]  # Modify based on the IsaacGym environment

			# Ensure that functions for computing rewards and distances are provided
			if not hasattr(args, 'compute_reward'):
				args.compute_reward = lambda achieved, goal, info: -float(np.linalg.norm(achieved - goal) > 0.05)
			if not hasattr(args, 'compute_distance'):
				args.compute_distance = lambda achieved, goal: np.linalg.norm(achieved - goal)
	else:
		if hasattr(env, 'observation_space'):
			args.obs_dims = list(env.observation_space.shape)
			args.acts_dims = list(env.action_space.shape)
		else:
			# specifying the dimensions for the IsaacGym environment
			args.obs_dims = [9]  # Modify based on the IsaacGym environment
			args.acts_dims = [3]  # Modify based on the IsaacGym environment

	args.buffer = buffer = ReplayBuffer_Episodic(args)
	args.learner = learner = create_learner(args)

	args.agent = agent = create_agent(args)
	args.bc_diffusion_agent = bc_diffusion_agent = create_bc_diffusion_agent()
	args.ql_diffusion_agent = ql_diffusion_agent = create_ql_diffusion_agent(args)

	print("Agent Type：", type(agent))
	args.logger.info('*** network initialization complete ***')
	args.tester = tester = Tester(args)
	args.logger.info('*** tester initialization complete ***')

	return env, env_test, agent, bc_diffusion_agent, ql_diffusion_agent, buffer, learner, tester

def goal_distance(goal_a, goal_b):
	return np.linalg.norm(goal_a - goal_b, ord=2)




















# def get_args():
# 	parser = get_arg_parser()
#
# 	parser.add_argument('--tag', help='terminal tag in logger', type=str, default='')
# 	parser.add_argument('--alg', help='backend algorithm', type=str, default='ddpg', choices=['ddpg', 'ddpg2'])
# 	parser.add_argument('--learn', help='type of training method', type=str, default='hgg', choices=learner_collection.keys())
#
# 	parser.add_argument('--env', help='gym env id', type=str, default='FetchPush-v1')
# 	args, _ = parser.parse_known_args()
# 	if args.env=='HandReach-v0':
# 		parser.add_argument('--goal', help='method of goal generation', type=str, default='reach', choices=['vanilla', 'reach'])
# 	else:
# 		parser.add_argument('--goal', help='method of goal generation', type=str, default='interval', choices=['vanilla', 'fixobj', 'interval', 'obstacle'])
# 		if args.env[:5]=='Fetch':
# 			parser.add_argument('--init_offset', help='initial offset in fetch environments', type=np.float32, default=1.0)
# 		elif args.env[:4]=='Hand':
# 			parser.add_argument('--init_rotation', help='initial rotation in hand environments', type=np.float32, default=0.25)
#
# 	parser.add_argument('--gamma', help='discount factor', type=np.float32, default=0.98)
# 	parser.add_argument('--clip_return', help='whether to clip return value', type=str2bool, default=True)
# 	parser.add_argument('--eps_act', help='percentage of epsilon greedy explorarion', type=np.float32, default=0.3)
# 	parser.add_argument('--std_act', help='standard deviation of uncorrelated gaussian explorarion', type=np.float32, default=0.2)
#
# 	parser.add_argument('--pi_lr', help='learning rate of policy network', type=np.float32, default=1e-3)
# 	parser.add_argument('--q_lr', help='learning rate of value network', type=np.float32, default=1e-3)
# 	parser.add_argument('--act_l2', help='quadratic penalty on actions', type=np.float32, default=1.0)
# 	parser.add_argument('--polyak', help='interpolation factor in polyak averaging for DDPG', type=np.float32, default=0.95)
#
# 	parser.add_argument('--epochs', help='number of epochs', type=np.int32, default=20)
# 	parser.add_argument('--cycles', help='number of cycles per epoch', type=np.int32, default=20)
# 	parser.add_argument('--episodes', help='number of episodes per cycle', type=np.int32, default=50)
# 	parser.add_argument('--timesteps', help='number of timesteps per episode', type=np.int32, default=(50 if args.env[:5]=='Fetch' else 100))
# 	parser.add_argument('--train_batches', help='number of batches to train per episode', type=np.int32, default=20)
#
# 	#parser.add_argument('--buffer_size', help='number of episodes in replay buffer', type=np.int32, default=10000)
# 	parser.add_argument('--buffer_type', help='type of replay buffer / whether to use Energy-Based Prioritization', type=str, default='energy', choices=['normal','energy'])
# 	parser.add_argument('--batch_size', help='size of sample batch', type=np.int32, default=256)
# 	#parser.add_argument('--warmup', help='number of timesteps for buffer warmup', type=np.int32, default=10000)
# 	#parser.add_argument('--her', help='type of hindsight experience replay', type=str, default='future', choices=['none', 'final', 'future'])
# 	parser.add_argument('--her_ratio', help='ratio of hindsight experience replay', type=np.float32, default=0.8)
# 	parser.add_argument('--pool_rule', help='rule of collecting achieved states', type=str, default='full', choices=['full', 'final'])
#
# 	#parser.add_argument('--hgg_c', help='weight of initial distribution in flow learner', type=np.float32, default=3.0)
# 	#parser.add_argument('--hgg_L', help='Lipschitz constant', type=np.float32, default=5.0)
# 	parser.add_argument('--hgg_pool_size', help='size of achieved trajectories pool', type=np.int32, default=1000)
#
# 	#parser.add_argument('--save_acc', help='save successful rate', type=str2bool, default=True)
#
# 	# training parameters
# 	parser.add_argument('--n_test_rollouts', type=int, default=1, help='number of test rollouts')
# 	parser.add_argument('--evaluate_episodes', type=int, default=10, help='max number of episodes')
#
# 	# setting for different environments
# 	#parser.add_argument('--env', type=str, default='FetchReach-v1', help='env name')
# 	parser.add_argument('--env_type', type=str, default='isaac', choices=['gym', 'isaac'], help='environment type')
# 	#parser.add_argument('--learn', type=str, default='hgg', help='learn type')
# 	parser.add_argument('--goal_based', type=str2bool, default=True, help='whether use goal-based RL method')
#
# 	# reward type
# 	parser.add_argument('--sparse_reward', type=str2bool, default=True, help='whether use sparse reward')
# 	parser.add_argument('--reward_type', type=str, default='sparse', help='reward type')
#
# 	# hyper parameters
# 	parser.add_argument('--buffer_size', type=int, default=100000, help='replay buffer size')
# 	parser.add_argument('--dynamics_buffer_size', type=int, default=100000, help='hyper params')
# 	parser.add_argument('--fake_buffer_size', type=int, default=10000, help='hyper params')
# 	parser.add_argument('--gen_buffer_size', type=int, default=10000, help='hyper params')
# 	parser.add_argument('--dynamic_batchsize', type=int, default=16, help='hyper params')
# 	parser.add_argument('--gen_batchsize', type=int, default=16, help='hyper params')
# 	parser.add_argument('--warmup', type=int, default=2000, help='warm up steps')
# 	parser.add_argument('--coll_r', type=float, default=0.1, help='hgg collision_threshold')
# 	parser.add_argument('--inner_r', type=float, default=0.8, help='hgg inner radius')
# 	parser.add_argument('--outer_r', type=float, default=1.0, help='hgg outer radius')
# 	#parser.add_argument('--buffer_type', type=str, default='energy', help='replay buffer type')
#
# 	# HER parameters
# 	parser.add_argument('--hgg_L', type=int, default=10, help='hyper params')
# 	parser.add_argument('--hgg_c', type=float, default=3.0, help='hyper params')
#
# 	# RIS parameters
# 	parser.add_argument('--her', type=str, default='future', help='her strategy during training')
# 	parser.add_argument('--her_k', type=int, default=4, help='use k experiences for each transition')
#
# 	# model save and load
# 	parser.add_argument('--save_acc', type=float, default=0.0, help='save exp when acc greater than this threshold')
# 	parser.add_argument('--save_episodes', type=int, default=10,
# 						help='save models when acc greater than this threshold')
#
# 	# 添加轨迹优化相关参数
# 	parser.add_argument('--use_waypoints', type=str2bool, default=True, help='使用路径点引导')
# 	parser.add_argument('--waypoint_num', type=int, default=5, help='路径点数量')
# 	parser.add_argument('--path_reward_weight', type=float, default=1.5, help='路径奖励权重')
# 	parser.add_argument('--optimize_trajectory', type=str2bool, default=True, help='是否优化轨迹')
#
# 	args = parser.parse_args()
# 	#args.goal_based = True
#
# 	# args.goal_based = (args.env in Robotics_envs_id)
# 	# args.clip_return_l, args.clip_return_r = clip_return_range(args)
#
# 	logger_name = args.alg+'-'+args.env+'-'+args.learn
# 	if args.tag!='': logger_name = args.tag+'-'+logger_name
# 	args.logger = get_logger(logger_name)
#
# 	for key, value in args.__dict__.items():
# 		if key!='logger':
# 			args.logger.info('{}: {}'.format(key,value))
#
# 	# # create logger
# 	# if args.tag != '':
# 	# 	log_path = './log/' + args.tag
# 	# else:
# 	# 	# 使用简单的时间戳而不是UUID
# 	# 	timestamp = time.strftime("%Y%m%d_%H%M%S")
# 	# 	log_path = './log/run_' + timestamp
# 	#
# 	# # 确保目录存在
# 	# if not os.path.exists(log_path):
# 	# 	os.makedirs(log_path, exist_ok=True)
# 	# if not os.path.exists(log_path + '/plt'):
# 	# 	os.makedirs(log_path + '/plt', exist_ok=True)
# 	# if not os.path.exists(log_path + '/text/log'):
# 	# 	os.makedirs(log_path + '/text/log', exist_ok=True)
# 	# args.log_path = log_path
# 	# args.logger = get_logger(log_path + '/logger.log')
# 	#
# 	# args.logger.info('Arguments: %s', str(args))
#
#
# 	return args
#
# def experiment_setup(args):
# 	# load and wrap the Isaac Gym environment
# 	# env = load_isaacgym_env_preview4(task_name="Ant", num_envs=64)
# 	# env = wrap_env(env)
# 	# env_test = wrap_env(env)
#
#
#
# 	env = wrapper.IsaacGymWrapper(
# 		"panda",
# 		"panda_env",
# 		num_envs=1,
# 		viewer=False,
# 		device="cuda:0",
# 		cube_on_shelf=False,
# 	)
# 	env_test = env
#
# 	#print("---------------------------env.num_envs------------------------", env.num_envs)
# 	# env = make_env(args)
# 	# env_test = make_env(args)
# 	# if args.goal_based:
# 	# 	args.obs_dims = list(goal_based_process(env.reset()).shape)
# 	# 	args.acts_dims = [env.action_space.shape[0]]
# 	#args.compute_reward = env.compute_reward
# 	# args.compute_distance = env.compute_distance
#
# 	# 处理观察空间和动作空间
# 	if args.goal_based:
# 		if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'spaces'):
# 			args.obs_dims = list(env.observation_space.spaces['observation'].shape)
# 			args.acts_dims = list(env.action_space.shape)
# 			args.goal_dims = list(env.observation_space.spaces['desired_goal'].shape)
#
# 			args.obs_dims[0] += args.goal_dims[0]
# 			if hasattr(env, 'compute_reward'):
# 				args.compute_reward = env.compute_reward
# 			if hasattr(env, 'compute_distance'):
# 				args.compute_distance = env.compute_distance
# 		else:
# 			# 对于IsaacGym环境，手动指定维度
# 			args.obs_dims = [9]  # 根据IsaacGym环境调整
# 			args.acts_dims = [3]  # 根据IsaacGym环境调整
# 			args.goal_dims = [3]  # 根据IsaacGym环境调整
#
# 			# 确保提供计算奖励和距离的函数
# 			if not hasattr(args, 'compute_reward'):
# 				args.compute_reward = lambda achieved, goal, info: -float(np.linalg.norm(achieved - goal) > 0.05)
# 			if not hasattr(args, 'compute_distance'):
# 				args.compute_distance = lambda achieved, goal: np.linalg.norm(achieved - goal)
# 	else:
# 		if hasattr(env, 'observation_space'):
# 			args.obs_dims = list(env.observation_space.shape)
# 			args.acts_dims = list(env.action_space.shape)
# 		else:
# 			# 对于IsaacGym环境，手动指定维度
# 			args.obs_dims = [9]  # 根据IsaacGym环境调整
# 			args.acts_dims = [3]  # 根据IsaacGym环境调整
#
#
#
# 	args.buffer = buffer = ReplayBuffer_Episodic(args)
# 	args.learner = learner = create_learner(args)
#
# 	args.agent = agent = create_agent(args)
# 	print("Agent Type：", type(agent))
# 	args.logger.info('*** network initialization complete ***')
# 	args.tester = tester = Tester(args)
# 	args.logger.info('*** tester initialization complete ***')
#
# 	return env, env_test, agent, buffer, learner, tester