# import sys
# sys.path.append('/home/my/Hindsight-Goal-Generation-master4/Hindsight-Goal-Generation-master')
from isaacgym import gymtorch
import torch, hydra, zerorpc
from m3p2i_aip.planners.motion_planner import m3p2i
from m3p2i_aip.planners.task_planner import task_planner
from m3p2i_aip.config.config_store import ExampleConfig
import  m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from m3p2i_aip.planners.motion_planner.cost_functions1 import Objective

from m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

import sys
sys.path.append('/home/my/Hindsight-Goal-Generation-master4/Hindsight-Goal-Generation-master/src/m3p2i_aip/planners/mlp_learn')
from sdf.robot_sdf import RobotSdfCollisionNet

import numpy as np

print("**************", torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__version__)


'''
Run in the command line:
    python3 reactive_tamp.py task=navigation goal="[-3, 3]"
    python3 reactive_tamp.py task=push goal="[-1, -1]"
    python3 reactive_tamp.py task=pull goal="[0, 0]"
    python3 reactive_tamp.py task=push_pull multi_modal=True goal="[-3.75, -3.75]"
    python3 reactive_tamp.py -cn config_panda
    python3 reactive_tamp.py -cn config_panda multi_modal=True cube_on_shelf=True
'''

# define tensor parameters (cpu or cuda:0)
# if 1:
#     params = {'device': 'cpu', 'dtype': torch.float32}
# else:
#     params = {'device': 'cuda:0', 'dtype': torch.float32}
params = {'device': 'cuda:0', 'dtype': torch.float32}
#params = {'device': 'cpu', 'dtype': torch.float32}

class REACTIVE_TAMP:
    def __init__(self, cfg) -> None:
        self.sim = wrapper.IsaacGymWrapper(
            cfg.isaacgym,
            cfg.env_type,
            num_envs=cfg.mppi.num_samples,
            viewer=False,
            device=cfg.mppi.device,
            cube_on_shelf=cfg.cube_on_shelf,
        )
        #self.sim = env
        self.cfg = cfg

        # DOF = 7
        # L = 1
        # # Load nn model
        # s = 256
        # n_layers = 5
        # skips = []
        # fname = '%ddof_sdf_%dx%d_mesh.pt' % (DOF, s, n_layers)

        DOF = 7
        L = 1

        # Load nn model
        s = 256
        n_layers = 5
        skips = []
        fname = '%ddof_sdf_%dx%d_mesh.pt' % (DOF, s, n_layers) # planar robot
        #fname = 'franka_%dx%d.pt' % (s, n_layers)  # franka robot
        #fname = '/home/my/m3p2i-aip-master/src/m3p2i_aip/planners/mlp_learn/models/7dof_sdf_256x5_mesh.pt'

        if skips == []:
            n_layers -= 1

        nn_model = RobotSdfCollisionNet(in_channels=DOF + 3, out_channels=DOF, layers=[s] * n_layers, skips=skips)
        print("-----------------------------------1")
        #nn_model.load_weights('../../mlp_learn/models/' + fname, params)
        #nn_model.load_weights('/home/my/m3p2i-aip-master/src/m3p2i_aip/planners/mlp_learn/models/' + fname, params)
        nn_model.load_weights('/home/my/Hindsight-Goal-Generation-master4/Hindsight-Goal-Generation-master/src/m3p2i_aip/planners/mlp_learn/models/' + fname, params)
        print("-----------------------------------2")
        nn_model.model.to(**params)
        # prepare models: standard (used for AOT implementation), jit, jit+quantization
        nn_model.model_jit = nn_model.model

        # nn_model.model_jit_q = torch.quantization.quantize_dynamic(
        #     nn_model.model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        # )

        nn_model.model_jit = torch.jit.script(nn_model.model_jit)
        nn_model.model_jit = torch.jit.optimize_for_inference(nn_model.model_jit)

        # nn_model.model_jit_q = torch.jit.script(nn_model.model_jit)
        # nn_model.model_jit_q = torch.jit.optimize_for_inference(nn_model.model_jit)

        nn_model.update_aot_lambda()

        self.objective = Objective(cfg, nn_model)
        #self.objective = Objective(cfg)

        self.task_planner = task_planner.set_task_planner(cfg)
        self.task_success = False
        
        self.motion_planner = m3p2i.M3P2I(
            cfg,
            dynamics=self.dynamics, 
            running_cost=self.running_cost
        )



    def run_tamp(self, dof_state, root_state):

        # Set rollout state from sim state
        self.sim._dof_state[:] = bytes_to_torch(dof_state)
        self.sim._root_state[:] = bytes_to_torch(root_state)
        self.sim.set_dof_state_tensor(self.sim._dof_state)
        self.sim.set_actor_root_state_tensor(self.sim._root_state)

        self.tamp_interface()
        
        if self.task_success:
            print("--------Task success--------")
            return torch_to_bytes(
                torch.zeros(self.sim.dofs_per_robot, device=self.cfg.mppi.device)
            )
        else:
            #print("--------Compute optimal action--------")
            return torch_to_bytes(
                self.motion_planner.command(self.sim._dof_state[0])[0]
            )
    
    def dynamics(self, _, u, t=None):
        self.sim.set_dof_velocity_target_tensor(u)
        self.sim.step1()
        states = torch.stack([self.sim.robot_pos[:, 0], 
                              self.sim.robot_vel[:, 0], 
                              self.sim.robot_pos[:, 1], 
                              self.sim.robot_vel[:, 1]], dim=1) # [num_envs, 4]
        return states, u
    
    def running_cost(self, _):
        return self.objective.compute_cost(self.sim)
    
    def tamp_interface(self):
        # self.task_planner.update_plan(self.sim)
        #print("task", self.task_planner.task, "goal", self.task_planner.curr_goal)
        # self.motion_planner.update_gripper_command(self.task_planner.task)
        self.objective.update_objective(self.task_planner.task, self.task_planner.curr_goal)
        self.suction_active = self.motion_planner.get_pull_preference()
        # self.task_success = self.task_planner.check_task_success(self.sim)

    def get_trajs(self):
        return torch_to_bytes(self.motion_planner.top_trajs)
    
    def get_suction(self):
        return torch_to_bytes(self.suction_active)



@hydra.main(version_base=None, config_path="/home/my/Hindsight-Goal-Generation-master4/Hindsight-Goal-Generation-master/m3p2i_aip/config", config_name="config_point")
def run_reactive_tamp(cfg: ExampleConfig):
    reactive_tamp = REACTIVE_TAMP(cfg)
    planner = zerorpc.Server(reactive_tamp)
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()

if __name__== "__main__":
    run_reactive_tamp()