import numpy as np
import time
import hydra
from omegaconf import DictConfig
from common import get_args, experiment_setup
# from scripts.reactive_tamp import REACTIVE_TAMP
# from scripts.sim import run_sim
from m3p2i_aip.config.config_store import ExampleConfig

import json
import logging
import numpy as np

import torch, hydra, zerorpc

import os

from algorithm.data_sampler import Data_Sampler

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarray(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, list) and all(isinstance(i, torch.Tensor) for i in x):
        return [i.tolist() for i in obj]

    return obj

#@hydra.main(version_base=None, config_path="/home/my/Hindsight-Goal-Generation-master4/Hindsight-Goal-Generation-master/src/m3p2i_aip/config", config_name="config_panda")
def main():
    args = get_args()
    env, env_test, agent, bc_diffusion_agent, ql_diffusion_agent, buffer, learner, tester = experiment_setup(args)

    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("Server found and wait for the viewer")

    total_episodes = args.episodes
    exploration_phase = int(total_episodes * 0.3)
    training_phase = int(total_episodes * 0.6)
    fine_tuning_phase = total_episodes

    training_state = {
        "total_episodes": total_episodes,
        "exploration_phase": exploration_phase,
        "training_phase": training_phase,
        "fine_tuning_phase": fine_tuning_phase,
        "total_epochs": args.epochs,
    }
    
    args.cycles = 20

    for epoch in range(args.epochs):
        training_state["current_epoch"] = epoch

        for cycle in range(args.cycles):
            print("*************************epoch***********************", epoch, args.epochs)
            print("*********************************cycle*******************************", cycle, args.cycles)
            args.logger.tabular_clear()
            start_time = time.time()

            learner.learn(args, env, env_test, agent, bc_diffusion_agent, ql_diffusion_agent, buffer, planner, training_state)

            log_entry = {
                "epoch": epoch,
                "cycle": cycle,
                "initial_goals": convert_ndarray(learner.initial_goals),
                                                                         "desired_goals": convert_ndarray(
                    learner.desired_goals),
                "explore_goals": convert_ndarray(learner.explore_goals),
                "trajectories": convert_ndarray(learner.achieved_trajectories),
                "success_trajectories": convert_ndarray(learner.success_trajectories),
                "episode_return": convert_ndarray(learner.episode_return),
            }
            with open("explore_goals60.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if hasattr(ql_diffusion_agent, "actor") and epoch>1:
                # >>> save model <<<
                model_save_path = f"saved_models_diffusion_QL46/actor_epoch_{epoch}_cycle_{cycle}.pth"
                os.makedirs("saved_models_diffusion_QL46", exist_ok=True)
                torch.save(ql_diffusion_agent.actor.state_dict(), model_save_path)
                print(f"model saved successfully: {model_save_path}")

            if hasattr(ql_diffusion_agent, "critic") and epoch>1:
                # >>> save model <<<
                model_save_path = f"saved_models_diffusion_QL46/critic_epoch_{epoch}_cycle_{cycle}.pth"
                os.makedirs("saved_models_diffusion_QL46", exist_ok=True)
                torch.save(ql_diffusion_agent.critic.state_dict(), model_save_path)
                print(f"model saved successfully: {model_save_path}")

        #tester.epoch_summary()

    tester.final_summary()

if __name__ == "__main__":
    main()

