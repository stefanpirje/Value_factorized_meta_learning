import sys
import os 
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
import actors
import trainers
import ppo
import storage
from model import Policy
from liftoff import parse_opts
import metaworld
from utils import set_all_seeds
import torch
import tracemalloc

def run(opt):
        set_all_seeds(opt.run_id)
        
        # initialize data regarding the selected task
        ml1 = metaworld.ML1(opt.task.name) 
        env = ml1.train_classes[opt.task.name]()  
        opt.task.action_size = env.action_space.shape[0]
        opt.task.action_high = env.action_space.high
        opt.task.action_low = env.action_space.low
        opt.task.episode_length = env.max_path_length
        opt.task.obs_shape = env.observation_space.shape
        opt.task.observation_size = env.observation_space.shape[0]
        opt.task.action_space = env.action_space
        if opt.model.obs_encoding == 'identity':
            opt.model.input_size = opt.task.observation_size + opt.task.action_size + 2
        else:
            opt.model.input_size = opt.model.obs_encoding_hidden_size + opt.task.action_size + 2
        del ml1, env
        # set epsilon for the succes rate metric according to table from appendix 12


        device = torch.device(opt.device)  
        actor_critic = Policy(config=opt,base_kwargs={'recurrent':opt.model.base_recurrent, 'hidden_size':opt.model.hidden_size}).to(device)                                     
        rollout_buffer = storage.RolloutStorage(config=opt)
        rollout_buffer.to(device)
        learner = ppo.PPO(actor_critic=actor_critic,config=opt)
        actor = actors.ContinuousActionsAgentNormalDist(rollout_buffer=rollout_buffer,model=actor_critic,config=opt)
        trainer = trainers.TrainerMetaWorld(actor=actor,learner=learner,rollout_buffer=rollout_buffer,config=opt)

        trainer.train_agent()

def main():
    run(parse_opts())

if __name__ == "__main__":
    main()