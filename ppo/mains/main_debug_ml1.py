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
        tracemalloc.start()
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

        snapshot1 = tracemalloc.take_snapshot()
        trainer.train_agent()
        snapshot2 = tracemalloc.take_snapshot()

        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        print("[ Top 10 differences ]")
        for stat in top_stats[:10]:
                print(stat)


        top_stats = snapshot2.statistics('lineno')

        print("[ Top 10 ]")
        for stat in top_stats[:10]:
                print(stat)

def main():
        opt = type('', (), {})()
        opt.device = 'cuda'
        print(opt.device)
        opt.run_id = 0
        opt.cfg_id = 0
        opt.task = type('', (), {})()
        opt.task.name = 'reach-v2'
        opt.task.max_reward_value = 10
        opt.model = type('', (), {})()
        opt.model.model_from_paper = True
        opt.model.hidden_size = 256
        opt.model.obs_encoding = 'mlp'
        opt.model.obs_encoding_hidden_size = 64
        opt.model.std_plus = 'log_relu'
        opt.model.actor_head_layers = [2,'tanh']
        opt.model.critic_head_layers = [2, 'relu']
        opt.model.input_builder = 'L2RL_input_builder'
        opt.model.base_recurrent = True
        opt.meta_learning = type('', (), {})()
        opt.meta_learning.meta_batch_size = 25
        opt.meta_learning.rollout_max_length = 500
        opt.meta_learning.rollouts_per_task = 1
        opt.rl_algorithm = type('', (), {})()
        opt.rl_algorithm.name = 'PPO'
        opt.rl_algorithm.gamma = 0.99
        opt.rl_algorithm.gradient_clipping = [True, 40]
        opt.rl_algorithm.beta_entropy = [1, True]
        opt.rl_algorithm.beta_entropy_final_value = 0.05
        opt.rl_algorithm.beta_value_function_loss = 0.05
        opt.rl_algorithm.policy_dist_type = 'Normal'
        opt.rl_algorithm.gae = True
        opt.rl_algorithm.gae_lambda = 0.95
        opt.rl_algorithm.n_steps = 500
        opt.rl_algorithm.ppo_epoch=1
        opt.rl_algorithm.num_mini_batch=5
        opt.rl_algorithm.use_proper_time_limits = False
        opt.rl_algorithm.clip_param = 0.01
        opt.rl_algorithm.use_clipped_value_loss = True
        opt.rl_algorithm.act_deterministic = False
        opt.optimizer = type('', (), {})()
        opt.optimizer.name = 'Adam'
        opt.optimizer.args = {'lr':0.001, 'eps':0.00001, 'betas':[0.9, 0.999]}
        opt.training = type('', (), {})()
        opt.training.nr_episodes = 20000
        opt.training.epoch_length = 1
        opt.training.nr_training_epochs = 3
        opt.training.model_checkpoint_frequency = 10
        opt.validation = type('', (), {})()
        opt.validation.nr_validation_episodes = 23
        opt.paths = type('', (), {})()
        opt.paths.tensorboard_logdir = '/root/RL/data/RNN_meta_learning/logs/tensorboard/BernoulliMetaWorld'
        opt.out_dir = './results/2023Feb05-222351_IndependentBandits/0000_rl_algorithm.gradient_clipping_False/0'
        opt.validation = type('', (), {})()
        opt.validation.n_exploration_episodes = 10
        opt.validation.n_test_episodes = 1
        opt.validation.device = 'cpu'
        opt.title = 'rl_algorithm.gradient_clipping=False'
        run(opt)

if __name__ == "__main__":
    main()