import sys
import os 
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)

import actors
import trainers
import a2c_learners
import replay_buffers
import models
import os
import metaworld
from utils import set_all_seeds
from liftoff import parse_opts

def run(opt):
        set_all_seeds(opt.run_id)
        if not os.path.isdir(os.path.join('/root/RL/data/RNN_meta_learning/utils',opt.task.name)):
                os.mkdir(os.path.join('/root/RL/data/RNN_meta_learning/utils',opt.task.name))
        
        # initialize data regarding the selected task
        ml1 = metaworld.ML1(opt.task.name) 
        env = ml1.train_classes[opt.task.name]()  
        opt.task.action_size = env.action_space.shape[0]
        opt.task.action_high = env.action_space.high
        opt.task.action_low = env.action_space.low
        opt.task.episode_length = env.max_path_length
        opt.task.observation_size = env.observation_space.shape[0]
        del ml1, env
        
        if opt.model.init == 'orthogonal':
                model = models.A2C_LSTM_Gaussian_OrthogonalInit(observation_size=opt.task.observation_size,action_size=opt.task.action_size,
                                                        hidden_size=opt.model.hidden_size,batch_size=opt.meta_learning.meta_batch_size)
        elif opt.model.init == 'default':
                model = models.A2C_LSTM_Gaussian(observation_size=opt.task.observation_size,action_size=opt.task.action_size,
                                                        hidden_size=opt.model.hidden_size,batch_size=opt.meta_learning.meta_batch_size)
        rollout_buffer = replay_buffers.RolloutBufferNormalDist()
        learner = a2c_learners.A2CLearnerMetaWorldNormalDist(rollout_buffer=rollout_buffer,model=model,config=opt)
        actor = actors.ContinuousActionsAgentNormalDistSingleTask(rollout_buffer=rollout_buffer,model=model,config=opt)
        trainer = trainers.TrainerMetaWorldSingleTask(actor=actor,learner=learner,config=opt)

        trainer.train_agent()

def main():
    run(parse_opts())

if __name__ == "__main__":
    main()