from os.path import join, exists
import torch
from torch.utils.tensorboard import SummaryWriter
import gc
import utils


######################################################### MetaWorld ################################################################################
class TrainerMetaWorld:
    def __init__(self,actor,learner,rollout_buffer,config):
        self.actor = actor
        self.learner = learner
        self.rollout_buffer = rollout_buffer
        
        self.epoch_length = config.training.epoch_length
        self.nr_training_epochs = config.training.nr_training_epochs
        self.nr_episodes = config.training.nr_episodes

        self.lr_schedule = config.optimizer.lr_schedule
        self.initial_lr = config.optimizer.args['lr']
        print(f'initial_lr = {self.initial_lr}')

        self.data_dir = config.out_dir
        tensorboard_dir = config.paths.tensorboard_logdir
        print(tensorboard_dir)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        self.run_id = config.title + '; seed ' + str(config.run_id)

        self.model_checkpoint_frequency = config.training.model_checkpoint_frequency
        self.episode_length = config.meta_learning.rollout_max_length
        self.rollouts_per_task = config.meta_learning.rollouts_per_task
        self.rollout_max_length = config.meta_learning.rollout_max_length
        self.n_steps = config.rl_algorithm.n_steps
        self.action_size = config.task.action_size

        self.use_gae = config.rl_algorithm.gae
        self.gamma = config.rl_algorithm.gamma
        self.gae_lambda = config.rl_algorithm.gae_lambda
        self.use_proper_time_limits = config.rl_algorithm.use_proper_time_limits

        self.n_exploration_episodes = config.validation.n_exploration_episodes
        self.n_test_episodes = config.validation.n_test_episodes
        self.val_device = torch.device(config.validation.device)
        self.device = torch.device(config.device)



    def train_agent(self) -> list:
        epoch = 0 
        episode = 0
        returns_train = []
        returns_test = []
        success_rates_train = []
        success_rates_test = []
        entropy_regularization = []
        actor_loss = []
        critic_loss = []

        while epoch < self.nr_training_epochs:
            if episode % self.rollouts_per_task == 0:
                self.actor.get_new_task()
                self.actor.store_initial_state()
            else:
                self.actor.reset_environments()
                self.actor.detach_model_hidden_state()
                self.actor.store_initial_state()
            R = self.actor.run_steps_until_done()
            with torch.no_grad():
                self.rollout_buffer.compute_returns(next_value=R,use_gae=self.use_gae,gamma=self.gamma,gae_lambda=self.gae_lambda,use_proper_time_limits=self.use_proper_time_limits)

            value_loss_epoch, action_loss_epoch, dist_entropy_epoch = self.learner.update(self.rollout_buffer)
            entropy_regularization.append(dist_entropy_epoch)
            actor_loss.append(action_loss_epoch)
            critic_loss.append(value_loss_epoch)
            episode += 1
            if self.lr_schedule:
                utils.update_linear_schedule(self.learner.optimizer, episode, self.nr_episodes, self.initial_lr)

            if episode%10==0:
                gc.collect()
                torch.cuda.empty_cache()

            if episode%self.epoch_length==0:
                epoch += 1
                self.actor.model.to(self.val_device) # move actor_critic model to validation device
                print(f'Epoch {epoch} done.')

                actor_loss = sum(actor_loss)/len(actor_loss)
                self.writer.add_scalar('%s/actor_loss'%(self.run_id),actor_loss,epoch)
                critic_loss = sum(critic_loss)/len(critic_loss)
                self.writer.add_scalar('%s/critic_loss'%(self.run_id),critic_loss,epoch)
                entropy_regularization = sum(entropy_regularization)/len(entropy_regularization)
                self.writer.add_scalar('%s/entropy_regularization'%(self.run_id),entropy_regularization,epoch)
                entropy_regularization = []
                actor_loss = []
                critic_loss = []
                
                return_, success_rate, actions, Vs = self.actor.run_episodes('train',n_exploration_eps=self.n_exploration_episodes,
                                                                             n_test_eps=self.n_test_episodes,return_trajectories=True) 
                returns_train.append(return_)
                success_rates_train.append(success_rate)
                self.writer.add_scalar('%s/train_return'%(self.run_id),returns_train[-1],epoch)
                self.writer.add_scalar('%s/train_success_rate'%(self.run_id),success_rates_train[-1],epoch)

                actions = torch.flatten(torch.stack(actions)).cpu()
                # pis = torch.stack(pis).cpu()
                Vs = torch.flatten(torch.stack(Vs)).cpu()
                # means, stds = torch.split(pis, self.action_size, -1)
                # means = torch.flatten(means)
                # stds = torch.flatten(stds)
                self.writer.add_histogram('%s/action_distribution'%(self.run_id),actions,epoch)
                # self.writer.add_histogram('%s/mean_distribution'%(self.run_id),means,epoch)
                # self.writer.add_histogram('%s/std_distribution'%(self.run_id),stds,epoch)
                self.writer.add_histogram('%s/V_distribution'%(self.run_id),Vs,epoch)
                
                return_, success_rate, actions, Vs = self.actor.run_episodes('test',n_exploration_eps=self.n_exploration_episodes,
                                                                             n_test_eps=self.n_test_episodes,return_trajectories=True) 
                returns_test.append(return_)
                success_rates_test.append(success_rate)
                self.writer.add_scalar('%s/test_return'%(self.run_id),returns_test[-1],epoch)
                self.writer.add_scalar('%s/test_success_rate'%(self.run_id),success_rates_test[-1],epoch)

                actions = torch.flatten(torch.stack(actions)).cpu()
                # pis = torch.stack(pis).cpu()
                Vs = torch.flatten(torch.stack(Vs)).cpu()
                # means, stds = torch.split(pis, self.action_size, -1)
                # means = torch.flatten(means)
                # stds = torch.flatten(stds)
                self.writer.add_histogram('%s/action_distribution_test'%(self.run_id),actions,epoch)
                # self.writer.add_histogram('%s/mean_distribution_test'%(self.run_id),means,epoch)
                # self.writer.add_histogram('%s/std_distribution_test'%(self.run_id),stds,epoch)
                self.writer.add_histogram('%s/V_distribution_test'%(self.run_id),Vs,epoch)

                torch.save({'returns_train':returns_train,'returns_test':returns_test,'success_rates_train':success_rates_train,
                        'successs_rates_test':success_rates_test},self.data_dir+'/loss_and_return.pt')
                if epoch%self.model_checkpoint_frequency==0:
                    torch.save((self.learner.actor_critic.state_dict(),self.learner.optimizer.state_dict()),join(self.data_dir,'model_epoch_'+str(epoch)+'.pt'))

                del actions, Vs
                self.actor.model.to(self.device) # move actor_critic model to training device

        # Save the final results
        torch.save({'returns_train':returns_train,'returns_test':returns_test,'success_rates_train':success_rates_train,
                        'success_rates_test':success_rates_test},self.data_dir+'/loss_and_return.pt')
        torch.save((self.learner.actor_critic.state_dict(),self.learner.optimizer.state_dict()),join(self.data_dir,'model_epoch_'+str(epoch)+'.pt'))
        
        return returns_train, returns_test