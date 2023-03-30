import torch
import tasks_generators
import numpy as np

class ContinuousActionsAgentNormalDist:
    def __init__(self,rollout_buffer,model,config):
        # Set the model and the buffer used to save the observed transitions
        self.model=model
        self.rollout_buffer = rollout_buffer

        self.device = torch.device(config.device) 

        self.observation_size = config.task.observation_size
        self.action_size = config.task.action_size
        self.batch_size = config.meta_learning.meta_batch_size
        self.hidden_size = config.model.hidden_size

        # Load training parameters
        self.episode_length = config.meta_learning.rollout_max_length

        self.task_generator = tasks_generators.VectorizedMetaWorldML1EnvironmentsGymnasium(config=config)
        self.get_new_task()

        self.test_tasks = self.task_generator.create_vectorized_ml1_test_environemnts_gymnasium()
        self.train_tasks = self.task_generator.create_vectorized_ml1_train_environemnts_gymnasium()
        
        self.max_reward_value = config.task.max_reward_value
        self.act_deterministic = config.rl_algorithm.act_deterministic

        self.validation_device = torch.device(config.validation.device)
        self.bootstrap_truncated_state = config.rl_algorithm.bootstrap_truncated_state        
        
    def step(self):
        with torch.no_grad():
            V, self.a_t, action_log_probs, self.rnn_hxs = self.model.act(self.next_inputs, self.rnn_hxs, self.masks, self.act_deterministic)

            env_a_t = torch.clamp(self.a_t,min=-1,max=1).cpu().numpy() 
            self.o_t, self.r_t, self.done, _, _ = self.env.step(env_a_t) # Step the environoment with the sampled action

            # Convert the numpy values returned by the MetaWorld environment to torch tensors used for the input of the LSTM
            self.r_t = (torch.from_numpy(self.r_t).unsqueeze(dim=1) / self.max_reward_value).to(self.device) # simplest normalization for metaworld (r in [0,10])
            self.o_t = torch.from_numpy(self.o_t).to(self.device)

            self.t += 1
            if self.t == self.episode_length: # MetaWorld envs don't signal that the episode should be truncated ....
                self.done = np.ones(self.batch_size,dtype=np.bool)
            self.masks = torch.logical_not(torch.from_numpy(self.done)).unsqueeze(1).to(self.device)
            self.bad_masks = torch.ones_like(self.masks) # in metaworld there is no truncation only termination -> should be torch.logical_not(terminated) for env with termination condition 


            if torch.any(self.o_t>1e4):
                print(f'Last observation: {self.o_t}')
                raise Exception("Observation values are too big!")

            self.next_inputs = self.model.input_builder(observation=self.o_t,action=self.a_t,reward=self.r_t,t=self.t/self.episode_length)
            self.rollout_buffer.insert(self.next_inputs, self.rnn_hxs, self.a_t, action_log_probs, V, self.r_t, self.masks, self.bad_masks)
        

    def run_steps(self, nr_steps):
        for _ in range(nr_steps):
            self.step()
        if self.t == self.episode_length:
            return torch.zeros_like(self.r_t)
        else:
            with torch.no_grad():
                V = self.model.get_value(self.next_inputs,self.rnn_hxs,self.rollout_buffer.masks[-1])
            return V

    def run_steps_until_done(self):
        while not np.any(self.done):
            self.step()

        if self.bootstrap_truncated_state:
            with torch.no_grad():
                V, _, _, _ = self.model.act(self.next_inputs, self.rnn_hxs, self.masks, self.act_deterministic)
            return V
        else:
            return torch.zeros_like(self.r_t)

    def get_new_task(self):
        self.env = self.task_generator.change_tasks()
        self.reset_environments()
        self.reset_model_hidden_state() # Reset RNN hidden state

    def reset_environments(self):
        #self.env.reset() # metaworld envs v2 keep the last state from the previous episode in the first observation of the new episode -> calling env.reset twice mitigates this
        self.o_t, _ = self.env.reset()  # Reset environment
        self.o_t = torch.from_numpy(self.o_t).to(self.device)
        self.a_t = torch.zeros((self.batch_size,self.action_size),dtype=torch.long,device=self.device)
        self.r_t = torch.zeros((self.batch_size,1),device=self.device)
        self.t = 0
        with torch.no_grad():
            self.next_inputs = self.model.input_builder(observation=self.o_t,action=self.a_t,reward=self.r_t,t=self.t/self.episode_length)

        self.done = np.zeros(self.batch_size,dtype=np.bool_)
        self.masks = torch.ones((self.batch_size,1),device=self.device)
        self.bad_masks = torch.ones((self.batch_size,1),device=self.device)

    def store_initial_state(self):
        self.rollout_buffer.after_reset(self.next_inputs,self.rnn_hxs)
    
    def detach_model_hidden_state(self):
        self.rnn_hxs = self.rnn_hxs.detach()

    def reset_model_hidden_state(self):
        self.rnn_hxs = torch.zeros((self.batch_size,self.hidden_size),device=self.device)

    def run_episodes(self,tasks_set,n_exploration_eps=10,n_test_eps=1,return_trajectories=False):
        if tasks_set == 'train':
            tasks = self.train_tasks
        elif tasks_set == 'test':
            tasks = self.test_tasks
        tasks.reset()
        nr_tasks = tasks.action_space.shape[0]
        if return_trajectories:
            # observations = []
            # rewards = []
            # infos = []
            # dones = []
            actions = []
            # pis = []
            Vs = [] 
        with torch.no_grad():    
            rnn_hxs = torch.zeros((nr_tasks,self.hidden_size),device=self.validation_device)
            for eps in range(n_exploration_eps+n_test_eps):
                o_t, _ = tasks.reset()  # Reset environment
                o_t = torch.from_numpy(o_t).to(self.validation_device)
                a_t = torch.zeros((nr_tasks,self.action_size),dtype=torch.long,device=self.validation_device)
                r_t = torch.zeros((nr_tasks,1),device=self.validation_device)
                done = np.zeros(nr_tasks,dtype=np.bool_)
                masks = torch.ones((nr_tasks,1),device=self.validation_device)
                return_ = 0
                for t in range(self.episode_length):
                    inputs = self.model.input_builder(observation=o_t,action=a_t,reward=r_t,t=t/self.episode_length)
                    V, a_t, _, rnn_hxs = self.model.act(inputs, rnn_hxs, masks, self.act_deterministic)
                    
                    env_a_t = torch.clamp(a_t,min=-1,max=1).cpu().numpy()
                    o_t, r_t, done, _, info = tasks.step(env_a_t) # Step the environoment with the sampled action
                    masks = torch.logical_not(torch.from_numpy(done)).unsqueeze(1).to(self.validation_device)

                    if eps >= n_exploration_eps and return_trajectories:
                        actions.append(a_t)
                        Vs.append(V)
                    # Convert the numpy values returned by the MetaWorld environment to torch tensors used for the input of the LSTM
                    r_t = (torch.from_numpy(r_t).unsqueeze(dim=1) / self.max_reward_value).to(self.validation_device) # simplest normalization for metaworld (r in [0,10])
                    o_t = torch.from_numpy(o_t).to(self.validation_device)
                    if eps >= n_exploration_eps:
                        return_ += r_t
        if return_trajectories:
            return return_.mean().item(), info['success'].mean().item()*100, actions, Vs
        else:
            return return_.mean().item(), info['success'].mean().item()*100