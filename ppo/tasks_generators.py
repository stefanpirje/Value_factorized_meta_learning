import torch
import metaworld
import random
import numpy as np
import gym
import gymnasium
from gymnasium.wrappers import TransformObservation, EnvCompatibility
from gymnasium.spaces import Box 

class VectorizedMetaWorldML1EnvironmentsGymnasium:
    def __init__(self,config):
        self.task_name = config.task.name
        self.nr_tasks = config.meta_learning.meta_batch_size

        self.ml1 = metaworld.ML1(self.task_name, seed=config.cfg_id) # Construct the benchmark, sampling tasks
        self.envs = []
        for _ in range(self.nr_tasks):
            env = self.ml1.train_classes[self.task_name]()  # Create an environment with selected task 
            self.envs.append(env)

    def generate_task(self, task, env_idx:int):
        self.envs[env_idx].set_task(task)
        compatible_env = TransformObservation(
                            EnvCompatibility(self.envs[env_idx]),
                            lambda obs: torch.from_numpy(obs),
        )
        _ = compatible_env.observation_space
        compatible_env.observation_space = Box(low=_.low, high=_.high, shape=_.shape, dtype=_.dtype)
        _ = compatible_env.action_space
        compatible_env.action_space = Box(low=_.low, high=_.high, shape=_.shape, dtype=_.dtype)
        return compatible_env

    def change_tasks(self):
        env_fns = [] 
        tasks = random.sample(self.ml1.train_tasks, self.nr_tasks)
        for i in range(self.nr_tasks):
            env_fns.append(lambda: self.generate_task(task=tasks[i],env_idx=i))
        tasks = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
        return tasks

    def create_vectorized_ml1_test_environemnts_gymnasium(self):
        tasks = self.ml1.test_tasks

        envs = []
        env_fns = []
        for i, task in enumerate(tasks):
            env = self.ml1.test_classes[self.task_name]()  # Create an environment with selected task 
            envs.append(env)
            envs[i].set_task(task)
            envs[i] = TransformObservation(
                                EnvCompatibility(envs[i]),
                                lambda obs: torch.from_numpy(obs),
            )
            _ = envs[i].observation_space
            envs[i].observation_space = Box(low=_.low, high=_.high, shape=_.shape, dtype=_.dtype)
            _ = envs[i].action_space
            envs[i].action_space = Box(low=_.low, high=_.high, shape=_.shape, dtype=_.dtype)
            env_fns.append(lambda: envs[i])
        tasks = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
        
        return tasks 

    def create_vectorized_ml1_train_environemnts_gymnasium(self):
        tasks = self.ml1.train_tasks

        envs = []
        env_fns = []
        for i, task in enumerate(tasks):
            env = self.ml1.train_classes[self.task_name]()  # Create an environment with selected task 
            envs.append(env)
            envs[i].set_task(task)
            envs[i] = TransformObservation(
                                EnvCompatibility(envs[i]),
                                lambda obs: torch.from_numpy(obs),
            )
            _ = envs[i].observation_space
            envs[i].observation_space = Box(low=_.low, high=_.high, shape=_.shape, dtype=_.dtype)
            _ = envs[i].action_space
            envs[i].action_space = Box(low=_.low, high=_.high, shape=_.shape, dtype=_.dtype)
            env_fns.append(lambda: envs[i])
        tasks = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
        
        return tasks 