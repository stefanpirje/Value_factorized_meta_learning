import torch
import metaworld
import random
import numpy as np
import gym


class VectorizedMetaWorldML1EnvironmentsSingleTask:
    def __init__(self,config):
        self.task_name = config.task.name
        self.nr_tasks = config.meta_learning.meta_batch_size

        self.ml1 = metaworld.ML1(self.task_name) # Construct the benchmark, sampling tasks
        self.envs = []
        for _ in range(self.nr_tasks):
            env = self.ml1.train_classes[self.task_name]()  # Create an environment with selected task 
            self.envs.append(env)
        self.task = random.choice(self.ml1.train_tasks)

    def generate_task(self, task, env_idx:int):
        self.envs[env_idx].set_task(task)
        return self.envs[env_idx]

    def change_tasks(self):
        env_fns = [] 
        for i in range(self.nr_tasks):
            env_fns.append(lambda: self.generate_task(task=self.task,env_idx=i))
        tasks = gym.vector.AsyncVectorEnv(env_fns=env_fns)
        return tasks