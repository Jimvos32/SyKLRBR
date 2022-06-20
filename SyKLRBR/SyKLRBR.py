import os, gym, time, sys, random, itertools
import numpy as np
import tensorflow as tf
from collections import defaultdict
from memory_profiler import profile
from tensorflow.saved_model import simple_save

from sacred import Experiment
from sacred.observers import FileStorageObserver

SyKLRBR_DATA_DIR = "data/SyKLRBR_runs/"

ex = Experiment('SyKLRBR')
ex.observers.append(FileStorageObserver.create(SyKLRBR_DATA_DIR + "SyKLRBR_exps"))


from overcooked_ai_py.utils import profile, load_pickle, save_pickle, save_dict_to_file, load_dict_from_file
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent

from human_aware_rl.utils import create_dir_if_not_exists, delete_dir_if_exists, reset_tf, set_global_seed
from human_aware_rl.baselines_utils import create_model, get_vectorized_gym_env, update_model, get_agent_from_model,\
save_baselines_model, overwrite_model, load_baselines_model, get_agent_from_saved_model, LinearAnnealer


class PBTAgent(object):
    """An agent that can be saved and loaded and all and the main data it contains is the self.model
    
    Goal is to be able to pass in save_locations or PBTAgents to workers that will load such agents
    and train them together.
    """
    
    def __init__(self, agent_name, start_params, start_logs=None, model=None, gym_env=None, top=False):
        self.params = start_params

        if top:
            self.logs = start_logs if start_logs is not None else {
                "agent_name": agent_name,
                "avg_rew_per_step": [],
                "params_hist": defaultdict(list),
                "num_ppo_runs": 0,
                "reward_shaping": [],
                "opponent": [],
                "distribution": [0]*5
            }
        else:
            self.logs = start_logs if start_logs is not None else {
                "agent_name": agent_name,
                "avg_rew_per_step": [],
                "params_hist": defaultdict(list),
                "num_ppo_runs": 0,
                "reward_shaping": [],
            }
        with tf.device('/device:GPU:{}'.format(self.params["GPU_ID"])):
            self.model = model if model is not None else create_model(gym_env, agent_name, **start_params)

    @property
    def num_ppo_runs(self):
        return self.logs["num_ppo_runs"]
    
    @property
    def agent_name(self):
        return self.logs["agent_name"]

    def get_agent(self):
        return get_agent_from_model(self.model, self.params["sim_threads"])

    def update(self, gym_env):
        with tf.device('/device:GPU:{}'.format(self.params["GPU_ID"])):
            train_info = update_model(gym_env, self.model, **self.params)
            for k, v in train_info.items():
                if k not in self.logs.keys():
                    self.logs[k] = []
                self.logs[k].extend(v)
            self.logs["num_ppo_runs"] += 1

        return train_info

    def update_avg_rew_per_step_logs(self, avg_rew_per_step_stats):
        self.logs["avg_rew_per_step"] = avg_rew_per_step_stats

    def save(self, save_folder):
        """Save agent model, logs, and parameters"""
        create_dir_if_not_exists(save_folder)
        save_baselines_model(self.model, save_folder)
        save_dict_to_file(dict(self.logs), save_folder + "logs")
        save_dict_to_file(self.params, save_folder + "params")

    @staticmethod
    def from_dir(load_folder):
        logs = load_dict_from_file(load_folder + "logs")
        agent_name = logs["agent_name"]
        params = load_dict_from_file(load_folder + "params")
        model = load_baselines_model(load_folder, agent_name)
        return PBTAgent(agent_name, params, start_logs=logs, model=model)

    @staticmethod
    def update_from_files(file0, file1, gym_env, save_dir):
        pbt_agent0 = PBTAgent.from_dir(file0)
        pbt_agent1 = PBTAgent.from_dir(file1)
        gym_env.other_agent = pbt_agent1
        pbt_agent0.update(gym_env)
        return pbt_agent0

    def save_predictor(self, save_folder):
        """Saves easy-to-load simple_save tensorflow predictor for agent"""
        simple_save(
            tf.get_default_session(),
            save_folder,
            inputs={"obs": self.model.act_model.X},
            outputs={
                "action": self.model.act_model.action, 
                "value": self.model.act_model.vf,
                "action_probs": self.model.act_model.action_probs
            }
        )

    def update_pbt_iter_logs(self):
        for k, v in self.params.items():
            self.logs["params_hist"][k].append(v)
        self.logs["params_hist"] = dict(self.logs["params_hist"])

    def explore_from(self, best_training_agent):
        overwrite_model(best_training_agent.model, self.model)
        self.logs["num_ppo_runs"] = best_training_agent.num_ppo_runs
        self.params = self.mutate_params(best_training_agent.params)

    def mutate_params(self, params_to_mutate):
        params_to_mutate = params_to_mutate.copy()
        for k in self.params["HYPERPARAMS_TO_MUTATE"]:
            if np.random.random() < params_to_mutate["RESAMPLE_PROB"]:
                mutation = np.random.choice(self.params["MUTATION_FACTORS"])
                
                if k == "LAM": 
                    # Move eps/2 in either direction
                    eps = min(
                        (1 - params_to_mutate[k]) / 2,      # If lam is > 0.5, avoid going over 1
                        params_to_mutate[k] / 2             # If lam is < 0.5, avoid going under 0
                    )
                    rnd_direction = (-1)**np.random.randint(2) 
                    mutation = rnd_direction * eps
                    params_to_mutate[k] = params_to_mutate[k] + mutation
                elif type(params_to_mutate[k]) is int:
                    params_to_mutate[k] = max(int(params_to_mutate[k] * mutation), 1)
                else:
                    params_to_mutate[k] = params_to_mutate[k] * mutation
                    
                print("Mutated {} by a factor of {}".format(k, mutation))

        print("Old params", self.params)
        print("New params", params_to_mutate)
        return params_to_mutate

@ex.config
def my_config():

    ##################
    # GENERAL PARAMS #
    ##################

    print("custom params")
    TIMESTAMP_DIR = True
    EX_NAME = "SyKLRBR_"

    SAVE_DIR = SyKLRBR_DATA_DIR + time.strftime('%Y_%m_%d-%H_%M_%S_') + "/"

    RUN_TYPE = "pbt"

    M_PARAMS = dict(
            sim_threads=15,                       
            total_steps_per_agent = 5e6,
            num_selection_games = 6,
            ppo_run_tot_timesteps = 12000,      #ppo2.py liner 165 166 first one is total_timesteps (balance sim : nsteps total steps)
            total_batch_size=6000,              #nsteps (total_batch / simthreads)!!!!  1:1
            minibatches=8,                      #nminibatches
            steps_per_update=8,                 #noptecpoch
            gradient_update=15000,
            testing=False,
            minimal = False,
            max_epoch=2,
            rew_swap = 3e6
    )
    # pbt = dict(
    #         sim_threads=20,                            
    #         total_steps_per_agent = 15000000,
    #         num_selection_games = 10,
    #         ppo_run_tot_timesteps = 40000,
    #         total_batch_size=20000,
    #         minibatches=5,
    #         steps_per_update=8,
    # )

    # ppo = dict(
    #         sim_threads=20,                            
    #         total_steps_per_agent = 15000000,
    #         num_selection_games = 10,
    #         ppo_run_tot_timesteps = 5000000,
    #         total_batch_size=12000,
    #         minibatches=6,
    #         steps_per_update=8,
    # )



    # Reduce parameters to be able to run locally to test for simple bugs
    LOCAL_TESTING = False

    # GPU id to use
    GPU_ID = 0

    # List of seeds to run
    SEEDS = [8015]

    # Number of parallel environments used for simulating rollouts
    sim_threads = M_PARAMS["sim_threads"]
   
    ##############
    # PBT PARAMS #
    ##############
    print(tf.__version__)
    if tf.test.gpu_device_name(): 

        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

    else:

        print("Please install GPU version of TF")

    TOTAL_STEPS_PER_AGENT = M_PARAMS["total_steps_per_agent"]
        
    POPULATION_SIZE = 5
    
    ITER_PER_SELECTION = 5 # How many pairings and model training updates before the worst model is overwritten

    RESAMPLE_PROB = 0.33
    MUTATION_FACTORS = [0.75, 1.25]
    HYPERPARAMS_TO_MUTATE = ["LAM", "CLIPPING", "LR", "STEPS_PER_UPDATE", "ENTROPY", "VF_COEF"]

    NUM_SELECTION_GAMES = M_PARAMS["num_selection_games"]
    ##############
    # PPO PARAMS #
    ##############

    # Total environment timesteps for the PPO run
    PPO_RUN_TOT_TIMESTEPS = M_PARAMS["ppo_run_tot_timesteps"]

    NUM_PBT_ITER = int(TOTAL_STEPS_PER_AGENT * (POPULATION_SIZE) // (ITER_PER_SELECTION * PPO_RUN_TOT_TIMESTEPS))

    if M_PARAMS["minimal"]:
            NUM_PBT_ITER =  M_PARAMS["max_epoch"]
    


    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    
    TOTAL_BATCH_SIZE = M_PARAMS["total_batch_size"]
    
    # Number of minibatches we divide up each batch into before
    # performing gradient steps

    MINIBATCHES = M_PARAMS["minibatches"]

    BATCH_SIZE = TOTAL_BATCH_SIZE // sim_threads
  
    # Number of gradient steps to perform on each mini-batch
    STEPS_PER_UPDATE = M_PARAMS["steps_per_update"]

    # Learning rate
    LR = 0.0015

    # Entropy bonus coefficient
    ENTROPY = 0.5

    # Value function coefficient
    VF_COEF = 0.5

    # Gamma discounting factor
    GAMMA = 0.99

    # Lambda advantage discounting factor
    LAM = 0.98

    # Max gradient norm
    MAX_GRAD_NORM = 0.1

    # PPO clipping factor
    CLIPPING = 0.05

    # 0 is default value that does no annealing
    REW_SHAPING_HORIZON = M_PARAMS["rew_swap"]

    ##################
    # NETWORK PARAMS #
    ##################

    # Network type used
    NETWORK_TYPE = "conv_and_mlp"

    # Network params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3


    ##################
    # MDP/ENV PARAMS #
    ##################

    # Mdp params
    layout_name = None
    start_order_list = None

    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0.015,
        "POT_DISTANCE_REW": 0.03,
        "SOUP_DISTANCE_REW": 0.1,
    }
    
    # Env params
    horizon = 400

    #########
    # OTHER #
    #########

    # For non fixed MDPs
    mdp_generation_params = {
        "padded_mdp_shape": (11, 7),
        "mdp_shape_fn": ([5, 11], [5, 7]),
        "prop_empty_fn": [0.6, 1],
        "prop_feats_fn": [0, 0.6]
    }

    # Approximate info stats
    GRAD_UPDATES_PER_AGENT = STEPS_PER_UPDATE * MINIBATCHES * (PPO_RUN_TOT_TIMESTEPS // TOTAL_BATCH_SIZE) * ITER_PER_SELECTION * NUM_PBT_ITER // (POPULATION_SIZE)
    #GRAD_UPDATES_PER_AGENT = M_PARAMS["gradient_update"]
   
    print("Total steps per agent", TOTAL_STEPS_PER_AGENT)
    print("Grad updates per agent", GRAD_UPDATES_PER_AGENT)

    params = {
        "LOCAL_TESTING": LOCAL_TESTING,
        "RUN_TYPE": RUN_TYPE,
        "EX_NAME": EX_NAME,
        "SAVE_DIR": SAVE_DIR,
        "GPU_ID": GPU_ID,
        "mdp_params": {
            "layout_name": layout_name,
            "start_order_list": start_order_list,
            "rew_shaping_params": rew_shaping_params
        },
        "env_params": {
            "horizon": horizon
        },
        "PPO_RUN_TOT_TIMESTEPS": PPO_RUN_TOT_TIMESTEPS,
        "NUM_PBT_ITER": NUM_PBT_ITER,
        "ITER_PER_SELECTION": ITER_PER_SELECTION,
        "POPULATION_SIZE": POPULATION_SIZE,
        "RESAMPLE_PROB": RESAMPLE_PROB,
        "MUTATION_FACTORS": MUTATION_FACTORS,
        "mdp_generation_params": mdp_generation_params, # NOTE: currently not used
        "HYPERPARAMS_TO_MUTATE": HYPERPARAMS_TO_MUTATE,
        "REW_SHAPING_HORIZON": REW_SHAPING_HORIZON,
        "ENTROPY": ENTROPY,
        "GAMMA": GAMMA,
        "sim_threads": sim_threads,
        "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "LR": LR,
        "VF_COEF": VF_COEF,
        "STEPS_PER_UPDATE": STEPS_PER_UPDATE,
        "MINIBATCHES": MINIBATCHES,
        "CLIPPING": CLIPPING,
        "LAM": LAM,
        "NETWORK_TYPE": NETWORK_TYPE,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS": SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS": NUM_FILTERS,
        "NUM_CONV_LAYERS": NUM_CONV_LAYERS,
        "SEEDS": SEEDS,
        "NUM_SELECTION_GAMES": NUM_SELECTION_GAMES,
        "total_steps_per_agent": TOTAL_STEPS_PER_AGENT,
        "grad_updates_per_agent": GRAD_UPDATES_PER_AGENT
    }

@ex.named_config
def fixed_mdp():
    LOCAL_TESTING = False
    # fixed_mdp = True
    layout_name = "simple"

    sim_threads = 30 if not LOCAL_TESTING else 2
    PPO_RUN_TOT_TIMESTEPS = 36000 if not LOCAL_TESTING else 1000
    TOTAL_BATCH_SIZE = 12000 if not LOCAL_TESTING else 1000

    STEPS_PER_UPDATE = 5
    MINIBATCHES = 6 if not LOCAL_TESTING else 2
    LR = 5e-4

  
    
def get_SyKLRBR_agent(save_dir, agent, seed=0):
    save_dir = SyKLRBR_DATA_DIR + save_dir + '/seed_{}'.format(seed)
    # save_dir = SyKLRBR_DATA_DIR + save_dir + '/seed_{}'.format(seed)
    # config = load_pickle(save_dir + '/config')
    # agent = get_agent_from_saved_model(save_dir + "/agent" + str(agent) + "/pbt_format", config["sim_threads"])
    # return agent, config



def save(params, agent_name, model):
    save_folder = params["SAVE_DIR"] + agent_name + '/'
    """Save agent model, logs, and parameters"""
    create_dir_if_not_exists(save_folder)
    save_baselines_model(model, save_folder)
    print("FINAL AGENT SAVED AT :", save_folder)
    # save_dict_to_file(dict(logs), save_folder + "logs")
    # save_dict_to_file(params, save_folder + "params")

# temporary plotting
def plot_easy(train_info):
    data = train_info['value_loss']
    x = np.array(list(range(len(data))))
    y = data
    fig = tpl.figure()
    fig.plot(x, y, width=60, height=20)
    # print("---------- value-loss plot ----------")
    fig.show()

def pbt_one_run(params):
    params["SAVE_DIR"] = SyKLRBR_DATA_DIR + time.strftime('%Y_%m_%d-%H_%M_%S_') + "/"
    create_dir_if_not_exists(params["SAVE_DIR"])#create save dictionary
    save_dict_to_file(params, params["SAVE_DIR"] + "config") #save params in dic/config.txt
    
    # save_pickle(params, params["SAVE_DIR"] + "config")
   
    train_infos = []                            

    for seed in params["SEEDS"]:
        print("Training with seed", seed)
        print(params)
        reset_tf()
        set_global_seed(seed)

        curr_seed_dir = params["SAVE_DIR"] + "seed_" + str(seed) + "/"
        param_Copy = params.copy()
        param_Copy["SAVE_DIR"] = curr_seed_dir
        curr_seed_dir = param_Copy["SAVE_DIR"]
        create_dir_if_not_exists(curr_seed_dir)                         #create dic/seed0

        save_dict_to_file(param_Copy, curr_seed_dir + "config")
        save_pickle(param_Copy, curr_seed_dir + "config")

        mdp = OvercookedGridworld.from_layout_name(**params["mdp_params"])
        overcooked_env = OvercookedEnv(mdp, **params["env_params"])

        population_size = params["POPULATION_SIZE"] + 1

        for _ in range(5):
            overcooked_env.reset()

        gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', featurize_fn=lambda x: mdp.lossless_state_encoding(x), **param_Copy)
        gym_env.update_reward_shaping_param(1.0)
 
        annealer = LinearAnnealer(horizon=params["REW_SHAPING_HORIZON"])

        pbt_population = []
        pbt_agent_names = ['agent' + str(i) for i in range(population_size)]

        for agent_name in pbt_agent_names:
            agent = None
            if agent_name == "agent0":
                agent = RandomAgent(agent_name, sim_threads=param_Copy["sim_threads"])
            else:
                agent = PBTAgent(agent_name, param_Copy, gym_env=gym_env, top=True)

            pbt_population.append(agent)

        for epoch in range(0, params["NUM_PBT_ITER"]):
            progress = epoch / (params["NUM_PBT_ITER"]) * 100
            print("Epoch", epoch)
            print("Progress", epoch, "/", params["NUM_PBT_ITER"], " epochs", progress, "%")

            pairs_to_train = []
            rng = np.random.default_rng()

            for i in range(1, population_size):
                randomOpponent = max(0, population_size - 2 - rng.poisson(1) - (5 - i))
                add = (i, randomOpponent)
                pairs_to_train.append(add)
                pbt_population[i].logs["distribution"][randomOpponent] += 1
                pbt_population[i].logs["opponent"].append(randomOpponent)

            print("training")
            for pair in pairs_to_train: 
                trainingAgent, competor = pbt_population[pair[0]], pbt_population[pair[1]]
                # if competor
                # print(type(competor))
                print("")
                print("Training " + trainingAgent.agent_name, "vs", competor.agent_name)
                agent_env_steps = trainingAgent.num_ppo_runs * params["PPO_RUN_TOT_TIMESTEPS"]
         
                reward_shaping_param = annealer.param_value(agent_env_steps)

                # if trainingAgent.agent_name == "agent1":
                #     # reward_shaping_param =  max(1 - (agent_env_steps / (params["TOTAL_STEPS_PER_AGENT"] * 1.5)), 0)
                #     reward_shaping_param =  max(reward_shaping_param, 0.5)

                print("Current reward shaping:", reward_shaping_param, "\t Save_dir", params["SAVE_DIR"])
                
                trainingAgent.logs["reward_shaping"].append(reward_shaping_param)

                gym_env.update_reward_shaping_param(reward_shaping_param)
                
                #train
                if competor.agent_name == "agent0":
                    gym_env.other_agent = competor
                else:
                    gym_env.other_agent = competor.get_agent()
                train_info = trainingAgent.update(gym_env)
    
                save_folder = curr_seed_dir + trainingAgent.agent_name + '/'
                trainingAgent.save(save_folder)

                # agent_pair = AgentPair(trainingAgent.get_agent(), competor.get_agent())
                # overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=True, reward_shaping=reward_shaping_param)
            for index, agent in enumerate(pbt_population):
                if index > 0:
                    agent.update_pbt_iter_logs()
                
                # if index == 1 :
                #     currentSize = len(agent.logs["eprewmean"])
                #     if currentSize > 2:
                #         print(agent.logs["eprewmean"])
                #         print(agent.logs["eprewmean"][currentSize - 2])
                #         print(agent.logs["eprewmean"][currentSize - 1])
                #         print(agent.logs["eprewmean"][currentSize - 2] - agent.logs["eprewmean"][currentSize - 1])
                #         if agent.logs["eprewmean"][currentSize - 2] - agent.logs["eprewmean"][currentSize - 1] > 1:
                #             save_folder = curr_seed_dir + agent.agent_name + "/intermediate" + str(progress)
                #             pbt_population[population_size -1].save_predictor(save_folder)
                    

        for index, agent in enumerate(pbt_population):
            if index > 0:
                save_folder = curr_seed_dir + agent.agent_name + "/pbt_format/"
                agent.save_predictor(save_folder)
                agent.save(save_folder)


@ex.automain
def run_pbt(params):
    pbt_one_run(params)