#MAPPO_MPE_main.py
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_mpe import MAPPO_MPE
#from make_env import make_env
import gym
from env import MECVehicleCacheEnv
import dgl
import time
import json

class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        self.use_gnn = args.use_gnn
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        #self.env = make_env(env_name, discrete=True) # Discrete action space
        # env_name = 'yyf-v1'
        # self.env = gym.make(env_name)

        self.args.N = self.args.n_agent  # The number of agents
        self.args.obs_dim_n = [args.obs_dim for i in range(self.args.N)]  # obs dimensions of N agents
        self.args.action_dim_n = [args.act_dim for i in range(self.args.N)]  # actions dimensions of N agents
        #self.args.action_dim_n = [self.env.action_space[i].n for i in range(self.args.N)]  # actions dimensions of N agents
        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
        self.embedding_dim = args.embedding_dim

        #state
        #self.args.state_dim = np.sum(self.args.obs_dim_n)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        if self.use_gnn:
            self.args.state_dim = (self.embedding_dim) * args.n_agent
            #self.args.state_dim = np.sum(self.args.obs_dim_n) * args.n_agent
        else:
            self.args.state_dim = np.sum(self.args.obs_dim_n)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）

        # self.u, self.v = torch.tensor\
        #                      ([ 0, 0, 0, 0, 0, 0,      1, 1,    2, 2, 2, 2,        3, 3, 3, 3]), torch.tensor(
        #                       [ 7, 9, 16, 17, 18, 21,  6, 22,   14, 15, 19, 23,    4,  8, 11, 13])
        # self.u, self.v = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), torch.tensor(
        #     [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 1, 1, 1, 2, 3])
        # self.g = dgl.graph((self.u, self.v))
        # self.args.graph = self.g
        self.env = MECVehicleCacheEnv(
            #mec_vehicle_graph=self.g,
            #use_gnn=True,
            num_mec=args.num_mec,
            num_vehicles=args.num_vehicles,
            max_cache_size_mec=10,
            max_cache_size_vehicle=5,
            max_steps=args.episode_limit,
            mec_radius=20,
            num_contents=50,
            min_content_size=1,
            max_content_size=5,
            seed = seed
        )
        print("Calculated state_dim:", self.args.state_dim)

        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create N agents
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=f'runs/MAPPO/MAPPO_env_{self.env_name}_number_{self.number}_seed_{self.seed}_{timestamp}')
        #self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self):
        evaluate_num = -1  # Record the number of evaluations
        # self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
        # evaluate_num += 1
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps, current_graph = self.run_episode_mpe(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                # self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                # evaluate_num += 1
                self.agent_n.train(self.replay_buffer, self.total_steps, current_graph)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ ,_ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        # Save the rewards and models
        np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

    def run_episode_mpe(self, evaluate=False,visualize=False, visualize_freq=100):
        episode_reward = 0
        obs_n = self.env.reset()

        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):


            #test2
            #print("Passing Graph to Agent at Step", self.current_step)
            #print("Graph Edges:", current_graph.edges())
            # 检查图的节点数量
            #print("Current Graph: Number of Nodes =", current_graph.number_of_nodes())
            #print("Current Graph: Number of Edges =", current_graph.number_of_edges())


            if self.use_gnn:
                current_graph = self.env.mec_vehicle_graph  # Obtain the current graph from the environment
                a_n, a_logprob_n, gnn_output = self.agent_n.choose_action(obs_n, current_graph, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
                s = np.array(gnn_output).flatten()
                v_n = self.agent_n.get_value(s)
            else:
                current_graph = self.env.mec_vehicle_graph  # Obtain the current graph from the environment
                a_n, a_logprob_n, gnn_output = self.agent_n.choose_action(obs_n, None, evaluate=evaluate)
                s = np.array(obs_n).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
                v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents

            obs_next_n, r_n, done_n, _ = self.env.step(a_n)
            # episode_reward += r_n[0]
            for re in r_n:
                episode_reward += re

            print(f"Episode Step: {episode_step}")
            print("Observations:", obs_n)
            print("Actions:", a_n)
            # 打印奖励
            print("Rewards:", r_n)
            # 记录到日志文件
            #self._log_episode_data(episode_step, obs_n, a_n, r_n)

            if visualize and episode_step % visualize_freq == 0:
                self.env.plot_environment()
            # 更新 total_steps
            #self.total_steps +=1

            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            obs_n = obs_next_n
            if all(done_n):
                break

        if not evaluate:
            if self.use_gnn:
                _, _, gnn_output = self.agent_n.choose_action(obs_n, current_graph, evaluate=evaluate)
                s = np.array(gnn_output).flatten()
            else:
                # An episode is over, store v_n in the last step
                s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1, current_graph

    def _log_episode_data(self, step, obs, actions, rewards):
        log_data = {
            "total_steps": self.total_steps,
            "episode_step": step,
            "observations": [o.tolist() for o in obs],
            "actions": actions.tolist(),
            "rewards": rewards,
            "SUM:":sum(rewards),

        }
        with open("episode_logTrue15.json", "a") as file:
            for key, value in log_data.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")


class Runner_Random:
    def __init__(self, env_name, args, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        self.env = MECVehicleCacheEnv(
            #mec_vehicle_graph=self.g,
            #use_gnn=True,
            num_mec=4,
            num_vehicles=20,
            max_cache_size_mec=10,
            max_cache_size_vehicle=5,
            max_steps=500,
            mec_radius=20,
            num_contents=50,
            min_content_size=1,
            max_content_size=5,
            seed=1
        )
        self.args.N = self.env.agent_num  # The number of agents

        self.total_steps = 0
        self.evaluate_rewards = []
        # Create a tensorboard
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(
            log_dir=f'runs/MAPPO/MAPPO_env_{self.env_name}_number_{self.number}_seed_{self.seed}_{timestamp}')
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self):
        evaluate_num = -1
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            _, episode_steps = self.run_episode_mpe(evaluate=False)
            self.total_steps += episode_steps

        self.evaluate_policy()
        self.env.close()

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        obs_n = self.env.reset()

        for episode_step in range(self.args.episode_limit):
            # Sample the entire action space
            a_n = self.env.action_space.sample()  # Generate actions for all agents
            obs_next_n, r_n, done_n, _ = self.env.step(a_n)
            # episode_reward += r_n[0]
            for re in r_n:
                episode_reward += re
            obs_n = obs_next_n

            if all(done_n):
                break

        return episode_reward, episode_step + 1

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("Total steps: {} \t Evaluate reward: {}".format(self.total_steps, evaluate_reward))

        # Record the evaluation rewards to the TensorBoard
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward,
                               global_step=self.total_steps)
        # Optional: Save the assessment rewards to a file
        np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed),
                np.array(self.evaluate_rewards))

class Runner_Greedy:
    def __init__(self, env_name,args, number, seed):

        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        self.env = MECVehicleCacheEnv(
            # mec_vehicle_graph=self.g,
            # use_gnn=True,
            num_mec=4,
            num_vehicles=20,
            max_cache_size_mec=10,
            max_cache_size_vehicle=5,
            max_steps=500,
            mec_radius=20,
            num_contents=50,
            min_content_size=1,
            max_content_size=5,
            seed=1
        )
        self.args.N = self.env.agent_num  # The number of agents
        # Create a tensorboard
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(
            log_dir=f'runs/MAPPO/MAPPO_env_{self.env_name}_number_{self.number}_seed_{self.seed}_{timestamp}')
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)


    def greedy_action(self, obs_n):
        actions = []
        for idx, obs in enumerate(obs_n):
            current_cache = obs[1]  # Currently cached capacity
            recommendation_score = obs[2]  # recommendation score
            content_size = self.env.contents_info[self.env.content_id][1]  # The size of the current content

            # Determine the maximum cache size based on the node type (MEC or vehicle)
            if idx < self.env.num_mec:  # MEC
                max_cache_size = self.env.max_cache_size_mec
            else:  # vehicle
                max_cache_size = self.env.max_cache_size_vehicle

            # Check if the cache space is sufficient
            if current_cache + content_size <= max_cache_size:
                # If the recommendation index is high, choose to cache and recommend
                if recommendation_score > 5:
                    actions.append(0)  # Cache and recommend
                else:
                    actions.append(1)  # only recommend
            else:
                # If there is not enough space, decide whether to recommend only based on the recommendation index
                if recommendation_score > 5:
                    actions.append(2)  # only recommend
                else:
                    actions.append(3)  # Neither cache nor recommend
            #print(f"Agent {idx}, Action: {actions[-1]}")  # Print the actions of each agent
            # 调试打印
            #print(f"Agent {idx}, Observation: {obs}, Chosen Action: {actions[-1]}")

        return np.array(actions)


    def run(self):
        evaluate_num = -1
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            episode_steps = self.run_episode(evaluate=False)  # Obtain the number of steps for execution
            self.total_steps += episode_steps  # Update total_steps only based on the number of steps

        self.evaluate_policy()
        self.env.close()


    def run_episode(self, evaluate=False):
        obs_n = self.env.reset()
        episode_steps = 0  # Use separate variables to count the number of steps
        episode_reward = 0
        for _ in range(self.args.episode_limit):
            a_n = self.greedy_action(obs_n)
            obs_next_n, r_n, done_n, _ = self.env.step(a_n)
            episode_steps += 1
            episode_reward += sum(r_n)
            # for re in r_n:
            #     episode_reward += re
            # test
            #print(f"Step: {_}, Actions: {a_n}, Reward: {r_n}, New Observation: {obs_next_n}")

            obs_n = obs_next_n

            if all(done_n):
                break
        # test
        #print(f"Total Reward for this Episode: {episode_reward}")
        return episode_steps


    def evaluate_policy(self):
        evaluate_reward = 0
        for eval_episode in range(self.args.evaluate_times):
            episode_reward = 0
            obs_n = self.env.reset()  # Reset the environment and obtain the initial observations
            #print(f"Evaluation Episode {eval_episode + 1}: Initial Observation: {obs_n}")  # Print the initial observations

            for step in range(self.args.episode_limit):
                a_n = self.greedy_action(obs_n)  # Select the action based on the current observed values
                obs_next_n, r_n, done_n, _ = self.env.step(a_n)  # Perform actions and obtain new observations and rewards

                episode_reward += sum(r_n)  # Cumulative rewards
                obs_n = obs_next_n  # Update the observed values

                #print(f"Step {step + 1}: Actions: {a_n}, Reward: {r_n}, New Observation: {obs_next_n}, Done: {done_n}")  # 打印每步信息

                if all(done_n):
                    #print("All agents done at step:", step + 1)
                    break

            #print(f"Evaluation Episode {eval_episode + 1} Total Reward: {episode_reward}")  # Print the total reward for each evaluation round
            evaluate_reward += episode_reward


        evaluate_reward = evaluate_reward / self.args.evaluate_times
        #self.total_steps += self.args.episode_limit  # Update the total number of steps
        print("Total steps: {} \t Evaluate reward: {}".format(self.total_steps, evaluate_reward))
        # Record the evaluation rewards to the TensorBoard
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward,
                               global_step=self.total_steps)
        # Optional: Save the assessment rewards to a file
        np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed),
                np.array(self.evaluate_rewards))
        #return evaluate_reward




if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--device", default="cpu", help=" gpu or cpu")
    parser.add_argument("--max_train_steps", type=int, default=int(8e6), help=" Maximum number of training steps") #int(3e6)
    parser.add_argument("--episode_limit", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=20, help="Evaluate times")

    parser.add_argument("--use_gnn", type=bool, default=False, help="Whether to use GNN")
    parser.add_argument("--gnn_n_hidden", type=int, default=128, help="action dim of every agent")
    parser.add_argument("--gnn_n_layers", type=int, default=4, help="action dim of every agent")
    parser.add_argument("--embedding_dim", type=int, default=3, help="embedding dim of every region")
    parser.add_argument("--gnn_activation", default=torch.nn.functional.relu, help="")
    parser.add_argument("--n_agent", type=int, default=24, help="number of agents")

    parser.add_argument("--obs_dim", type=int, default=5, help="observation dim of every agent")
    parser.add_argument("--act_dim", type=int, default=4, help="action dim of every agent")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate") #5e-4
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

    parser.add_argument("--env_name", type=str, default='yyf-ZHONGtest', help="Name of environment.")
    parser.add_argument("--seed", type=int, default=30, help="Random seed")
    parser.add_argument("--number", type=int, default=4, help="number")
    parser.add_argument("--use_Rec", type=bool, default=False, help="Whether only to use Rec")

    parser.add_argument("--num_mec", type=int, default=4, help="Random seed")
    parser.add_argument("--num_vehicles", type=int, default=20, help="number")
    parser.add_argument("--num_contents", type=int, default=100, help="number")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name=args.env_name, number=args.number, seed=args.seed)
    runner.run_episode_mpe(evaluate=True, visualize=False, visualize_freq=100)
    runner.run()

    #runner_random = Runner_Random(env_name="yyf-v30", args=args, number=3, seed=30)
    #runner_random.run()

    #runner_greedy = Runner_Greedy(env_name="yyf-v15", args=args, number=1, seed=1)
    #runner_greedy.run()