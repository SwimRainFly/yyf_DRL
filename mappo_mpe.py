#mappo_mpe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from dgl.nn.pytorch import GraphConv, GATConv
from torch.distributions import Categorical
from torch.utils.data.sampler import *

def positive_safe_sigmoid(x):
    return torch.sigmoid(x) + 1e-8


class GCN(nn.Module):
    def __init__(self,
                 #g,  # Graph structure
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 device,
                 args):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.n_classes = n_classes
        #self.g = None
        self.device = device
        #self.g = (dgl.add_self_loop(g)).to(device)
        self.layers = nn.ModuleList() #Create a list of modules for storing each layer of the network
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation).to(device))

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation).to(device))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, activation=positive_safe_sigmoid).to(device))
        if args.use_orthogonal_init:
            print("------use_orthogonal_init gnn------")
            for layer_index in range(len(self.layers)):
                orthogonal_init(self.layers[layer_index])

    def forward(self, features,g):
        # if g is None:
        #     raise ValueError("Graph cannot be None")
        # print("Features Tensor Shape:", features.shape)

        # print("Graph Info: Number of Nodes =", g.number_of_nodes())
        # print("Graph Info: Number of Edges =", g.number_of_edges())

        # if features.size(1) != self.in_feats:
        #     raise ValueError(f"Feature size {features.size(1)} does not match in_feats {self.in_feats}")


        self.g = (dgl.add_self_loop(g)).to(self.device)
        h = features.to(torch.float32)
        #print("Shape of feature matrix h:", h.shape)
        for i, layer in enumerate(self.layers):
            #print("Shape of feature matrix h:", h.shape)
            h = layer(self.g, h) #Calculate the node features of the next layer using the graph structure and the current node features
        return h

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        return prob


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value

class GnnMlpActor(nn.Module):
    def __init__(self, args):
        self.embedding_dim = args.embedding_dim
        super(GnnMlpActor, self).__init__()
        #GCN
        self.gnn = GCN(
                 #g=None,
                 in_feats=args.obs_dim,
                 n_hidden=args.gnn_n_hidden,
                 n_classes=args.embedding_dim,
                 n_layers=args.gnn_n_layers,
                 activation=args.gnn_activation,
                 device=args.device,
                 args=args)
        #MLP
        self.mlp = Actor_MLP(args=args, actor_input_dim=args.embedding_dim )

    def forward(self, actor_input, current_graph):
        # if train:
        # gnn_input.shape = (batch_size, max_episode_len, region_N, gnn_input_dim)
        # actor_input.shape = (batch_size, max_episode_len, N, actor_input_dim)
        # if choose action:
        # gnn_input.shape = (region_N, gnn_input_dim)
        # actor_input.shape = (N, actor_input_dim)

        # Make sure to pass in a valid graph
        if current_graph is None:
            raise ValueError("current_graph cannot be None")

        actor_input_shape = actor_input.shape

        # Update the graphs in GCN
        self.gnn.g = current_graph

        if (len(actor_input.shape) == 4):
            actor_input = actor_input.permute(2, 0, 1, 3)
            gnn_output = self.gnn(actor_input, current_graph) # gnn_output n*embedding_dim
            actor_output = self.mlp(gnn_output)
            return actor_output.permute(1, 2, 0, 3), gnn_output
        else:
            gnn_output = self.gnn(actor_input, current_graph)  # gnn_output n*embedding_dim
            actor_output = self.mlp(gnn_output)
            return actor_output, gnn_output


class MAPPO_MPE:
    def __init__(self, args):
        self.N = args.N

        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim

        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.device = args.device
        self.use_gnn = args.use_gnn
        self.embedding_dim = args.embedding_dim

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip

        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N

        if self.use_rnn:
            print("------use rnn------")
            self.actor = Actor_RNN(args, self.actor_input_dim).to(args.device)
            self.critic = Critic_RNN(args, self.critic_input_dim).to(args.device)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim).to(args.device)
            self.critic = Critic_MLP(args, self.critic_input_dim).to(args.device)

        if self.use_gnn and (not self.use_rnn):
            print("------use gnn------")
            self.actor = GnnMlpActor(args).to(self.device)
            self.critic = Critic_MLP(args, self.critic_input_dim).to(self.device)
            # torch.save(self.actor.gnn.state_dict(), "test.pt")
            # self.one = GnnMlpActor(args).to(self.device)
            # self.one.gnn.load_state_dict(torch.load("test.pt"))
        if (not self.use_gnn) and (not self.use_rnn):
            self.actor = Actor_MLP(args, self.actor_input_dim).to(self.device)
            self.critic = Critic_MLP(args, self.critic_input_dim).to(self.device)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

    def choose_action(self, obs_n, current_graph, evaluate):
        with torch.no_grad():
            actor_inputs = []
            # gnn_inputs = []
            obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
            #print("obs_n shape:", obs_n.shape)  # 应该是 (9, 5)
            actor_inputs.append(obs_n)
            # gnn_inputs.append(obs_n)


            if self.add_agent_id:
                """
                    Add an one-hot vector to represent the agent_id
                    For example, if N=3
                    [obs of agent_1]+[1,0,0]
                    [obs of agent_2]+[0,1,0]
                    [obs of agent_3]+[0,0,1]
                    So, we need to concatenate a N*N unit matrix(torch.eye(N))
                """
                actor_inputs.append(torch.eye(self.N))

            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1).to(self.device)  # actor_input.shape=(N, actor_input_dim)
            # gnn_inputs = torch.cat([x for x in gnn_inputs], dim=-1)
            if self.use_gnn:
                prob, gnn_output = self.actor(actor_inputs, current_graph.to(self.device))
                #test3
                #print("Graph Edges:", current_graph.edges())
            else:
                prob = self.actor(actor_inputs)  # prob.shape=(N,action_dim)


            if evaluate:  # When evaluating the policy, we select the action with the highest probability
                if self.use_gnn:
                    a_n = prob.argmax(dim=-1)
                    return a_n.numpy(), None, gnn_output
                else:
                    a_n = prob.argmax(dim=-1)
                    return a_n.numpy(), None, None
            else:
                if self.use_gnn:
                    dist = Categorical(probs=prob)
                    a_n = dist.sample()
                    a_logprob_n = dist.log_prob(a_n)
                    return a_n.numpy(), a_logprob_n.numpy(), gnn_output
                else:
                    dist = Categorical(probs=prob)
                    a_n = dist.sample()
                    a_logprob_n = dist.log_prob(a_n)
                    return a_n.numpy(), a_logprob_n.numpy(), None

    def get_value(self, s):
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
            critic_inputs.append(s)
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1).to(self.device)  # critic_input.shape=(N, critic_input_dim)
            v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return v_n.numpy().flatten()

    def train(self, replay_buffer, total_steps, current_graph):
        batch = replay_buffer.get_training_data()  # get training data

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:,:-1]  # deltas.shape=(batch_size,episode_limit,N)
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_limit,N)
            v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,episode_limit,N)
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs = self.get_inputs(batch)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    get probs_now and values_now
                    probs_now.shape=(mini_batch_size, episode_limit, N, action_dim)
                    values_now.shape=(mini_batch_size, episode_limit, N)
                """

                if self.use_rnn:
                    # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    probs_now, values_now = [], []
                    for t in range(self.episode_limit):
                        prob = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size * self.N, -1)) # prob.shape=(mini_batch_size*N, action_dim)
                        probs_now.append(prob.reshape(self.mini_batch_size, self.N, -1))  # prob.shape=(mini_batch_size,N,action_dim）
                        v = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))  # v.shape=(mini_batch_size*N,1)
                        values_now.append(v.reshape(self.mini_batch_size, self.N))  # v.shape=(mini_batch_size,N)
                    # Stack them according to the time (dim=1)
                    probs_now = torch.stack(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                else:
                    if self.use_gnn:
                        probs_now, _ = self.actor(actor_inputs[index], current_graph)
                    else:
                        probs_now = self.actor(actor_inputs[index])

                    values_now = self.critic(critic_inputs[index]).squeeze(-1)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()  # dist_entropy.shape=(mini_batch_size, episode_limit, N)
                # batch['a_n'][index].shape=(mini_batch_size, episode_limit, N)
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])  # a_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())  # ratios.shape=(mini_batch_size, episode_limit, N)
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss.mean() + critic_loss.mean()
                ac_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])

        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.episode_limit, 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)

        if self.use_gnn:
            return actor_inputs, critic_inputs
        else:
            return actor_inputs, critic_inputs


    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(), "./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps / 1000)))

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(torch.load("./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))