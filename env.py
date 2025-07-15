import numpy as np
import gym
from gym import spaces
import dgl
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

class MECVehicleCacheEnv:

    def __init__(self, num_mec, num_vehicles, max_cache_size_mec, max_cache_size_vehicle, max_steps, mec_radius, num_contents, min_content_size, max_content_size, seed):
        super(MECVehicleCacheEnv, self).__init__()

        #self.mec_vehicle_graph = mec_vehicle_graph
        #self.use_gnn = use_gnn
        # Setting random seeds
        self.seed_value = seed
        np.random.seed(self.seed_value)
        self.random_state = np.random.RandomState(seed)

        self.num_mec = num_mec
        self.num_vehicles = num_vehicles
        self.max_cache_size_mec = max_cache_size_mec
        self.max_cache_size_vehicle = max_cache_size_vehicle
        self.max_steps = max_steps
        self.mec_radius = mec_radius  # MEC Coverage radius
        self.agent_num = self.num_mec + self.num_vehicles #Number of agents

        self.action_space = spaces.MultiDiscrete([4] * self.agent_num)

        obs_dim = 5  # ID, Cached capacity，Recommendation score, x-coordinate, y-coordinate
        self.observation_space = spaces.Tuple([spaces.Box(low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
                                                          high=np.array([float('inf'), max_cache_size_mec, float('inf'),
                                                                         float('inf'), float('inf')], dtype=np.float32))
                                               for _ in range(self.agent_num)])
        # Boundaries of the environment (assuming a square area)
        self.x_max = 100
        self.y_max = 100
        # Maximum distance travelled by the vehicle per step
        self.max_vehicle_move = 8
        # Define direction for each vehicle: 0 = up, 1 = right, 2 = down, 3 = left
        self.vehicle_directions = np.random.choice([0, 1, 2, 3], num_vehicles)

        # Initialise content size
        self.content_id = 0  # Initialise content ID
        self.cache = {i: {} for i in range(self.agent_num)}  # Initialising the cache
        self.num_contents = num_contents
        self.min_content_size = min_content_size
        self.max_content_size = max_content_size
        #self._initialize_content_sizes(num_contents, min_content_size, max_content_size)

        # Create information about each content
        self.contents_info = np.array(
            [[i, np.random.randint(self.min_content_size, self.max_content_size + 1)] +
             list(np.random.randint(1, 11, self.agent_num)) for i in range(self.num_contents)]
        )
        # Retrieve content information
        df = pd.read_excel(r"C:\Users\82770\Downloads\ContentInfoSelect.xlsx", engine='openpyxl')
        for _, row in df.iterrows():
            content_id, agent_id, rating = int(row['contentID']), int(row['agentID']), row['Rating']
            # Update the corresponding recommendation scores in contents_info
            if 0 <= content_id < self.num_contents and 0 <= agent_id < self.agent_num:
                self.contents_info[content_id][2 + agent_id] = rating

        print("contents_info:",self.contents_info)
        self.state = self._create_initial_state()
        #print("self.state:", self.state)
        self.current_step = 0
        self._update_graph()  #update graph


    def seed(self, seed=None):
        self.seed_value = seed
        np.random.seed(self.seed_value)

    def _get_obs(self, agent_id):
        # Returns the current state of the agent
        return self.state[agent_id]

    def _create_initial_state(self):
        state = []
        # Fixed MEC coordinates
        mec_coordinates = [(25, 75), (75, 75), (25, 25), (75, 25)]

        for i in range(self.num_mec):
            # Initialise MEC using fixed coordinates
            x, y = mec_coordinates[i]
            state.append([i, np.random.randint(1, self.max_cache_size_mec), np.random.randint(1, 10), x, y])

        # for i in range(self.num_mec, self.num_mec + self.num_vehicles):
        #     # For vehicles, initialise random coordinates
        #     state.append([i, np.random.randint(1, self.max_cache_size_vehicle), np.random.randint(1, 10),
        #                   np.random.rand() * self.x_max, np.random.rand() * self.y_max])
        for i in range(self.num_mec, self.num_mec + self.num_vehicles):
            # For vehicles, initialise random coordinates
            state.append([i, np.random.randint(1, self.max_cache_size_vehicle), np.random.randint(1, 10),
                            np.random.rand() * self.x_max, np.random.rand() * self.y_max])

        return np.array(state, dtype=np.float32)


    def step(self, actions):
        obs_n = [] #Storing new observations (states) for each agent
        reward_n = [] #Storing per agent rewards
        done_n = [] # Storing a flag for whether each agent has reached a termination condition
        info_n = {'n': []} #Storing additional information

        # Randomly select a piece of content to simulate a content request
        self.content_id = np.random.choice(self.num_contents)
        selected_content_info = self.contents_info[self.content_id]
        # Update the recommended score for each agent
        for i in range(self.agent_num):
            self.state[i][2] = selected_content_info[2 + i]  # 从第3个位置开始是推荐指数


        # Update vehicle coordinates
        for i, action in enumerate(actions):
            if i >= self.num_mec:  # Vehicle indexes start at num_mec
                vehicle_index = i - self.num_mec
                direction = self.vehicle_directions[vehicle_index]

                # Get current coordinates
                x, y = self.state[i][3], self.state[i][4]

                # Calculate new coordinates based on direction
                if direction == 0:  # up
                    new_y = y - self.max_vehicle_move
                    new_x = x
                elif direction == 1:  # right
                    new_x = x + self.max_vehicle_move
                    new_y = y
                elif direction == 2:  # down
                    new_y = y + self.max_vehicle_move
                    new_x = x
                elif direction == 3:  # left
                    new_x = x - self.max_vehicle_move
                    new_y = y

                # Checking and handling boundary collisions
                if new_x < 0 or new_x > self.x_max:
                    new_x = max(0, min(self.x_max, new_x))
                    direction = (direction + 2) % 4  # turn round
                if new_y < 0 or new_y > self.y_max:
                    new_y = max(0, min(self.y_max, new_y))
                    direction = (direction + 2) % 4  # turn round

                # Update status and direction
                self.state[i][3:5] = [new_x, new_y]
                self.vehicle_directions[vehicle_index] = direction

        # # Print all vehicle coordinate updates
        # print("Vehicle Coordinates:")
        # for i in range(self.num_mec, self.num_mec + self.num_vehicles):
        #     vehicle_index = i - self.num_mec
        #     direction = self.vehicle_directions[vehicle_index]
        #     print(f"Vehicle {vehicle_index}: x={self.state[i][3]}, y={self.state[i][4]}, direction={direction}")

        for i, action in enumerate(actions):
            if i < self.num_mec:
                reward, _ = self._mec_decision(i, action)
            else:
                reward = self._vehicle_decision(i, action)

            reward_n.append(reward)
            obs_n.append(self._get_obs(i))
            done_n.append(self.current_step >= self.max_steps)

        self.current_step += 1
        # # Update content_id
        # self.content_id = (self.content_id + 1) % self.num_contents
        self._update_graph()  # Update graph
        # #Print graph topology
        # print("Graph Edges:", self.mec_vehicle_graph.edges())
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # # Resetting NumPy's seeds
        # self.seed_value =+ 1
        # np.random.seed(self.seed_value)
        # Reset the state of the environment
        self.state = self._create_initial_state()
        self._update_graph()  # updated graph
        self.current_step = 0
        self.content_id = 0
        self.cache = {i: {} for i in range(self.agent_num)}
        # Generate initial observations for each agent
        initial_obs = tuple(self._get_obs(agent_id) for agent_id in range(self.agent_num))

        # Randomly select a piece of content to simulate a content request
        self.content_id = np.random.choice(self.num_contents)
        selected_content_info = self.contents_info[self.content_id]
        # Update the recommended score for each agent
        for i in range(self.agent_num):
            self.state[i][2] = selected_content_info[2 + i]

        # Randomly select a new content ID
        # self.content_id = np.random.randint(0, self.num_contents)
        # Reinitialize the content size (if the content size changes with each round)
        #self._initialize_content_sizes(self.num_contents, self.min_content_size, self.max_content_size)
        return initial_obs

    def _calculate_reward(self, agent_id, action, distance, content_cached, content_size):
        CLOUD_DOWNLOAD_COST = 20
        BANDWIDTH_COST = 5
        DISTANCE_WEIGHT = 0.1
        WEIGHT = 5

        reward = 0
        if agent_id < self.num_mec:  # MEC node

            connected_vehicles = self._find_connected_vehicles(agent_id)
            total_transmission_cost = 0
            for v_id in connected_vehicles:
                mec_vehicle_distance = self._calculate_distance(agent_id, v_id) * DISTANCE_WEIGHT
                total_transmission_cost += mec_vehicle_distance * BANDWIDTH_COST * DISTANCE_WEIGHT

            if action == 0:  # Both caching and recommendation
                reward += self.state[agent_id][2] * WEIGHT  # MEC Recommendation score
                if not content_cached:
                    reward -= CLOUD_DOWNLOAD_COST  # The cost of downloading content from the cloud

                for v_id in connected_vehicles:
                    reward += self.state[v_id][2] * WEIGHT  # Add the vehicle's recommendation score
                reward -= total_transmission_cost  # Subtract the transmission cost to the covered vehicles


            elif action == 1:  # Only cache
                if not content_cached:
                    reward = -CLOUD_DOWNLOAD_COST

            elif action == 2:  # No cache, only recommended

                reward += self.state[agent_id][2]*WEIGHT  # MEC recommendation score
                if not content_cached:
                    reward -= CLOUD_DOWNLOAD_COST  # If there is no cache, subtract the cost of downloading content from the cloud
                # Add the recommendation score of all the covered vehicles
                connected_vehicles = self._find_connected_vehicles(agent_id)
                for v_id in connected_vehicles:
                    reward += self.state[v_id][2]*WEIGHT  # Add the vehicle's recommendation score
                reward -= total_transmission_cost  # Subtract the transmission cost to the covered vehicles

            elif action == 3:  # Neither cache nor recommend
                reward = -100

            if not content_cached and (action == 0 or action == 1):
                self._update_cache(agent_id, self.content_id, content_size)


        else:  # vehicle node
            nearest_mec, mec_distance = self._find_nearest_mec(agent_id)
            mec_cached = self._check_cache(nearest_mec, self.content_id)

            if action == 0 or action == 1:  # Both cache and recommend or only cache
                if action == 0:
                    reward += self.state[agent_id][2] * WEIGHT

                # Calculate the cost of downloading content from the connected MEC
                download_cost = mec_distance * BANDWIDTH_COST * DISTANCE_WEIGHT
                reward -= download_cost

                # If the connected MEC does not cache this content, the cost of downloading the content from the cloud must also be deducted
                if not mec_cached:
                    reward -= CLOUD_DOWNLOAD_COST

            if action == 2:  # No cache, only recommended

                if content_cached:
                    # If the vehicle already has this content, there is no need to pull it from the MEC. The reward is the recommendation score
                    reward += self.state[agent_id][2] * WEIGHT
                else:
                    # If the vehicle does not cache this content
                    # Subtract the cost of downloading content from the connected MEC
                    download_cost = mec_distance * BANDWIDTH_COST * DISTANCE_WEIGHT
                    reward -= download_cost
                    # If the connected MEC does not cache this content, the cost of downloading the content from the cloud also needs to be subtracted
                    if not mec_cached:
                        reward -= CLOUD_DOWNLOAD_COST

            elif action == 3:  # Neither cache nor recommend
                reward = -100

            if not content_cached and (action == 0 or action == 1):
                self._update_cache(agent_id, self.content_id, content_size)

        return reward

    def _mec_decision(self, mec_id, action):
        # Current content ID
        current_content_cached = self._check_cache(mec_id, self.content_id)
        content_size = self.contents_info[self.content_id][1]

        reward = self._calculate_reward(mec_id, action, 0, current_content_cached, content_size)
        return reward, None

    def _vehicle_decision(self, vehicle_id, action):
        # Find the nearest MEC node and its distance
        nearest_mec, distance = self._find_nearest_mec(vehicle_id)
        current_content_cached = self._check_cache(vehicle_id, self.content_id)
        content_size = self.contents_info[self.content_id][1]

        reward = self._calculate_reward(vehicle_id, action, distance, current_content_cached, content_size)
        return reward

    def _initialize_content_sizes(self, num_contents, min_size, max_size):
        # Initialize the size of each content
        self.content_sizes = {content_id: np.random.randint(min_size, max_size)
                              for content_id in range(num_contents)}

    def _update_cache(self, agent_id, content_id, content_size):
        # Update the cache of the specified agent
        if agent_id < self.num_mec:
            max_cache_size = self.max_cache_size_mec
        else:
            max_cache_size = self.max_cache_size_vehicle

        # Update the cache of the specified agent
        current_cache_size = sum(item['size'] for item in self.cache[agent_id].values())

        # Check if there is enough space to add new content
        if current_cache_size + content_size <= max_cache_size:
            # Enough space. Just add directly
            self.cache[agent_id][content_id] = {'size': content_size, 'last_accessed': self.current_step}
            current_cache_size += content_size  # Update the current cache size
        else:
            # Not enough space and some content needs to be removed
            sorted_items = sorted(self.cache[agent_id].items(), key=lambda x: x[1]['last_accessed'])
            while sorted_items and current_cache_size + content_size > max_cache_size:
                removed_item = sorted_items.pop(0)
                removed_content_id, removed_info = removed_item
                current_cache_size -= removed_info['size']
                del self.cache[agent_id][removed_content_id]

            # Add new content
            if current_cache_size + content_size <= max_cache_size:
                self.cache[agent_id][content_id] = {'size': content_size, 'last_accessed': self.current_step}
                current_cache_size += content_size  # Update the current cache size

        # Update the last access time
        for content in self.cache[agent_id].values():
            content['last_accessed'] = self.current_step

        # Update the cache size information in the agent state
        self.state[agent_id][1] = current_cache_size

    def _check_cache(self, agent_id, content_id):
        # Check whether the  agent has cached the content
        return content_id in self.cache[agent_id]

    def _update_graph(self):
        num_total_nodes = self.num_mec + self.num_vehicles  # Total number of nodes
        edges = []
        for i in range(self.num_mec):
            mec_x, mec_y = self.state[i][3:5]
            for j in range(self.num_mec, self.num_mec + self.num_vehicles):
                vehicle_x, vehicle_y = self.state[j][3:5]
                distance = np.sqrt((mec_x - vehicle_x) ** 2 + (mec_y - vehicle_y) ** 2)
                if distance <= self.mec_radius:
                    edges.append((i, j))

        if edges:  # Check if edges is empty
            u, v = zip(*edges)
            self.mec_vehicle_graph = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=num_total_nodes)
        else:
            # When there are no edges, create an empty graph with a fixed number of nodes
            self.mec_vehicle_graph = dgl.graph(([], []), num_nodes=num_total_nodes)

        #test1
        #print("Updated Graph at Step", self.current_step)
        #print("Graph Edges:", self.mec_vehicle_graph.edges())



    def _find_connected_vehicles(self, mec_id):
        mec_x, mec_y = self.state[mec_id][3:5] #Obtain the x and y coordinates of the specified MEC node from the state array
        connected_vehicles = [] #An empty list stores an index
        for i in range(self.num_mec, self.num_mec + self.num_vehicles):# Traverse all vehicles (the index of the vehicles is after all MEC nodes)
            vehicle_x, vehicle_y = self.state[i][3:5]
            distance = np.sqrt((mec_x - vehicle_x)**2 + (mec_y - vehicle_y)**2) #Calculate the distance
            if distance <= self.mec_radius:
                connected_vehicles.append(i)
        return connected_vehicles #Return a list containing the indexes of all connected vehicles

    def _find_nearest_mec(self, vehicle_id):
        # Find the nearest MEC node and its distance
        nearest_mec = None
        min_distance = float('inf')
        for i in range(self.num_mec):
            distance = self._calculate_distance(vehicle_id, i)
            if distance < min_distance:
                min_distance = distance
                nearest_mec = i
        return nearest_mec, min_distance

    def _calculate_distance(self, agent_id1, agent_id2):
        x1, y1 = self.state[agent_id1][3], self.state[agent_id1][4]
        x2, y2 = self.state[agent_id2][3], self.state[agent_id2][4]
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def plot_environment(self):
        fig, ax = plt.subplots()
        # Draw MEC nodes and their coverage areas
        for i in range(self.num_mec):
            mec_x, mec_y = self.state[i][3], self.state[i][4]
            circle = patches.Circle((mec_x, mec_y), self.mec_radius, fill=False, color='blue', linestyle='dotted')
            ax.add_patch(circle)
            ax.plot(mec_x, mec_y, 'bo', markersize=10)

        # Draw vehicle nodes
        for i in range(self.num_mec, self.num_mec + self.num_vehicles):
            vehicle_x, vehicle_y = self.state[i][3], self.state[i][4]
            ax.plot(vehicle_x, vehicle_y, 'ro', markersize=5)  # 车辆节点为红色

        # Draw the connection line between the MEC and the vehicle
        for u, v in zip(*self.mec_vehicle_graph.edges()):
            mec_x, mec_y = self.state[u][3], self.state[u][4]
            vehicle_x, vehicle_y = self.state[v][3], self.state[v][4]
            ax.plot([mec_x, vehicle_x], [mec_y, vehicle_y], 'gray', linestyle='--')  # Use grey dotted lines to indicate the connection lines

        ax.set_xlim(0, self.x_max)
        ax.set_ylim(0, self.y_max)
        plt.show()

    def close(self):
        pass


# #
# gym.register(
#     id='yyf-v1',
#     entry_point='env:MECVehicleCacheEnv',
#     kwargs={
#         'num_mec': 2,
#         'num_vehicles': 7,
#         'max_cache_size_mec': 10,
#         'max_cache_size_vehicle': 5,
#         'max_steps': 240,
#         'mec_radius': 20,
#         'num_contents': 10,
#         'min_content_size': 1,
#         'max_content_size': 10
#     }
# )

if __name__ == "__main__":
    # Create a graph structure between MEC and Vehicle
    u, v = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2, 3, 4, 5, 6, 0])
    mec_vehicle_graph = dgl.graph((u, v))

    # Initialize the environment
    # Print the intermediate result to determine if the topology has changed
    # Random seeds, averaging
    #baseline
    env = MECVehicleCacheEnv(
        #mec_vehicle_graph=mec_vehicle_graph,
        #use_gnn=True,
        num_mec=4,
        num_vehicles=20,
        max_cache_size_mec=10,
        max_cache_size_vehicle=5,
        max_steps=25,
        mec_radius=20,
        num_contents=50,
        min_content_size=1,
        max_content_size=5,
        seed = 1
    )

    #env.seed(1)

    # Reset the environment and start a new round
    observation = env.reset()
    print("Initial Observation:", observation)

    # Run several time steps
    for step in range(5):
        # Randomly select an action
        actions = np.random.choice(4, env.agent_num)
        # Perform actions and receive environmental feedback
        obs, rewards, dones, info = env.step(actions)

        # Draw the current environmental state
        env.plot_environment()

        print(f"Step {step} ---")
        print("Observations:", obs)
        print("Rewards:", rewards)
        print("Dones:", dones)
        print("Info:", info)
        print("Graph:", env.mec_vehicle_graph.edges())

        # Check if any agent has achieved its goal
        if any(dones):
            break

# def test_environment():
#
#     env = MECVehicleCacheEnv(
#         num_mec=4,
#         num_vehicles=20,
#         max_cache_size_mec=10,
#         max_cache_size_vehicle=5,
#         max_steps=25,
#         mec_radius=20,
#         num_contents=50,
#         min_content_size=1,
#         max_content_size=5,
#         seed=1
#     )
#
#
#     observation = env.reset()
#
#
#     for step in range(5):
#         actions = np.random.choice(4, env.agent_num)
#         obs, rewards, dones, info = env.step(actions)
#
#
#         print(f"\nStep {step + 1} ---")
#         for agent_id in range(env.agent_num):
#             cache = env.cache[agent_id]
#             print(f"Agent {agent_id} Cache: {cache}")
#
#
#             cache_size = sum(item['size'] for item in cache.values())
#             max_cache_size = env.max_cache_size_mec if agent_id < env.num_mec else env.max_cache_size_vehicle
#             assert cache_size <= max_cache_size, f"Cache overflow in Agent {agent_id}"
#
#
#         env.plot_environment()
#
#
#
# test_environment()
