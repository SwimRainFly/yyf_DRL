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
        # 设置随机种子
        self.seed_value = seed
        np.random.seed(self.seed_value)
        self.random_state = np.random.RandomState(seed)

        self.num_mec = num_mec
        self.num_vehicles = num_vehicles
        self.max_cache_size_mec = max_cache_size_mec
        self.max_cache_size_vehicle = max_cache_size_vehicle
        self.max_steps = max_steps
        self.mec_radius = mec_radius  # MEC 覆盖范围半径
        self.agent_num = self.num_mec + self.num_vehicles #智能体的数量

        self.action_space = spaces.MultiDiscrete([4] * self.agent_num)

        obs_dim = 5  # ID, 已缓存容量，推荐指数, x坐标, y坐标
        self.observation_space = spaces.Tuple([spaces.Box(low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
                                                          high=np.array([float('inf'), max_cache_size_mec, float('inf'),
                                                                         float('inf'), float('inf')], dtype=np.float32))
                                               for _ in range(self.agent_num)])
        # 环境的边界（假设为正方形区域）
        self.x_max = 100
        self.y_max = 100
        # 车辆每步的最大移动距离
        self.max_vehicle_move = 8
        # 为每辆车定义方向：0=上，1=右，2=下，3=左
        self.vehicle_directions = np.random.choice([0, 1, 2, 3], num_vehicles)

        # 初始化内容大小
        self.content_id = 0  # 初始内容ID
        self.cache = {i: {} for i in range(self.agent_num)}  # 初始化缓存
        self.num_contents = num_contents
        self.min_content_size = min_content_size
        self.max_content_size = max_content_size
        #self._initialize_content_sizes(num_contents, min_content_size, max_content_size)

        # 创建每个内容的信息，每行包括content ID、content大小和对应每个代理的推荐指数
        self.contents_info = np.array(
            [[i, np.random.randint(self.min_content_size, self.max_content_size + 1)] +
             list(np.random.randint(1, 11, self.agent_num)) for i in range(self.num_contents)]
        )
        # 从 Excel 文件读取内容信息
        df = pd.read_excel(r"C:\Users\82770\Downloads\ContentInfoSelect.xlsx", engine='openpyxl')
        for _, row in df.iterrows():
            content_id, agent_id, rating = int(row['contentID']), int(row['agentID']), row['Rating']
            # 更新 contents_info 中相应的推荐指数
            if 0 <= content_id < self.num_contents and 0 <= agent_id < self.agent_num:
                self.contents_info[content_id][2 + agent_id] = rating

        print("contents_info:",self.contents_info)
        self.state = self._create_initial_state()
        #print("self.state:", self.state)
        self.current_step = 0
        self._update_graph()  # 更新图


    def seed(self, seed=None):
        self.seed_value = seed
        np.random.seed(self.seed_value)

    def _get_obs(self, agent_id):
        # 返回指定代理的当前状态
        return self.state[agent_id]

    def _create_initial_state(self):
        state = []
        # 固定的 MEC 坐标
        mec_coordinates = [(25, 75), (75, 75), (25, 25), (75, 25)]

        for i in range(self.num_mec):
            # 使用固定坐标初始化MEC
            x, y = mec_coordinates[i]
            state.append([i, np.random.randint(1, self.max_cache_size_mec), np.random.randint(1, 10), x, y])

        # for i in range(self.num_mec, self.num_mec + self.num_vehicles):
        #     # 对于车辆，初始化随机坐标
        #     state.append([i, np.random.randint(1, self.max_cache_size_vehicle), np.random.randint(1, 10),
        #                   np.random.rand() * self.x_max, np.random.rand() * self.y_max])
        for i in range(self.num_mec, self.num_mec + self.num_vehicles):
            # 对于车辆，初始化随机坐标使用 self.random_state 生成随机数
            state.append([i, np.random.randint(1, self.max_cache_size_vehicle), np.random.randint(1, 10),
                            np.random.rand() * self.x_max, np.random.rand() * self.y_max])

        return np.array(state, dtype=np.float32)


    def step(self, actions):
        obs_n = [] #存储每个agent的新观察（状态）
        reward_n = [] #存储每个代理的奖励
        done_n = [] # 存储每个代理是否达到终止条件的标志
        info_n = {'n': []} #存储额外的信息，可能用于调试或信息记录

        # 随机选择一个内容 模拟content请求
        self.content_id = np.random.choice(self.num_contents)
        selected_content_info = self.contents_info[self.content_id]
        # 更新每个代理的推荐指数
        for i in range(self.agent_num):
            self.state[i][2] = selected_content_info[2 + i]  # 从第3个位置开始是推荐指数


        #更新车辆的坐标
        for i, action in enumerate(actions):
            if i >= self.num_mec:  # 车辆索引从 num_mec 开始
                vehicle_index = i - self.num_mec
                direction = self.vehicle_directions[vehicle_index]

                # 获取当前坐标
                x, y = self.state[i][3], self.state[i][4]

                # 根据方向计算新坐标
                if direction == 0:  # 上
                    new_y = y - self.max_vehicle_move
                    new_x = x
                elif direction == 1:  # 右
                    new_x = x + self.max_vehicle_move
                    new_y = y
                elif direction == 2:  # 下
                    new_y = y + self.max_vehicle_move
                    new_x = x
                elif direction == 3:  # 左
                    new_x = x - self.max_vehicle_move
                    new_y = y

                # 检查并处理边界碰撞
                if new_x < 0 or new_x > self.x_max:
                    new_x = max(0, min(self.x_max, new_x))
                    direction = (direction + 2) % 4  # 掉头
                if new_y < 0 or new_y > self.y_max:
                    new_y = max(0, min(self.y_max, new_y))
                    direction = (direction + 2) % 4  # 掉头

                # 更新状态和方向
                self.state[i][3:5] = [new_x, new_y]
                self.vehicle_directions[vehicle_index] = direction

        # # 打印所有车辆坐标更新
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
        # # 在这里更新 content_id，希望每个时间步都面对新内容
        # self.content_id = (self.content_id + 1) % self.num_contents
        self._update_graph()  # 更新图
        # #打印图拓扑结构
        # print("Graph Edges:", self.mec_vehicle_graph.edges())
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # # 重置 NumPy 的种子
        # self.seed_value =+ 1
        # np.random.seed(self.seed_value)
        # 重置环境状态
        self.state = self._create_initial_state()
        self._update_graph()  # 更新图
        self.current_step = 0
        self.content_id = 0
        self.cache = {i: {} for i in range(self.agent_num)}
        # 为每个智能体生成初始观察值
        initial_obs = tuple(self._get_obs(agent_id) for agent_id in range(self.agent_num))

        # 随机选择一个内容 模拟content请求
        self.content_id = np.random.choice(self.num_contents)
        selected_content_info = self.contents_info[self.content_id]
        # 更新每个代理的推荐指数
        for i in range(self.agent_num):
            self.state[i][2] = selected_content_info[2 + i]  # 从第3个位置开始是推荐指数

        # 随机选择一个新的内容ID
        # self.content_id = np.random.randint(0, self.num_contents)
        # 重新初始化内容大小（如果如果内容大小会随着每个回合变化）
        #self._initialize_content_sizes(self.num_contents, self.min_content_size, self.max_content_size)
        return initial_obs

    def _calculate_reward(self, agent_id, action, distance, content_cached, content_size):
        CLOUD_DOWNLOAD_COST = 20
        BANDWIDTH_COST = 5
        DISTANCE_WEIGHT = 0.1
        WEIGHT = 5

        reward = 0
        if agent_id < self.num_mec:  # 如果是 MEC 节点

            connected_vehicles = self._find_connected_vehicles(agent_id)
            total_transmission_cost = 0
            for v_id in connected_vehicles:
                mec_vehicle_distance = self._calculate_distance(agent_id, v_id) * DISTANCE_WEIGHT
                total_transmission_cost += mec_vehicle_distance * BANDWIDTH_COST * DISTANCE_WEIGHT

            if action == 0:  # 既缓存又推荐
                reward += self.state[agent_id][2] * WEIGHT  # MEC 推荐指数
                if not content_cached:
                    reward -= CLOUD_DOWNLOAD_COST  # 从云端下载内容的成本

                for v_id in connected_vehicles:
                    reward += self.state[v_id][2] * WEIGHT  # 加上车辆的推荐指数
                reward -= total_transmission_cost  # 减去向覆盖车辆的传输代价


            elif action == 1:  # 只缓存
                if not content_cached:
                    reward = -CLOUD_DOWNLOAD_COST

            elif action == 2:  # 不缓存仅推荐

                reward += self.state[agent_id][2]*WEIGHT  # MEC 推荐指数
                if not content_cached:
                    reward -= CLOUD_DOWNLOAD_COST  # 如果没有缓存，减去从云下载内容的成本
                # 加上覆盖的所有车辆的推荐指数
                connected_vehicles = self._find_connected_vehicles(agent_id)
                for v_id in connected_vehicles:
                    reward += self.state[v_id][2]*WEIGHT  # 加上车辆的推荐指数
                reward -= total_transmission_cost  # 减去向覆盖车辆的传输代价

            elif action == 3:  # 不缓存不推荐
                reward = -100  # 设计为一个很小的负数

            if not content_cached and (action == 0 or action == 1):
                self._update_cache(agent_id, self.content_id, content_size)


        else:  # 如果是车辆
            nearest_mec, mec_distance = self._find_nearest_mec(agent_id)
            mec_cached = self._check_cache(nearest_mec, self.content_id)

            if action == 0 or action == 1:  # 既缓存又推荐 或 只缓存
                if action == 0:
                    reward += self.state[agent_id][2] * WEIGHT  # 推荐的有推荐指数

                # 计算从所连接的 MEC 下载内容的代价
                download_cost = mec_distance * BANDWIDTH_COST * DISTANCE_WEIGHT
                reward -= download_cost

                # 如果所连接的 MEC 没有缓存该内容，则还需减去从云端下载内容的代价
                if not mec_cached:
                    reward -= CLOUD_DOWNLOAD_COST

            if action == 2:  # 不缓存仅推荐

                if content_cached:
                    # 如果车辆已经有这个内容，无需从 MEC 中拉取，奖励为推荐指数
                    reward += self.state[agent_id][2] * WEIGHT
                else:
                    # 如果车辆没有缓存这个内容
                    # 减去从所连接的 MEC 下载内容的代价
                    download_cost = mec_distance * BANDWIDTH_COST * DISTANCE_WEIGHT
                    reward -= download_cost
                    # 如果所连接的 MEC 没有缓存该内容，还需减去从云端下载内容的成本
                    if not mec_cached:
                        reward -= CLOUD_DOWNLOAD_COST

            elif action == 3:  # 不缓存不推荐
                reward = -100  # 设计为一个很小的负数

            if not content_cached and (action == 0 or action == 1):
                self._update_cache(agent_id, self.content_id, content_size)

        return reward

    def _mec_decision(self, mec_id, action):
        # 当前内容ID
        current_content_cached = self._check_cache(mec_id, self.content_id)
        content_size = self.contents_info[self.content_id][1]

        reward = self._calculate_reward(mec_id, action, 0, current_content_cached, content_size)
        return reward, None

    def _vehicle_decision(self, vehicle_id, action):
        # 找到最近的MEC节点和距离
        nearest_mec, distance = self._find_nearest_mec(vehicle_id)
        current_content_cached = self._check_cache(vehicle_id, self.content_id)
        content_size = self.contents_info[self.content_id][1]

        reward = self._calculate_reward(vehicle_id, action, distance, current_content_cached, content_size)
        return reward

    def _initialize_content_sizes(self, num_contents, min_size, max_size):
        # 初始化每个内容的大小
        self.content_sizes = {content_id: np.random.randint(min_size, max_size)
                              for content_id in range(num_contents)}

    def _update_cache(self, agent_id, content_id, content_size):
        # 更新指定代理的缓存
        if agent_id < self.num_mec:
            max_cache_size = self.max_cache_size_mec
        else:
            max_cache_size = self.max_cache_size_vehicle

        # 计算当前缓存的大小
        current_cache_size = sum(item['size'] for item in self.cache[agent_id].values())

        # 检查是否有足够的空间添加新内容
        if current_cache_size + content_size <= max_cache_size:
            # 有足够的空间，直接添加
            self.cache[agent_id][content_id] = {'size': content_size, 'last_accessed': self.current_step}
            current_cache_size += content_size  # 更新当前缓存大小
        else:
            # 没有足够的空间，需要移除一些内容
            sorted_items = sorted(self.cache[agent_id].items(), key=lambda x: x[1]['last_accessed'])
            while sorted_items and current_cache_size + content_size > max_cache_size:
                removed_item = sorted_items.pop(0)
                removed_content_id, removed_info = removed_item
                current_cache_size -= removed_info['size']
                del self.cache[agent_id][removed_content_id]

            # 添加新内容
            if current_cache_size + content_size <= max_cache_size:
                self.cache[agent_id][content_id] = {'size': content_size, 'last_accessed': self.current_step}
                current_cache_size += content_size  # 更新当前缓存大小

        # 更新最后访问时间
        for content in self.cache[agent_id].values():
            content['last_accessed'] = self.current_step

        # 更新智能体状态中的缓存大小信息
        self.state[agent_id][1] = current_cache_size

    def _check_cache(self, agent_id, content_id):
        # 检查指定代理是否缓存了指定内容
        return content_id in self.cache[agent_id]

    def _update_graph(self):
        num_total_nodes = self.num_mec + self.num_vehicles  # 总节点数
        edges = []
        for i in range(self.num_mec):
            mec_x, mec_y = self.state[i][3:5]
            for j in range(self.num_mec, self.num_mec + self.num_vehicles):
                vehicle_x, vehicle_y = self.state[j][3:5]
                distance = np.sqrt((mec_x - vehicle_x) ** 2 + (mec_y - vehicle_y) ** 2)
                if distance <= self.mec_radius:
                    edges.append((i, j))

        if edges:  # 检查 edges 是否为空
            u, v = zip(*edges)
            self.mec_vehicle_graph = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=num_total_nodes)
        else:
            # 当没有边时，创建一个具有固定节点数的空图
            self.mec_vehicle_graph = dgl.graph(([], []), num_nodes=num_total_nodes)

        #test1
        #print("Updated Graph at Step", self.current_step)
        #print("Graph Edges:", self.mec_vehicle_graph.edges())



    def _find_connected_vehicles(self, mec_id):
        mec_x, mec_y = self.state[mec_id][3:5] #从状态数组中获取指定 MEC 节点的 x 和 y 坐标
        connected_vehicles = [] #空列表存储索引
        for i in range(self.num_mec, self.num_mec + self.num_vehicles):#遍历所有车辆（车辆的索引是在所有MEC节点之后）
            vehicle_x, vehicle_y = self.state[i][3:5]
            distance = np.sqrt((mec_x - vehicle_x)**2 + (mec_y - vehicle_y)**2) #计算距离
            if distance <= self.mec_radius:
                connected_vehicles.append(i)
        return connected_vehicles #返回一个包含所有连接车辆索引的列表

    def _find_nearest_mec(self, vehicle_id):
        # 找到最近的MEC节点及其距离
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
        # 绘制 MEC 节点和它们的覆盖范围
        for i in range(self.num_mec):
            mec_x, mec_y = self.state[i][3], self.state[i][4]
            circle = patches.Circle((mec_x, mec_y), self.mec_radius, fill=False, color='blue', linestyle='dotted')
            ax.add_patch(circle)
            ax.plot(mec_x, mec_y, 'bo', markersize=10)  # MEC 节点为蓝色

        # 绘制车辆节点
        for i in range(self.num_mec, self.num_mec + self.num_vehicles):
            vehicle_x, vehicle_y = self.state[i][3], self.state[i][4]
            ax.plot(vehicle_x, vehicle_y, 'ro', markersize=5)  # 车辆节点为红色

        # 绘制 MEC 与车辆之间的连线
        for u, v in zip(*self.mec_vehicle_graph.edges()):
            mec_x, mec_y = self.state[u][3], self.state[u][4]
            vehicle_x, vehicle_y = self.state[v][3], self.state[v][4]
            ax.plot([mec_x, vehicle_x], [mec_y, vehicle_y], 'gray', linestyle='--')  # 用灰色虚线表示连线

        ax.set_xlim(0, self.x_max)
        ax.set_ylim(0, self.y_max)
        plt.show()

    def close(self):
        pass
    # ... 其余的方法（render, close） ...

# # 注册环境
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
#         'num_contents': 10,  # 假设有10种不同的内容
#         'min_content_size': 1,  # 最小内容大小
#         'max_content_size': 10  # 最大内容大小
#     }
# )

if __name__ == "__main__":
    # 创建 MEC 和 Vehicle 之间的图结构
    u, v = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2, 3, 4, 5, 6, 0])
    mec_vehicle_graph = dgl.graph((u, v))

    # 初始化环境
    #打印中间结果 判断拓扑是否改变
    #随机种子，求平均
    #baseline
    env = MECVehicleCacheEnv(
        #mec_vehicle_graph=mec_vehicle_graph,
        #use_gnn=True,
        num_mec=4,#正方形
        num_vehicles=20,#设置多
        max_cache_size_mec=10,
        max_cache_size_vehicle=5,
        max_steps=25,
        mec_radius=20,
        num_contents=50,#50-100
        min_content_size=1,
        max_content_size=5,
        seed = 1  # 设置一个固定的种子值
    )

    #env.seed(1)  # 使用一个固定的种子值

    # 重置环境，开始新的回合
    observation = env.reset()
    print("Initial Observation:", observation)

    # 运行几个时间步骤
    for step in range(5):
        # 随机选择动作
        actions = np.random.choice(4, env.agent_num)
        # 执行动作并接收环境反馈
        obs, rewards, dones, info = env.step(actions)

        # 绘制当前环境状态
        env.plot_environment()

        print(f"Step {step} ---")
        print("Observations:", obs)
        print("Rewards:", rewards)
        print("Dones:", dones)
        print("Info:", info)
        print("Graph:", env.mec_vehicle_graph.edges())

        # 检查是否有智能体完成了它的目标
        if any(dones):
            break

# def test_environment():
#     # 创建环境实例
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
#     # 重置环境
#     observation = env.reset()
#
#     # 运行环境几个步骤，并检查缓存
#     for step in range(5):
#         actions = np.random.choice(4, env.agent_num)
#         obs, rewards, dones, info = env.step(actions)
#
#         # 输出每个智能体的缓存情况
#         print(f"\nStep {step + 1} ---")
#         for agent_id in range(env.agent_num):
#             cache = env.cache[agent_id]
#             print(f"Agent {agent_id} Cache: {cache}")
#
#             # 检查缓存大小是否未超过最大值
#             cache_size = sum(item['size'] for item in cache.values())
#             max_cache_size = env.max_cache_size_mec if agent_id < env.num_mec else env.max_cache_size_vehicle
#             assert cache_size <= max_cache_size, f"Cache overflow in Agent {agent_id}"
#
#         # 绘制当前环境状态
#         env.plot_environment()
#
#
# # 运行测试
# test_environment()
