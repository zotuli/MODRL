
try:  # Package import when launched with ``python -m rl_for_moo...``.
    from . import function_revised as default_backend
    from .function_revised import *
except ImportError:  # Direct execution from inside rl_for_moo/ (legacy workflow).
    import function_revised as default_backend
    from function_revised import *
from scipy.spatial import KDTree
class osyczka_kundu:
    """
    多目标优化问题定义类（Osyczka and Kundu）。
    包含：
    - 决策变量的上下界
    - 目标函数的定义
    - 约束条件的定义
    """

    def __init__(self):
        # 决策变量的上下界
        self.obj_dim = 2  # 目标空间维度（Osyczka and Kundu问题有2个目标）
        self.x_low = np.array([0, 0, 1, 0, 0, 0])  # 决策变量的下界
        self.x_up = np.array([10, 10, 5, 6, 5, 5])  # 决策变量的上界
        self.x_dim = len(self.x_low)  # 决策空间维度
        self.inital_state = np.zeros(self.x_dim)  # 初始状态

    def obj(self, x):
        """
        计算 Osyczka and Kundu 测试问题的目标函数值。
        参数:
            x (np.ndarray): 决策变量，形状为 (batch_size, x_dim)。
        返回:
            np.ndarray: 目标函数值，形状为 (batch_size, obj_dim)。
        """
        batch_size = x.shape[0]
        f = np.full((batch_size, self.obj_dim), 1e20, dtype=np.float32)  # 初始化目标函数值为大数

        for i in range(batch_size):
            x_i = x[i]
            # 计算 f1
            f1 = -25 * (x_i[0] - 2) ** 2 - (x_i[1] - 2) ** 2 - (x_i[2] - 1) ** 2 - (x_i[3] - 4) ** 2 - (x_i[4] - 1) ** 2
            f1 -= (x_i[5] - 4) ** 2
            # 计算 f2
            f2 = np.sum(x_i[:6] ** 2)

            f[i] = [f1, f2]  # 存储目标函数值

        return f

    def constraints(self, x):
        """
        计算 Osyczka and Kundu 测试问题的约束条件。
        参数:
            x (np.ndarray): 决策变量，形状为 (batch_size, x_dim)。
        返回:
            np.ndarray: 约束条件，形状为 (batch_size, num_constraints)。
        """
        batch_size = x.shape[0]
        g = np.full((batch_size, 6), 1e20, dtype=np.float32)  # 初始化约束条件为一个大数

        for i in range(batch_size):
            x_i = x[i]
            # 计算 g1 到 g6
            g1 = x_i[0] + x_i[1] - 2
            g2 = 6 - x_i[0] - x_i[1]
            g3 = 2 - x_i[0] + x_i[2]
            g4 = 2 - x_i[1] + 3 * x_i[2]
            g5 = 4 - (x_i[3] - 3) ** 2 - x_i[4]
            g6 = (x_i[4] - 3) ** 2 + x_i[5] - 4

            g[i] = [g1, g2, g3, g4, g5, g6]  # 存储约束条件

        return g
class zdt6:
    """
    多目标优化问题定义类（ZDT6）。
    包含：
    - 决策变量的上下界
    - 目标函数的定义
    """

    def __init__(self, x_bin_num):
        # 决策变量的上下界
        self.obj_dim = 2  # 目标空间维度（ZDT6 问题中有 2 个目标）
        self.x_low = np.zeros(10)  # 决策变量的下界，假设 x_dim = 10
        self.x_up = np.ones(10)  # 决策变量的上界
        self.x_dim = len(self.x_low)  # 决策空间维度
        self.x_bin_num = x_bin_num  # 每个决策变量的二进制位数
        self.inital_state = np.zeros(10)

    def obj(self, x):
        """
        计算 ZDT6 测试问题的目标函数值。
        参数:
            x (np.ndarray): 决策变量，形状为 (batch_size, x_dim)。
        返回:
            np.ndarray: 目标函数值，形状为 (batch_size, obj_dim)。
        """
        batch_size = x.shape[0]
        f = np.full((batch_size, self.obj_dim), 1e20, dtype=np.float32)  # 初始化目标函数值为大数

        for i in range(batch_size):
            x_i = x[i]
            # 计算 g(x)
            g = 1 + 9 * (np.sum(x_i[1:])/9) ** 0.25
            # 计算 f1
            f1 = 1 - np.exp(-4 * x_i[0]) * np.sin(6 * np.pi * x_i[0]) ** 6
            # 计算 f2
            h = 1 - (f1 / g) ** 2

            f2 = g * h
            f[i] = [f1, f2]  # 存储目标函数值

        return f

class zdt2:
    """
    多目标优化问题定义类（ZDT2）。
    包含：
    - 决策变量的上下界
    - 目标函数的定义
    """

    def __init__(self, x_bin_num):
        # 决策变量的上下界
        self.obj_dim = 2  # 目标空间维度（ZDT2 问题中有 2 个目标）
        self.x_low = np.zeros(30)  # 决策变量的下界，假设 x_dim = 30
        self.x_up = np.ones(30)  # 决策变量的上界
        self.x_dim = len(self.x_low)  # 决策空间维度
        self.x_bin_num = x_bin_num  # 每个决策变量的二进制位数
        self.inital_state = np.zeros(30)

    def obj(self, x):
        """
        计算 ZDT2 测试问题的目标函数值。
        参数:
            x (np.ndarray): 决策变量，形状为 (batch_size, x_dim)。
        返回:
            np.ndarray: 目标函数值，形状为 (batch_size, obj_dim)。
        """
        batch_size = x.shape[0]
        f = np.full((batch_size, self.obj_dim), 1e20, dtype=np.float32)  # 初始化目标函数值为大数

        for i in range(batch_size):
            x_i = x[i]
            # 计算 f1
            f1 = x_i[0]
            # 计算 g(x)
            g = 1 + 9 * np.sum(x_i[1:]) / (self.x_dim - 1)
            # 计算 h(x)
            h = 1 - (f1 / g) ** 2
            # 计算 f2
            f2 = g * h
            f[i] = [f1, f2]  # 存储目标函数值

        return f


class kursawe:
    """
    多目标优化问题定义类（kursawe）。
    包含：
    - 决策变量的上下界
    - 线性不等式约束
    - 目标函数的定义
    """

    def __init__(self,x_bin_num):
        # 决策变量的上下界
        self.obj_dim = 2  # 目标空间维度（Kursawe 问题中有 2 个目标）
        self.x_low = np.array([-5,-5,-5])
        self.x_up = np.array([5,5,5])
        self.x_dim = len(self.x_low)  # 决策空间维度
        self.inital_state = np.array([0,0,0])
        self.x_bin_num = x_bin_num

    def obj(self, x):
        """
        计算 Kursawe 测试问题的目标函数值。
        参数:
            x (np.ndarray): 决策变量，形状为 (batch_size, x_dim)。
        返回:
            np.ndarray: 目标函数值，形状为 (batch_size, obj_dim)。
        """
        batch_size = x.shape[0]
        f = np.full((batch_size, self.obj_dim), 1e20, dtype=np.float32)  # 初始化目标函数值为一个大数

        for i in range(batch_size):
            x_i = x[i]
            # 计算 Kursawe 目标函数
            f1 = -10 * sum(np.exp(-0.2 * np.sqrt(x_i[j] ** 2 + x_i[j + 1] ** 2)) for j in range(self.x_dim - 1))
            f2 = sum(np.abs(x_i[j]) ** 0.8 + 5 * np.sin(x_i[j] ** 3) for j in range(self.x_dim))
            f[i] = [f1, f2]  # 存储目标函数值

        return f


class MooSCH:
    def __init__(self, parameters, x_bin_num, idx, ws_method='PRBD', backend=None):
        # 决策变量的下界和上界
        self.obj_dim = 7  # 目标空间维度
        self.x_bin_num =  x_bin_num
        self.backend = backend or default_backend
        global my_fitnessfcn
        my_fitnessfcn = self.backend.my_fitnessfcn
        self.A, self.b, _, _, self.x_low, self.x_up, self.inital_state = self.backend.generGACon(idx)
        self.x_dim = len(self.x_low)  # 决策空间维度
        self.ws_method = ws_method.upper()
        if self.ws_method not in {'PRBD', 'PABD'}:
            raise ValueError("ws_method must be 'PRBD' or 'PABD'")
        try:
            self.init_Data = self.backend.generInitData(
                parameters, idx, ws_method=self.ws_method
            )
        except TypeError:
            self.init_Data = self.backend.generInitData(parameters, idx)

    def obj(self, x):
        """
        计算目标函数值，确保生成的 x 满足约束 A * x <= b。
        参数:
            x (np.ndarray): 决策变量，形状为 (batch_size, x_dim)。
        返回:
            np.ndarray: 目标函数值，形状为 (batch_size, obj_dim)。
        """
        batch_size = x.shape[0]

        f = np.full((batch_size, self.obj_dim), 1e20, dtype=np.float32)  # 初始化目标函数值为大数

        for i in range(batch_size):
            # 检查约束是否满足
            x_i = x[i]
            constraints = np.dot(self.A, x_i)  # 形状: (l_A,)
            violated = constraints > self.b  # 形状: (l_A,)

            if not np.any(violated):  # 如果没有违反约束
                f[i] = my_fitnessfcn(x[i], self.init_Data)  # 计算目标函数值v
        return f

# class MooSCH:
#     """
#     多目标优化问题定义类（MooSCH）。
#     包含：
#     - 决策变量的上下界
#     - 线性不等式约束
#     - 目标函数的定义
#     """
#
#     def __init__(self, parameters,x_bin_num,idx):
#         # 决策变量的下界和上界
#         self.obj_dim = 7  # 目标空间维度
#         self.x_bin_num =  x_bin_num
#         self.A, self.b, _, _, self.x_low, self.x_up, self.inital_state = generGACon(idx)
#         self.x_dim = len(self.x_low)  # 决策空间维度
#         self.init_Data = generInitData(parameters, idx)
#     def obj(self, x):
#         """
#         计算目标函数值，确保生成的 x 满足约束 A * x <= b。
#         参数:
#             x (np.ndarray): 决策变量，形状为 (batch_size, x_dim)。
#         返回:
#             np.ndarray: 目标函数值，形状为 (batch_size, obj_dim)。
#         """
#         batch_size = x.shape[0]
#         f = np.full((batch_size, self.obj_dim), 1e20, dtype=np.float32)  # 初始化目标函数值为大数
#
#         for i in range(batch_size):
#             # 检查约束是否满足
#             x_i = x[i]
#             constraints = np.dot(self.A, x_i)  # 形状: (l_A,)
#             violated = constraints > self.b  # 形状: (l_A,)
#
#             if not np.any(violated):  # 如果没有违反约束
#                 f[i] = my_fitnessfcn(x[i], self.init_Data)  # 计算目标函数值v
#         return f

class MooTestSettingA:
    """
    多目标优化测试设置类（MooTestSettingA）。
    处理：
    - 状态的编码与解码
    - 奖励的计算（包括支配关系奖励和多样性奖励）
    - 精英解集的管理
    """

    def __init__(self, moo_pro,  batch_size=1):
        """
        初始化测试设置。
        参数:
            moo_pro (MooSCH): 多目标优化问题实例。
            x_bin_num (int): 每个决策变量的二进制位数。
        """
        self.moo_pro = moo_pro
        self.x_bin_num = self.moo_pro.x_bin_num
        self.batch_size = batch_size
        self.bit_dec_list = 2 ** np.arange(self.x_bin_num - 1, -1, -1)  # 二进制权重（高位到低位）
        self.state0 = self._x_to_state(self.moo_pro.inital_state)
        # 全局精英解集，平坦列表
        self.elite_list = []  # 列表形式为 [(obj_value1, variable1), (obj_value2, variable2), ...]

    def _x_to_state(self, x):
        """
        将实际的决策变量转换为格雷编码的二进制状态矩阵。
        参数:
            x (np.ndarray): 实际决策变量，形状为 (batch_size, x_dim)。
        返回:
            np.ndarray: 格雷编码的二进制状态矩阵，形状为 (batch_size, x_dim, x_bin_num)。
        """
        # 确保 x 是 numpy 数组
        x = np.asarray(x).reshape(1,self.moo_pro.x_dim)
        # 将决策变量映射到整数范围
        integer_values = ((x - self.moo_pro.x_low) * (2 ** self.x_bin_num - 1) / (
                    self.moo_pro.x_up - self.moo_pro.x_low)).astype(int)
        # 创建一个格雷编码状态矩阵
        state = np.zeros((x.shape[0], x.shape[1], self.x_bin_num), dtype=int)

        # 对每个整数值转换为格雷编码并填充到状态矩阵
        for i in range(x.shape[0]):  # 对每个 batch 处理
            for j in range(x.shape[1]):  # 对每个决策变量处理
                gray_code = self._int_to_gray(integer_values[i, j])
                state[i, j] = np.array([int(bit) for bit in gray_code])  # 将格雷编码转换为整数数组

        return state

    def _state_to_x(self, state):
        """
        将格雷编码的状态矩阵转换为实际的决策变量。
        参数:
            state (np.ndarray): 格雷编码的二进制矩阵，形状为 (batch_size, x_dim, x_bin_num)。
        返回:
            np.ndarray: 实际决策变量，形状为 (batch_size, x_dim)。
        """
        # 确保 state 是 numpy 数组
        state = np.asarray(state)

        # 计算每个决策变量的整数值

        # state 的形状: (batch_size, x_dim, x_bin_num)
        integer_values = np.sum(state * self.bit_dec_list, axis=2)  # 形状: (batch_size, x_dim)

        # 将格雷编码转换为二进制
        binary_values = np.array([self._gray_to_int(gray) for gray in integer_values])

        # 缩放到决策空间范围
        x = binary_values * (self.moo_pro.x_up - self.moo_pro.x_low) / (2 ** self.x_bin_num - 1) + self.moo_pro.x_low

        return x.astype(np.float32)  # 形状: (batch_size, x_dim)



    def obj_value(self, state):
        """
        计算当前状态下的目标函数值。
        参数:
            state (np.ndarray): 当前状态，形状为 (batch_size, x_dim * x_bin_num)。
        返回:
            tuple: (目标函数值列表，实际决策变量 x)
        """
        # 重塑状态为 (batch_size, x_dim, x_bin_num)
        state = state.reshape(self.batch_size, self.moo_pro.x_dim, self.x_bin_num)
        x = self._state_to_x(state)  # 形状: (batch_size, x_dim)
        y = self.moo_pro.obj(x)  # 形状: (batch_size, obj_dim)
        return y, x  # 返回目标值和决策变量

    def reward_value(self, elite_list, obj_value, variable):
        """
        计算奖励，结合支配关系奖励和多样性奖励。
        参数:
            elite_list (list): 当前的精英解列表，形式为 [(obj_value1, variable1), (obj_value2, variable2), ...]。
            obj_value (np.ndarray): 当前解的目标值，形状为 (batch_size, obj_dim)。
            variable (np.ndarray): 当前解的决策变量，形状为 (batch_size, x_dim)。
        返回:
            np.ndarray: 奖励值，形状为 (batch_size,)。
        """
        rewards = np.zeros(self.batch_size, dtype=np.float32)

        for i in range(self.batch_size):
            y = obj_value[i]
            x = variable[i]
            y_j = y > 1e19
            if np.any(y_j):
                rewards[i] = 0
                continue
            # 打印当前样本的目标值（调试）
            # print(f"Sample {i}: obj_value = {y}")
            # 检查支配关系
            dominance = self._check_dominance(y, elite_list)

            # 更新精英解集
            dominance_status = self._update_elite_list(y, x, dominance)

            # 计算支配关系奖励
            dominance_reward = self._compute_dominance_reward(dominance_status, y)

            # # 计算多样性奖励
            # diversity_reward = self._compute_simplified_crowding_distance(dominance_status, y, K=2, num_neighbors=5)

            # 合并奖励
            rewards[i] = dominance_reward
            #满足约束奖励
            rewards[i] += 0.1
        return rewards, self.elite_list  # 形状: (batch_size,)

    def _check_dominance(self, obj_value, elite_list):
        """
        检查当前解是否被支配或是否支配精英解列表中的某些解。
        参数:
            obj_value (np.ndarray): 当前解的目标值，形状为 (obj_dim,)。
            elite_list (list): 全局精英解列表，形式为 [(obj_value1, variable1), ...]。
        返回:
            dict: {'dominated': bool, 'dominate_others': bool}
        """
        dominated = False
        dominate_others = False
        for elite_obj, _ in elite_list:
            if self._is_dominated(elite_obj, obj_value):
                # 当前解支配精英解
                dominate_others = True
            if self._is_dominated(obj_value, elite_obj):
                # 当前解被精英解支配
                dominated = True
        return {'dominated': dominated, 'dominate_others': dominate_others}

    def _is_dominated(self, obj1, obj2):
        """
        判断 obj1 是否被 obj2 支配。
        参数:
            obj1 (np.ndarray): 第一个目标值。
            obj2 (np.ndarray): 第二个目标值。
        返回:
            bool: 如果 obj1 被 obj2 支配，返回 True，否则返回 False。
        """
        return np.all(obj2 <= obj1) and np.any(obj2 < obj1)

    def _update_elite_list(self, obj_value, variable, dominance):
        """
        根据支配关系更新精英解列表。
        参数:
            obj_value (np.ndarray): 当前解的目标值。
            variable (np.ndarray): 当前解的决策变量。
            dominance (dict): 支配关系。
        返回:
            int: dominance_status（0: no dominance, 1: dominated, 2: dominate others）
        """
        dominated = dominance['dominated']
        dominate_others = dominance['dominate_others']
        dominance_status = 0  # 0: no dominance, 1: dominated, 2: dominate others

        if dominated:
            dominance_status = 1
            # 不更新精英列表
        elif dominate_others:
            dominance_status = 2
            # 移除被当前解支配的精英解
            new_elites = []
            for elite_obj, elite_var in self.elite_list:
                if not self._is_dominated(elite_obj, obj_value):
                    new_elites.append((elite_obj, elite_var))
            # 添加当前解到精英解列表
            new_elites.append((obj_value, variable))
            self.elite_list = new_elites
        else:
            # 检查是否已存在相似的解
            exists = False
            for elite_obj, _ in self.elite_list:
                if np.allclose(elite_obj, obj_value, atol=1e-5):
                    exists = True
                    break
            if not exists:
                self.elite_list.append((obj_value, variable))

        return dominance_status

    def _count_improvements(self, obj_value, elite_list):
        """
        计算当前解在多少个目标上优于精英解集中的解。
        参数:
            obj_value (np.ndarray): 当前解的目标值。
            elite_list (list): 全局精英解列表。
        返回:
            int: 优于精英解集的目标数。
        """
        if not elite_list:
            return self.moo_pro.obj_dim

        elite_objs = np.array([elite[0] for elite in elite_list])  # (num_elites, obj_dim)
        current_obj = np.array(obj_value)  # (obj_dim,)

        # 计算每个目标在当前解中优于精英解集的比例
        improvements = np.sum(current_obj < elite_objs, axis=0)  # (obj_dim,)

        # 返回有改善的目标数
        improvement_count = np.sum(improvements > 0)

        return improvement_count

    def _compute_dominance_reward(self, dominance_status, obj_value):
        """
        计算基于支配关系的奖励。
        参数:
            dominance_status (int): 支配状态，0: 无支配，1: 被支配，2: 支配其他。
            obj_value (np.ndarray): 当前解的目标值。
        返回:
            float: 支配关系奖励。
        """
        if dominance_status == 2:
            # 当前解支配其他解，给予较高奖励
            return 2
        elif dominance_status == 1:
            # 当前解被支配，不给予负奖励
            return 0.0
        else:
            # 非支配且不支配，基于优化目标数给予奖励
            improvement_count = self._count_improvements(obj_value, self.elite_list)
            weight_score = 2 / self.moo_pro.obj_dim
            return weight_score * improvement_count  # 可根据需要调整比例

    def _compute_simplified_crowding_distance(self, dominance_status, obj_value, K=2, num_neighbors=5):
        """
        简化的拥挤距离计算方法。
        参数：
            dominance_status (int): 支配状态。
            obj_value (np.ndarray): 当前解的目标值。
            K (int): 用于计算拥挤距离的目标维度数量，默认为 2。
            num_neighbors (int): 近邻数量，默认为 5。
        返回:
            float: 简化的拥挤距离。
        """
        if dominance_status == 1:
            return 0.0

        elite_objs = np.array([elite_obj for elite_obj, _ in self.elite_list])  # (N, K)
        if elite_objs.size == 0:
            return 0.0

        # Step 1: 选择用于计算的目标维度
        M = self.moo_pro.obj_dim
        if K is None or K >= M:
            selected_dimensions = np.arange(M)
        else:
            # 为了简化，这里选择前 K 个目标维度
            selected_dimensions = np.arange(K)

        # 提取选定维度的目标值
        selected_solutions = elite_objs[:, selected_dimensions]  # (N, K)

        # 提取当前解的选定维度目标值
        current_selected_obj = obj_value[selected_dimensions]  # (K,)

        # Step 2: 对目标值进行归一化
        f_min = np.min(selected_solutions, axis=0)
        f_max = np.max(selected_solutions, axis=0)
        normalized_solutions = (selected_solutions - f_min) / (f_max - f_min + 1e-9)  # (N, K)
        normalized_current = (current_selected_obj - f_min) / (f_max - f_min + 1e-9)  # (K,)

        # Step 3: 构建KD树，加速最近邻搜索
        tree = KDTree(normalized_solutions)

        # Step 4: 计算每个解的局部邻域距离
        crowding_distance = 0.0
        if selected_solutions.shape[0] <= 1:
            return crowding_distance

        distances, indices = tree.query(normalized_current, k=num_neighbors + 1)  # 包含自身
        # 排除自身（距离为0的点）
        mask = distances > 1e-9
        valid_distances = distances[mask]

        # 如果没有足够的邻居，使用最大距离代替
        if len(valid_distances) < num_neighbors:
            max_distance = np.max(distances)
            valid_distances = np.append(valid_distances, [max_distance] * (num_neighbors - len(valid_distances)))

        # 拥挤距离定义为邻域内距离的倒数之和
        crowding_distance = np.sum(1.0 / (valid_distances + 1e-9))

        return crowding_distance

    """
    格雷编码+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    def state_update(self, state, action):
        """
        使用按位异或操作更新状态，基于格雷编码。
        参数:
            state (np.ndarray): 当前状态，形状为 (batch_size, x_dim * x_bin_num)。
            action (np.ndarray): 动作，形状为 (batch_size, x_dim * x_bin_num)。
        返回:
            np.ndarray: 更新后的状态，形状为 (batch_size, x_dim * x_bin_num)。
        """
        # 确保输入为 numpy 数组并转换为整数类型
        state = np.asarray(state, dtype=np.uint8)
        action = np.asarray(action, dtype=np.uint8)
        # 将动作与状态进行异或操作（在格雷编码空间下）
        updated_state_bin = np.bitwise_xor(state, action)

        return updated_state_bin.astype(np.uint8)

    def _int_to_gray(self, n):
        """
        将整数值转换为格雷编码。
        参数:
            n (int): 整数值。
        返回:
            str: 格雷编码字符串。
        """
        return format(n ^ (n >> 1), f'0{self.x_bin_num}b')

    import numpy as np

    def _gray_to_int(self, g):
        """
        将格雷编码转换为原始整数值。
        参数:
            g (np.ndarray): 格雷编码，形状为 (bvc n , ) 或 (batch_size, x_dim)。
        返回:
            np.ndarray: 原始整数值，形状为 (batch_size, ) 或 (batch_size, x_dim)。
        """
        # 如果 g 是 ndarray，处理每个元素
        batch_size = g.shape[0]
        result = np.zeros(batch_size, dtype=int)  # 创建一个存放结果的数组

        for i in range(batch_size):
            gray_value = int(g[i])  # 将格雷编码转换为整数
            mask = gray_value
            while mask:
                mask >>= 1
                gray_value ^= mask
            result[i] = gray_value  # 保存转换后的整数

        return result

    # def gray_to_binary(self, gray):
    #     """
    #     将格雷码转换为二进制码。
    #     参数:
    #         gray (np.ndarray): 格雷码，形状为 (batch_size, x_dim, x_bin_num)。
    #     返回:
    #         np.ndarray: 二进制码，形状为 (batch_size, x_dim, x_bin_num)。
    #     """
    #     binary = np.copy(gray)
    #     for i in range(1, binary.shape[2]):  # 第三维度是格雷码的进制位
    #         binary[:, :, i] ^= binary[:, :, i - 1]  # 对每一位进行异或操作
    #     return binary

    def gray_to_binary(self, gray):
        # gray: (B,D,BIT)
        binary = gray.copy()
        # cumulative XOR along last axis
        np.cumsum(gray, axis=2, out=binary, dtype=int)
        return binary % 2

    def binary_to_gray(self, binary):
        """
        将二进制码转换为格雷码。
        参数:
            binary (np.ndarray): 二进制码，形状为 (batch_size, x_dim, x_bin_num)。
        返回:
            np.ndarray: 格雷码，形状为 (batch_size, x_dim, x_bin_num)。
        """
        gray = np.copy(binary)

        # 对每个batch，每个决策变量，每个位进行转换
        for i in range(binary.shape[0]):  # batch size
            for j in range(binary.shape[1]):  # x_dim
                # 对每个决策变量的每个位进行转换
                gray[i, j, 0] = binary[i, j, 0]  # 格雷码的第一位与二进制的第一位相同
                for k in range(1, binary.shape[2]):  # 从第二位开始
                    gray[i, j, k] = binary[i, j, k] ^ binary[i, j, k - 1]  # 按位异或转换

        return gray

    """
    格雷编码
    """
    def initial_state(self, seed=None):
        """
        初始化随机二进制状态。
        参数:
            seed (int, 可选): 随机种子，用于可重复性。
        返回:
            np.ndarray: 随机初始化的状态，形状为 (batch_size, x_dim * x_bin_num)。
        """
        if seed is not None:
            np.random.seed(seed)
        a = self.state0
        # 将格雷码转换为二进制
        a_bin = self.gray_to_binary(a)

        # 对二进制进行加1操作
        a_bin[0, :, -1] = (a_bin[0, :, -1] + 1) % 2  # 对最后一位进行加1操作，并模2确保是0或1
        carry = (a_bin[0, :, -1] == 0).astype(int)  # 如果有进位，carry为1
        carry_c = carry
        for i in range(a_bin.shape[2] - 2, -1, -1):  # 从倒数第二位开始处理进位
            a_bin[0, :, i] = (a_bin[0, :, i] + carry) % 2
            carry = (a_bin[0, :, i] == 0).astype(int) & carry_c
            carry_c = carry

        # 将加1后的二进制值转换回格雷码
        a_gray = self.binary_to_gray(a_bin)

        b = np.random.randint(0, 2, (self.batch_size-1, self.moo_pro.x_dim * self.x_bin_num)).astype(np.uint8)
        # 确保 'a' 的形状为 (1, x_dim * x_bin_num)
        a_gray = a_gray.reshape(1, -1)

        # 拼接 'a' 和 'b'，得到形状为 (batch_size, x_dim * x_bin_num)
        initial_states = np.concatenate([a_gray, b], axis=0)

        return initial_states



class MooEnv:
    """
    强化学习环境类（MooEnv），用于多目标优化问题。

    特点：
    - 状态定义为决策变量的二进制编码
    - 奖励基于支配关系奖励和多样性奖励
    - 支持随机探索策略（ε-贪婪策略）
    """

    def __init__(self, moo_test, max_steps=50000, exploration_epsilon=0.05):
        """
        初始化环境。

        参数:
            moo_test (MooTestSettingA): 多目标测试设置实例。
            max_steps (int): 最大步数，环境终止条件之一。
            exploration_epsilon (float): ε-贪婪策略中的探索概率。
        """
        self.obj_func = moo_test.obj_value
        self.red_func = moo_test.reward_value
        self.sta_func = moo_test.state_update
        self.initial_state_func = moo_test.initial_state

        self.elite_list = moo_test.elite_list  # 全局精英解列表，形式为 [(obj_value1, variable1), ...]
        self.state = None
        self.object = None
        self.variable = None
        self.max_steps = max_steps
        self.current_step = 0

        # 探索参数
        self.exploration_epsilon = exploration_epsilon

    def reset(self, seed=None):
        """
        重置环境到初始状态。

        参数:
            seed (int, 可选): 随机种子，用于初始化状态。

        返回:
            np.ndarray: 初始状态，形状为 (batch_size, x_dim * x_bin_num)。
        """
        initial_state = self.initial_state_func(seed=seed)
        self.state = initial_state
        obj_fun_values, variable = self.obj_func(self.state)
        self.object = obj_fun_values
        self.variable = variable
        # 初始化精英解集为初始解
        self.elite_list = []
        for i in range(self.state.shape[0]):
            y = obj_fun_values[i]
            x = variable[i]
            # 仅添加满足约束的解
            if not np.any(abs(y-1e20) <= 1):
                self.elite_list.append((y, x))
        self.current_step = 0
        return self.state  # 返回初始状态

    def step(self, action):
        """
        执行动作，更新状态并计算奖励。
        参数:
            action (np.ndarray): 动作，形状为 (batch_size, x_dim * x_bin_num)。
        返回:
            tuple:
                state (np.ndarray): 更新后的状态，形状为 (batch_size, x_dim * x_bin_num)。
                reward (np.ndarray): 奖励值，形状为 (batch_size,)。
                done (np.ndarray): 是否终止，形状为 (batch_size,)。
                info (dict): 额外信息。
        """
        # 应用动作更新状态
        state_next = self._next_state(action)

        # 计算新状态下的目标函数值
        obj_fun_values, variable = self.obj_func(state_next)  # (batch_size, obj_dim), (batch_size, x_dim)

        # 计算奖励并更新精英解集
        reward, elite_update = self.red_func(self.elite_list, obj_fun_values, variable)  # (batch_size,)
        self.elite_list = elite_update
        # 更新环境状态
        self.state = state_next
        self.object = obj_fun_values
        self.variable = variable

        return self.state, reward

    def _next_state(self, action):
        """
        根据动作更新状态。

        参数:
            action (np.ndarray): 动作，形状为 (batch_size, x_dim * x_bin_num)。

        返回:
            np.ndarray: 更新后的状态，形状为 (batch_size, x_dim * x_bin_num)。
        """
        return self.sta_func(self.state, action)

    def select_action(self,  actions):
        """
        根据策略选择动作，结合 ε-贪婪策略进行随机探索。

        参数:
            policy (callable): 策略函数，接受状态并返回动作。
            state (np.ndarray): 当前状态，形状为 (batch_size, x_dim * x_bin_num)。

        返回:
            np.ndarray: 选择的动作，形状为 (batch_size, x_dim * x_bin_num)。
        """
        if np.random.rand() < self.exploration_epsilon:
            # 随机选择动作
            actions = np.random.randint(0, 2, actions.shape).astype(np.uint8)
        return actions

    @staticmethod
    def sample_policy(state):
        """
        示例策略函数，基于当前状态选择动作。
        这里简单地返回与当前状态不同的动作。

        参数:
            state (np.ndarray): 当前状态，形状为 (batch_size, x_dim * x_bin_num)。

        返回:
            np.ndarray: 动作，形状为 (batch_size, x_dim * x_bin_num)。
        """
        # 例如，随机翻转部分位
        flip_mask = np.random.randint(0, 2, state.shape).astype(np.uint8)
        action = flip_mask
        return action
