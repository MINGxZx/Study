import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib
from IPython import display


class GA:
    def __init__(self, nums, bound, func, DNA_SIZE=None, cross_rate=0.8, mutation=0.003):
        nums = np.array(nums)
        bound = np.array(bound)
        self.bound = bound
        if nums.shape[1] != bound.shape[0]:
            raise Exception('范围的数量与变量的数量不一致， 您有{}个变量，却有{}个范围'.format(nums.shape[1], bound.shape[0]))

        for var in nums:
            for index, var_curr in enumerate(var):
                if var_curr < bound[index][0] or var_curr > bound[index][1]:
                    raise Exception('{}不在取值范围内'.format(var_curr))

        for min_bound, max_bound in bound:
            if max_bound < min_bound:
                raise Exception('抱歉，({}, {})不是合格的取值范围'.format(min_bound, max_bound))

        min_nums, max_nums = np.array(list(zip(*bound)))        # 所有变量的最大最小值
        var_len = max_nums - min_nums  # 所有变量的取值范围大小        # 所有变量的取值范围
        self.var_len = var_len
        bits = np.ceil(np.log2(var_len + 1))  # 每个变量按整数编码最小的二级制位数

        if DNA_SIZE is None:
            DNA_SIZE = int(np.max(bits))
        self.DNA_SIZE = DNA_SIZE

        self.POP_SIZE = len(nums)       # 进化的种群数
        POP = np.zeros((*nums.shape, DNA_SIZE))
        for i in range(nums.shape[0]):
            for j in range(nums.shape[1]):
                num = int(round((nums[i, j] - bound[j][0]) * ((2 ** DNA_SIZE) / var_len[j])))  # 编码方式
                POP[i, j] = [int(k) for k in ('{0:0' + str(DNA_SIZE) + 'b}').format(num)]
        self.POP = POP
        self.copy_POP = POP.copy()
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = func

    # 解码：二进制转十进制(用向量相乘的方式)
    def translateDNA(self):
        W_vector = np.array([2 ** i for i in range(self.DNA_SIZE)]).reshape((self.DNA_SIZE, 1))[::-1]
        binary_vector = self.POP.dot(W_vector).reshape(self.POP.shape[0:2])
        for i in range(binary_vector.shape[0]):
            for j in range(binary_vector.shape[1]):
                binary_vector[i, j] /= ((2 ** self.DNA_SIZE) / self.var_len[j])
                binary_vector[i, j] += self.bound[j][0]
        return binary_vector

    # 计算适应度
    def get_fitness(self, non_negative=False):
        result = self.func(*np.array(list(zip(*self.translateDNA()))))
        if non_negative:
            min_fit = np.min(result, axis=0)
            result -= min_fit
        if np.all(result == 0):
            result = result + min_fit
        return result

    # 选择
    def select(self):
        fitness = self.get_fitness(non_negative=True)
        self.POP = self.POP[np.random.choice(np.arange(self.POP.shape[0]), size=self.POP.shape[0],
                                             replace=True, p=fitness / np.sum(fitness))]

    # 交叉
    def crossover(self):
        for people in self.POP:
            if np.random.rand() < self.cross_rate:
                i_ = np.random.randint(0, self.POP.shape[0], size=1)
                cross_points = np.random.randint(0, 2, size=(len(self.var_len), self.DNA_SIZE)).astype(np.bool)
                people[cross_points] = self.POP[i_, cross_points]

    # 变异
    def mutate(self):
        for people in self.POP:
            for var in people:
                for point in range(self.DNA_SIZE):
                    if np.random.rand() < self.mutation:
                        var[point] = 1 if var[point] == 0 else 1

    # 进化
    def evolution(self):
        self.select()
        self.crossover()
        self.mutate()

    def reset(self):
        self.POP = self.copy_POP.copy()

    def log(self):
        return pd.DataFrame(np.hstack((self.translateDNA(), self.get_fitness().reshape((len(self.POP), 1)))),
                            columns=[f'x{i}' for i in range(len(self.var_len))] + ['F'])

    def plot_in_jupyter_1d(self, iter_time=500):
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display
        r = {}
        plt.ion()
        for _ in range(iter_time):
            plt.clf()
            x = np.linspace(*self.bound[0], self.var_len[0]*50)
            plt.plot(x, self.func(x))
            x = self.translateDNA().reshape(self.POP_SIZE)
            r[x.max()] = self.func(x).max()
            plt.scatter(x, self.func(x), s=200, lw=0, c='red', alpha=0.5)
            if is_ipython:
                display.clear_output(wait=True)
                display.display(plt.gcf())

            self.evolution()
            plt.pause(0.01)

        print(max(r.items(), key=lambda e: e[1]))
        # xtick = max(r.items(), key=lambda e: e[1])[0]
        # ytick = max(r.items(), key=lambda e: e[1])[1]
        # plt.scatter(xtick, ytick, s=200, lw=0, c='red', alpha=0.5)
        plt.pause(0)


func = lambda x: np.sin(10 * math.pi * x) * x + 2.0
ga = GA([[np.random.rand()] for _ in range(22)], [(-1, 2)], DNA_SIZE=22, func=func)
ga.plot_in_jupyter_1d()

