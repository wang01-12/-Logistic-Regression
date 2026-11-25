"""
画图
已补充 3 个函数：draw_loss()、draw_learning_rate()、draw_decision_boundary()
"""
import numpy as np
import matplotlib.pyplot as plt

from logistic_regression import Logistic_Regression


def draw_loss(loss_list):
    """
    函数 5　画迭代轮数与损失函数的关系图
    功能：横轴为迭代轮数，纵轴为损失函数值，画出损失随迭代轮数变化的曲线
    输入：loss_list：Logistic_Regression() 返回的损失列表
    输出：无（只画图，不调用 plt.show()）
    """
    plt.figure()
    plt.plot(range(1, len(loss_list) + 1), loss_list, color='tab:blue')
    plt.xlabel('迭代轮数')
    plt.ylabel('损失函数值')
    plt.title('损失函数随迭代轮数的变化')
    plt.grid(True, linestyle='--', alpha=0.5)


def draw_learning_rate(x, y, lr_list, num_iter):
    """
    函数 6　画不同学习率下损失函数的变化图
    功能：对 lr_list 中的每个学习率，调用 Logistic_Regression() 各训练 num_iter 轮
          （threshold 取 0，不提前停止），把各自的损失曲线画在同一张图上，用图例标出学习率
    输入：x、y：特征和标签
          lr_list：学习率列表
          num_iter：迭代轮数
    输出：无（只画图，不调用 plt.show()）
    """
    plt.figure()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i, lr in enumerate(lr_list):
        loss_list, _ = Logistic_Regression(x, y, lr, num_iter, 0)
        plt.plot(range(1, len(loss_list) + 1), loss_list,
                 color=colors[i % len(colors)], label='学习率 = {}'.format(lr))
    plt.xlabel('迭代轮数')
    plt.ylabel('损失函数值')
    plt.title('不同学习率下损失函数的变化')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)


def draw_decision_boundary(x, y, w):
    """
    函数 7　画分类线
    功能：画出所有同学的散点图（挂科和没挂科用不同颜色、大小区分），
          再画出分类线 w0 + w1 × x1 + w2 × x2 = 0
    输入：x：特征，形状 (m, 2)，第 1 列学习时长，第 2 列刷手机时长
          y：标签，形状 (m,)
          w：Logistic_Regression() 返回的参数 [w0, w1, w2]
    输出：无（只画图，不调用 plt.show()）
    """
    plt.figure()
    # 没挂科(y=0)与挂科(y=1)用不同颜色、大小区分
    plt.scatter(x[y == 0, 0], x[y == 0, 1], c='tab:blue', s=25,
                label='没挂科', alpha=0.7)
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c='tab:red', s=60,
                marker='^', label='挂科', alpha=0.8)

    # 分类线：w0 + w1*x1 + w2*x2 = 0  →  x2 = -(w0 + w1*x1) / w2
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x1_line = np.linspace(x1_min, x1_max, 100)
    x2_line = -(w[0] + w[1] * x1_line) / w[2]
    plt.plot(x1_line, x2_line, 'k-', linewidth=2, label='分类线')

    plt.xlabel('每周学习时长（小时）')
    plt.ylabel('每天刷手机时长（小时）')
    plt.title('逻辑回归分类线')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
