"""
逻辑回归（参照 PPT【例8.4】）—— 参考答案
已补充 4 个函数：sigmoid()、loss()、gradient()、Logistic_Regression()
"""
import numpy as np


def sigmoid(z):
    """
    函数 1　定义 Sigmoid 函数
    功能：计算 1 / (1 + e^(-z))
    输入：z：一个数，或 numpy 数组
    输出：与 z 形状相同的结果，取值在 0～1 之间
    """
    return 1 / (1 + np.exp(-z))


def loss(h, y):
    """
    函数 2　定义对数损失函数
    功能：计算所有样本对数损失的平均值：平均值( -y × log(h) - (1 - y) × log(1 - h) )
    输入：h：sigmoid 函数的结果（每名同学挂科的概率），numpy 数组，形状 (m,)
          y：真实标签，numpy 数组，形状 (m,)，1 表示挂科，0 表示没挂科
    输出：损失值（一个数）
    """
    # 裁剪 h，避免 log(0) 出现无穷大 / nan
    eps = 1e-15
    h = np.clip(h, eps, 1 - eps)
    return np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))


def gradient(X, h, y):
    """
    函数 3　定义梯度计算函数
    功能：用矩阵计算梯度：X 的转置 × (h - y) ÷ m，m 是样本个数
    输入：X：第一列全为 1 的特征矩阵，形状 (m, 3)
          h：sigmoid 函数的结果，形状 (m,)
          y：真实标签，形状 (m,)
    输出：梯度，numpy 数组，形状 (3,)，与参数 w 一一对应
    """
    m = y.size
    return np.dot(X.T, (h - y)) / m


def Logistic_Regression(x, y, lr, num_iter, threshold):
    """
    函数 4　逻辑回归过程（梯度下降）
    功能：
      1. 在 x 前面加一列 1，得到 X；参数 w 全部初始化为 0
      2. 梯度下降迭代，每一轮：
           z = X 与 w 的矩阵乘积 → h = sigmoid(z) → 计算梯度 → w = w - lr × 梯度
           → 用更新后的 w 重新算 h，计算损失，存入 loss_list
      3. 收敛判断：从第 2 轮起，本轮损失与上一轮损失之差的绝对值 < threshold 时，停止迭代；
         迭代满 num_iter 轮也停止
    输入：x：特征，numpy 数组，形状 (m, 2)（不含 1 那一列）
          y：标签，numpy 数组，形状 (m,)
          lr：学习率
          num_iter：最多迭代轮数
          threshold：收敛阈值
    输出：loss_list：列表，loss_list[i] 是第 i+1 轮的损失；迭代了多少轮就是 len(loss_list)
          w：训练好的参数 [w0, w1, w2]，w0 是截距
    """
    m = x.shape[0]
    # ① 在 x 前加一列 1 得到 X，参数 w 全部初始化为 0
    X = np.concatenate((np.ones((m, 1)), x), axis=1)
    w = np.zeros(X.shape[1])
    loss_list = []

    for i in range(num_iter):
        # ② z = X·w → h = sigmoid(z)
        z = np.dot(X, w)
        h = sigmoid(z)
        # ③ 计算梯度  ④ 更新参数
        g = gradient(X, h, y)
        w = w - lr * g
        # ⑤ 用更新后的 w 重新算 h，计算损失，存入 loss_list
        h_new = sigmoid(np.dot(X, w))
        current_loss = loss(h_new, y)
        loss_list.append(current_loss)
        # ⑥ 从第 2 轮起，相邻两轮损失之差的绝对值 < threshold 即停止
        if i >= 1 and abs(loss_list[-1] - loss_list[-2]) < threshold:
            break

    return loss_list, w
