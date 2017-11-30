#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

def show_img(img,window_name='img'):
    cv2.imshow(window_name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_img(img):
    # 压缩图像至80*80
    img=cv2.resize(img, (80, 80))
    # 转换为灰阶图像
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 二值化图像
    _, img = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
    return img

def create_network():
    # 输入层 80（高）*80（宽）*4（连续4幅图像的在该位置像素值）
    input_layer = tf.placeholder("float", [None, 80, 80, 4])

    # 根据shape生成truncated normal分布的权重张量
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)
    
    # 根据shape生成truncated normal分布的偏差张量
    def bias_variable(shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    # 2d图像卷积
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    # 隐藏层
    # 卷积池化层1
    W_conv1 = weight_variable([8, 8, 4, 32]) 
    b_conv1 = bias_variable([32]) 
    h_conv1 = tf.nn.relu(conv2d(input_layer, W_conv1, 4) + b_conv1)
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
    h_pool1 = max_pool_2x2(h_conv1)

    # 卷积层2
    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    # 卷积层3
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    
    # 全连接层
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # 输出层
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return input_layer, readout, h_fc1

def train_network(s, readout, h_fc1, sess):
    # 定义误差函数
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    
    # 训练目标
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # 经验内存
    mem = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    
    # 得到初始状态，预处理图像压缩至80*80灰阶
    img, _, _ = game_state.frame_step(do_nothing)
    img=preprocess_img(img)
    state = np.stack((img, img, img, img), axis=2)

    # 保存、加载网络
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # 开始训练
    epsilon = INITIAL_EPSILON
    t = 0
    while True:
        # choose an action epsilon-greedily
        readout_t = readout.eval(feed_dict={s : [state]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # 随着游戏训练进行逐渐降低epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 输入所选动作并观察下一时刻状态及收益
        img, reward, terminal = game_state.frame_step(a_t)
        img=preprocess_img(img)
        img = np.reshape(img, (80, 80, 1))
        state_ = np.append(img, state[:, :, :3], axis=2)

        # 存储状态至经验内存
        mem.append((state, a_t, reward, state_, terminal))
        
        if len(mem) > REPLAY_MEMORY:
            mem.popleft()

        # only train if done observing
        if t > OBSERVE:
            # 从经验内存中抽样作为训练样本
            minibatch = random.sample(mem, BATCH)

            # 批处理变量
            S = [d[0] for d in minibatch]
            A = [d[1] for d in minibatch]
            R = [d[2] for d in minibatch]
            S_ = [d[3] for d in minibatch]

            Y = []
            S_readout = readout.eval(feed_dict = {s : S_})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    Y.append(R[i])
                else:
                    Y.append(R[i] + GAMMA * np.max(S_readout[i]))

            # 使用梯度下降算法调整网络权重
            train_step.run(feed_dict = {
                y : Y,
                a : A,
                s : S}
            )

        # 更新状态
        state=state_
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        phase = ""
        if t <= OBSERVE:
            phase = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            phase = "explore"
        else:
            phase = "train"

        print("timestep", t,
              " phase", phase,
              " epsilon", epsilon,
              " action", action_index,
              " reward", reward,
              " q_max %e" % np.max(readout_t))
        
        ### write info to files
        # if t % 10000 <= 100:
        #     a_file.write(",".join([str(x) for x in readout_t]) + '\n')
        #     h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
        #     cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = create_network()
    train_network(s, readout, h_fc1, sess)
