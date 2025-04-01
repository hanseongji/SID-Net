import math
import numpy as np
import os
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import csv

example_idx_=2
dirSep = "/" if platform.system() == "Darwin" else "\\"
curr_path = f"{os.getcwd()}{dirSep}Numerical_Problem{example_idx_}{dirSep}"

model_path = f"{curr_path}DNN_Model_Ex2.h5"
loss_Path = f"{curr_path}loss.csv"
opt_res = f"{curr_path}opt_res.csv"

Loss_Data = f"{curr_path}Loss_Opt.csv"
Output_Data = f"{curr_path}Output_Opt.csv" 
model = tf.keras.models.load_model(model_path)

# Target
test_Vel_x_c = []
test_Vel_x_B = []
test_Vel_y_B = []
test_Vel_beta_c = []
test_beta = []

l_c1= 0.3285
w_01 = 4.43
Theta_01 = 12.86
np_Time = np.linspace(start = 0, stop = 5, num = 501)

for dt in np_Time:
    r_c = 2 * l_c1
    vel_y_b = l_c1 * np.cos(w_01 * dt + Theta_01 * math.pi / 180) * w_01
    vel_x_b = -1 * l_c1 * np.sin(w_01 * dt + Theta_01 * math.pi / 180) * w_01
    y_b = l_c1 * np.sin(w_01 * dt + Theta_01 * math.pi / 180)
    beta_c = np.arcsin(y_b/r_c)
    vel_beta_c = l_c1 * np.cos(w_01 * dt + Theta_01 * math.pi / 180) * w_01 / r_c / np.cos(beta_c)
    vel_x_c = -1 * l_c1 * np.sin(w_01 * dt + Theta_01 * math.pi / 180) * w_01 - r_c * np.sin(beta_c) * vel_beta_c
    test_Vel_x_c.append(vel_x_c)
    test_Vel_x_B.append(vel_x_b)
    test_Vel_y_B.append(vel_y_b)
    test_Vel_beta_c.append(vel_beta_c)
    test_beta.append(beta_c)
test_Vel_x_c = np.array(test_Vel_x_c)
test_Vel_x_B = np.array(test_Vel_x_B)
test_Vel_y_B = np.array(test_Vel_y_B)
test_Vel_beta_c = np.array(test_Vel_beta_c)
n_zeros = np.zeros(501)
rms_Vel_x_c = np.sqrt(np.sum((test_Vel_x_c-n_zeros)**2)*1/501)
rms_Vel_x_B = np.sqrt(np.sum((test_Vel_x_B - n_zeros) ** 2)*1/501)
rms_Vel_y_B = np.sqrt(np.sum((test_Vel_y_B - n_zeros) ** 2)*1/501)
rms_Vel_beta_c = np.sqrt(np.sum((test_Vel_beta_c - n_zeros) ** 2)*1/501)

Target_Vel_x_c = (rms_Vel_x_c - 0.211779460)/1.621362
Target_Vel_x_B = (rms_Vel_x_B - 0.21779784)/1.55288368
Target_Vel_y_B = (rms_Vel_y_B - 0.191443)/1.5745421
Target_Vel_beta_c = (rms_Vel_beta_c - 0.333714)/1.4946615

InputFixed_Value = Theta_01/30

model = tf.keras.models.load_model(model_path)

x1 = [0.1 , 0.1]
layerNumber = 7
x_in = tf.constant(x1, dtype=tf.float32, shape=[1, 2])
ini = tf.constant_initializer(value=0.3)
w1 = tf.Variable(ini(shape=[2, layerNumber], dtype=tf.float32))
b1 = tf.Variable(tf.zeros(shape=[1, layerNumber], dtype=tf.float32))
w2 = tf.Variable(ini(shape=[layerNumber, layerNumber], dtype=tf.float32))
b2 = tf.Variable(tf.zeros(shape=[1, layerNumber], dtype=tf.float32))
w3 = tf.Variable(ini(shape=[layerNumber, layerNumber], dtype=tf.float32))
b3 = tf.Variable(tf.zeros(shape=[1, layerNumber], dtype=tf.float32))
w4 = tf.Variable(ini(shape=[layerNumber, layerNumber], dtype=tf.float32))
b4 = tf.Variable(tf.zeros(shape=[1, layerNumber], dtype=tf.float32))
w5 = tf.Variable(ini(shape=[layerNumber,2], dtype=tf.float32))
b5 = tf.Variable(tf.zeros(shape=[1, 2], dtype=tf.float32))


#learning_rate = 0.00001
epochs1 = range(10000)
list_Loss = []
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001)
for epoch in enumerate(epochs1):
    with tf.GradientTape() as t:
        t.watch(x_in)
        h1 = tf.nn.gelu(tf.matmul(x_in, w1) + b1)
        h2 = tf.nn.gelu(tf.matmul(h1, w2) + b2)
        h3 = tf.nn.gelu(tf.matmul(h2, w3) + b3)
        h4 = tf.nn.gelu(tf.matmul(h3, w4) + b4)
        h5 = tf.nn.gelu(tf.matmul(h4, w5) + b5)

        x_Test = x_in + h5
        Fixed_Value = tf.constant([[InputFixed_Value]])
        x_New = tf.concat([x_Test, Fixed_Value], axis=1)

        test = model(x_New)
        loss = (test[0, 0]-Target_Vel_x_c)**2 + (test[0, 1] - Target_Vel_x_B)**2 + (test[0, 2] - Target_Vel_y_B)**2 + (test[0, 3] - Target_Vel_beta_c)**2
        loss_func = tf.reduce_sum(tf.sqrt(loss)/4)
        print(loss_func.numpy())
        list_Loss.append(loss.numpy())
        grad = t.gradient(loss_func,  [w1,b1, w2, b2, w3, b3, w4, b4, w5, b5])
        optimizer.apply_gradients(zip(grad, [w1,b1, w2, b2, w3, b3, w4, b4, w5, b5]))


Predict_Vel_x_C = test[0,0]*1.621362 + 0.211779460
Predict_Vel_x_B = test[0,1]*1.55288368 + 0.21779784
Predict_Vel_y_B = test[0,2]*1.5745421 + 0.191443
Predict_Vel_beta = test[0,3]*1.4946615 + 0.333714
Sol_l_c = x_Test[0,0]*0.2 + 0.3
Sol_w_c = x_Test[0,1]*4 + 1


Predict_Vel_x_C = float(Predict_Vel_x_C)
Predict_Vel_x_B = float(Predict_Vel_x_B)
Predict_Vel_y_B = float(Predict_Vel_y_B)
Predict_Vel_beta = float(Predict_Vel_beta)
Sol_l_c = float(Sol_l_c)
Sol_w_c = float(Sol_w_c)

Output = []
Output.append([' ','l_c','initial_theta', 'Sol_w_c', 'Vel_x_C', 'Vel_x_B', 'Vel_y_B', 'Vel_beta'])
Output.append(["Real", l_c1, Theta_01,w_01, rms_Vel_x_c, rms_Vel_x_B,  rms_Vel_y_B, rms_Vel_beta_c ])
Output.append(["Predict", Sol_l_c,Theta_01, Sol_w_c, Predict_Vel_x_C, Predict_Vel_x_B, Predict_Vel_y_B, Predict_Vel_beta])


list_Loss = np.array(list_Loss)
list_Loss = list_Loss.reshape(-1, 1)
f3 = open(Loss_Data, 'w', newline='')
wr3 = csv.writer(f3)
wr3.writerows(list_Loss)


f3 = open(Output_Data, 'w', newline='')
wr3 = csv.writer(f3)
wr3.writerows(Output)
