import numpy as np
import os
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import csv

example_idx_=1
dirSep = "/" if platform.system() == "Darwin" else "\\"
curr_path = f"{os.getcwd()}{dirSep}Numerical_Problem{example_idx_}{dirSep}"

model_path = f"{curr_path}DNN_Model_Ex1.h5"
loss_Path = f"{curr_path}loss.csv"
opt_res = f"{curr_path}opt_res.csv"

Loss_Data = f"{curr_path}Loss_Opt.csv"
Output_Data = f"{curr_path}Output_Opt.csv" 
model = tf.keras.models.load_model(model_path)

x1 = [0.1, 0.1]
layerNumber = 12
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
w5 = tf.Variable(ini(shape=[layerNumber, 2], dtype=tf.float32))

epochs1 = range(5000)
list_Loss = []
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0005, ema_momentum=0.8)
for epoch in enumerate(epochs1):
    with tf.GradientTape() as t:
        t.watch(x_in)
        h1 = tf.nn.gelu(tf.matmul(x_in, w1) + b1)
        h2 = tf.nn.gelu(tf.matmul(h1, w2) + b2)
        h3 = tf.nn.gelu(tf.matmul(h2, w3) + b3)
        h4 = tf.nn.gelu(tf.matmul(h3, w4) + b4)
        h5 = tf.nn.gelu(tf.matmul(h4, w5))

        x_Test = x_in + h5

        test = model(x_Test)
        loss = (test[0, 0]-0.9079)**2 + (test[0, 1] - 0.6693)**2 + (test[0, 2] - 0.1462496)**2
        loss_func = tf.reduce_mean(loss)
        print(loss_func.numpy())
        list_Loss.append(loss_func.numpy())
        grad = t.gradient(loss_func,  [w1,b1, w2, b2, w3, b3, w4, b4, w5])
        optimizer.apply_gradients(zip(grad, [w1,b1, w2, b2, w3, b3, w4, b4, w5]))

Test_Out = [x_Test, test[0,0], test[0,1], test[0,2]]

Predict_Acceleartion_a = test[0,0]*31.5 - 23.84
Predict_Acceleration_b = test[0,1]*12.1106505345577 - 13.24007
Predict_Tension = test[0,2]*93.5982304665036 + 21.3242
Sol_x_a = x_Test[0,0]*0.3 + 0.1
Sol_rw = x_Test[0,1]*3 +1.06

Predict_Acceleartion_a = float(Predict_Acceleartion_a)
Predict_Acceleration_b = float(Predict_Acceleration_b)
Predict_Tension = float(Predict_Tension)
Sol_x_a = float(Sol_x_a)
Sol_rw = float(Sol_rw)

Output = []
Output.append([' ','X_a', 'Rw', 'Acc_a', 'Acc_b', 'Tension'])
Output.append(["Real", 0.3, 2.4, 4.6, -5.16, 34.4])
Output.append(["Predict", Sol_x_a, Sol_rw, Predict_Acceleartion_a, Predict_Acceleration_b, Predict_Tension])


list_Loss = np.array(list_Loss)
list_Loss = list_Loss.reshape(-1, 1)
f3 = open(Loss_Data, 'w', newline='')
wr3 = csv.writer(f3)
wr3.writerows(list_Loss)


f3 = open(Output_Data, 'w', newline='')
wr3 = csv.writer(f3)
wr3.writerows(Output)
