import numpy as np
import os
import math
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
import csv


example_idx_=1
dirSep = "/" if platform.system() == "Darwin" else "\\"
curr_path = f"{os.getcwd()}{dirSep}Numerical_Problem{example_idx_}{dirSep}"

model_path = f"{curr_path}DNN_Model.h5"
loss_Path = f"{curr_path}loss.csv"
opt_res = f"{curr_path}opt_res.csv"

input_file = f"{curr_path}Input_Training_50.csv"
output_file = f"{curr_path}Output_Training_50.csv"

def SystemMetamodel(n_layer, n_neuron, learning_rate):
    with open(input_file) as file_name:
        x_train = np.loadtxt(file_name, delimiter=",");
    with open(output_file) as out_name:
        y_train = np.loadtxt(out_name, delimiter=",");

    input_shape = 2
    output_shape = 3

    i = n_layer
    j = n_neuron
    k = learning_rate

    X = tf.keras.layers.Input(shape=[input_shape])
    firstLayer = tf.keras.layers.Dense(j, activation='swish')(X)
    if i != 1:
        for layer_num in range(i):
            if layer_num == 0:
                Current_Layer = tf.keras.layers.Dense(j, activation='swish')(firstLayer)
                before_Layer = tf.keras.layers.Dense(j, activation='swish')(firstLayer)
            else:
                Current_Layer = tf.keras.layers.Dense(j, activation='swish')(before_Layer)
                before_Layer = Current_Layer
        y = tf.keras.layers.Dense(output_shape)(Current_Layer)
    else:
        y = tf.keras.layers.Dense(output_shape)(firstLayer)

    model = tf.keras.models.Model(X, y)
    opt = tf.keras.optimizers.Adam(learning_rate=k)
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    cb_early_stopping = EarlyStopping(monitor='loss', patience=300)
    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='loss', verbose=1, save_best_only=True)

    model.fit(x_train, y_train, epochs=2000, verbose=1, callbacks=[cb_checkpoint, cb_early_stopping])
    loss, acc = model.evaluate(x_train, y_train)

    with open(loss_Path,'w') as f2:
        writer = csv.writer(f2)
        writer.writerow([loss, acc])

i = 0
j = 0
list_history = []
list_Neuron = []
list_gradient = []
list_layer = []
list_loss = []
list_Accuracy = []
list_learningrate = []

alpha = 8
beta = 20
gamma = 0.00000001

while 1:
    if i == 0:
        i = i+1
        SystemMetamodel(n_layer=i, n_neuron=i, learning_rate=1e-3)
        with open(loss_Path) as file_name:
            loss = np.loadtxt(file_name, delimiter=",");

        loss_before = loss[0]
        list_loss.append(loss[0])

        Accuracy_before = loss[1]
        list_Accuracy.append(loss[1])

        list_history.append([i, i, loss[0], loss[1]])
        list_layer.append(i)
        list_Neuron.append(i)

    elif i == 1:
        i = i+1
        SystemMetamodel(n_layer=i, n_neuron=i, learning_rate=1e-3)
        with open(loss_Path) as file_name:
            loss = np.loadtxt(file_name, delimiter=",");

        loss_before = loss[0]
        list_loss.append(loss[0])

        Accuracy_before = loss[1]
        list_Accuracy.append(loss[1])

        list_history.append([i, i, loss[0], loss[1]])
        list_layer.append(i)
        list_Neuron.append(i)
        list_learningrate.append(0.001)

        loss_Dif = list_loss[i-1] - list_loss[i-2]

        if Accuracy_before >= 0.85:
            New_layer = 3
            New_Neuron = 4

        elif 0.5 < Accuracy_before < 0.85:
            New_layer = 4
            New_Neuron = 5

        else:
            New_layer = 6
            New_Neuron = 7

        list_layer.append(New_layer)
        list_Neuron.append(New_Neuron)

        Accuracy_before = loss[1]
        list_Accuracy.append(loss[1])


    else:
        i = i+1
        loss_after = loss_before
        SystemMetamodel(n_layer=New_layer, n_neuron=New_Neuron, learning_rate=1e-3)
        with open(loss_Path) as file_name:
            loss = np.loadtxt(file_name, delimiter=",");

        loss_after = loss[0]
        list_loss.append(loss[0])

        list_history.append([New_layer, New_Neuron, loss_after, loss[1]])

        loss_Dif = list_loss[i - 1] - list_loss[i - 2]
        if list_layer[i-1] - list_layer[i-2] != 0:
            grad_layer_before = (list_loss[i - 2] - list_loss[i - 3]) / (list_layer[i - 2] - list_layer[i - 3])
            grad_layer_after = (list_loss[i - 1] - list_loss[i - 2]) / (list_layer[i - 1] - list_layer[i - 2])
            grad_diff_layer = grad_layer_after/grad_layer_before
            d_layer = math.ceil(grad_diff_layer * alpha)
            if d_layer >= 5:
                d_layer = 5
            if d_layer <= -5:
                d_layer = -5
            New_layer = list_layer[i - 1] - d_layer
        else :
            New_layer = list_layer[i - 1] + 1
        if list_Neuron[i-1] - list_Neuron[i-2] != 0:
            grad_Neuron_before = (list_loss[i - 2] - list_loss[i - 3]) / (list_Neuron[i - 2] - list_Neuron[i - 3])
            grad_Neuron_after = (list_loss[i - 1] - list_loss[i - 2]) / (list_Neuron[i - 1] - list_Neuron[i - 2])
            grad_diff_Neuron = grad_Neuron_after/grad_Neuron_before
            d_Neuron = math.ceil(grad_diff_Neuron * beta)
            if d_Neuron >= 10:
                d_Neuron = 10
            if d_Neuron <= -10:
                d_Neuron = -10
            New_Neuron = list_Neuron[i - 1] - d_Neuron
        else :
            New_Neuron = list_Neuron[i - 1] + 1
        list_layer.append(New_layer)
        list_Neuron.append(New_Neuron)
        Accuracy_before = loss[1]
        list_Accuracy.append(loss[1])
    if loss[1] > 0.997:
        break

with open(opt_res,'w') as file_name:
    writer = csv.writer(file_name)
    writer.writerows(list_history)


