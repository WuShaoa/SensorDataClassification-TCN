from cProfile import label
import itertools
from turtle import color
from sklearn import config_context
from tcn import TCN, tcn_full_summary
from keras.layers import Dense, Softmax, Input, Lambda
from keras.models import Sequential
from keras.preprocessing import sequence
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, Precision
from pre_process_data import data_process
import pandas as pd
from sklearn.metrics import confusion_matrix


'''
TCN(
    nb_filters=64,
    kernel_size=3,
    nb_stacks=1,
    dilations=(1, 2, 4, 8, 16, 32),
    padding='causal',
    use_skip_connections=True,
    dropout_rate=0.0,
    return_sequences=False,
    activation='relu',
    kernel_initializer='he_normal',
    use_batch_norm=False,
    use_layer_norm=False,
    use_weight_norm=False,
    **kwargs
)
nb_filters: Integer. The number of filters to use in the convolutional layers. Would be similar to units for LSTM. Can be a list.
kernel_size: Integer. The size of the kernel to use in each convolutional layer.
dilations: List/Tuple. A dilation list. Example is: [1, 2, 4, 8, 16, 32, 64].
nb_stacks: Integer. The number of stacks of residual blocks to use.
padding: String. The padding to use in the convolutions. 'causal' for a causal network (as in the original implementation) and 'same' for a non-causal network.
use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
activation: The activation used in the residual blocks o = activation(x + F(x)).
kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
use_batch_norm: Whether to use batch normalization in the residual layers or not.
use_layer_norm: Whether to use layer normalization in the residual layers or not.
use_weight_norm: Whether to use weight normalization in the residual layers or not.
kwargs: Any other set of arguments for configuring the parent class Layer. For example "name=str", Name of the model. Use unique names when using multiple TCN.
'''

# if time_steps > tcn_layer.receptive_field, then we should not
# be able to solve this task.
batch_size, time_steps, input_dim = None, None, 18
b, s, i = -1, 1, 18 #batch_size, time_steps, input_dim
n_splits=2
#nb, nb_test = 40, 20
data_path = "./data/my_data.csv"
checkpoint_path = "./"
model_path = "./my_model3"
logdir = "./log"

#tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

def model_def():
    tcn_layer = TCN(
        input_shape=(time_steps, input_dim),
        nb_filters=64,
        kernel_size=20,
        dilations=[1, 2, 4, 8, 16, 32],
        #input_shape=(time_steps, input_dim), 
        activation='LeakyReLU',
        )

    # The receptive field tells you how far the model can see in terms of timesteps.
    print('Receptive field size =', tcn_layer.receptive_field)

    # model definition
    model = Sequential([
        Input(shape=(time_steps, input_dim), name='input'),
        tcn_layer,
        Dense(1, activation='sigmoid', name='output'),
        #Softmax()
    ])

    model.build()
    #model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.metrics.BinaryAccuracy()])# categorical_crossentropy mse
    tcn_full_summary(model, expand_residual_blocks=False)
    return model

def plot_model(model, to_file='model.png'):
    tf.keras.utils.plot_model(
        model,
        to_file=to_file,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir='TB',
        dpi=200,
        layer_range=None,
        show_layer_activations=True,
        expand_nested=True,
    )


def train(model, epochs=10, verbose=2, data_path=data_path, n_splits=n_splits, checkpoint_path=checkpoint_path, enable_cb=False,save_best_only=True,monitor='val_binary_accuracy'):
    data_generator = data_process(data_path, n_splits=n_splits)
    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=verbose,
                                                    save_best_only=save_best_only,
                                                    monitor=monitor)
    results = []
    for _ in range(n_splits):
        # reshape the data for input
        x, y, x_test, y_test = next(data_generator)
        
        #print(x[:nb*b].shape, y[:nb*b].shape, x_test.shape, y_test.shape)
        print(x.shape, y.shape, x_test.shape, y_test.shape)
        #x, y, x_test, y_test = x[:nb*b].reshape(b, s, i), y[:nb*b].reshape(b, -1), x_test[:nb_test*b].reshape(b, s, i), y_test[:nb_test*b].reshape(b, -1)
        x, y, x_test, y_test = x.reshape(b, s, i), y, x_test.reshape(b, s, i), y_test
        print(x.shape, y.shape, x_test.shape, y_test.shape)

        # model.load_weights(checkpoint_path)

        result = model.fit(x, 
                  y,
                  epochs=epochs,
                  #validation_split=0.3
                  #batch_size=batch_size
                  callbacks=[cp_callback] if enable_cb else None,
                  validation_data=(x_test, y_test),
                  verbose=verbose
                  )
        results.append(result)
    return results
        
        
#绘制混淆矩阵


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,#这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
   
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label accuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
    plt.savefig('./confusionmatrix32.png',dpi=350)
    plt.show()
# 显示混淆矩阵
def plot_confuse(model, x_val, y_val,labels, cmap=plt.cm.Blues, title='Confusion Matrix', convert=None, func=None):
    predictions = list(map(lambda p: 0 if p[0]<0.5 else 1,model.predict(x_val.reshape(b, s, i))))
    truelabel = y_val    # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions if convert is None else convert(predictions))
    if func is not None: func(conf_mat,title)
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False,target_names=labels,title=title,cmap=cmap)
#=========================================================================================
#最后调用这个函数即可。 test_x是测试数据，test_y是测试标签（这里用的是One——hot向量）
#labels是一个列表，存储了你的各个类别的名字，最后会显示在横纵轴上。
#比如这里我的labels列表

def cmprint(cm, title):
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    print(f"{title} Accuracy: {Accuracy: .2f} Precision: {Precision: .2f} Recall: {Recall: .2f}")

######################## convert algrithms #######################
LABEL_NROMAL = 0
LABEL_FALL = 1

WINDOW = 10
RIGHT_ADD = 5

CONVERT_RATIO = 0.4
NOT_CONVERT_RATIO = 0.6

def label_ratio(index, label_list, delta_window, left_add=0, right_add=0):
    di = np.abs(np.floor(delta_window / 2))
    label = label_list[index]
    if di + left_add < index < len(label_list) - di - right_add:
        return np.sum(np.array(label_list[int(index-di-left_add):int(index+di+right_add+1)]) == label) / (2 * di + left_add + right_add + 1)
    elif index <= di + left_add:
        return np.sum(np.array(label_list[:int(index+di+right_add)]) == label) / (index + di + right_add)
    else:
        return np.sum(np.array(label_list[int(index-di-left_add):]) == label) / (len(label_list) - index + di + left_add)

def filter_convert(result_label):
    for i in range(len(result_label)):
        if label_ratio(i, result_label, WINDOW) < CONVERT_RATIO:
            result_label[i] = LABEL_NROMAL if result_label[i] == LABEL_FALL else LABEL_FALL
        elif label_ratio(i, result_label, WINDOW) < NOT_CONVERT_RATIO:
            if label_ratio(i, result_label, WINDOW, right_add=RIGHT_ADD) < CONVERT_RATIO:
                result_label[i] = LABEL_NROMAL if result_label[i] == LABEL_FALL else LABEL_FALL
            else:
                pass
    
    return result_label

LABEL_NROMAL = 0
LABEL_FALL = 1

# 1st order event
EVENT_NORMAL2FALL = "EVENT_NORMAL2FALL"
EVENT_FALL2NORMAL = "EVENT_FALL2NORMAL"

# # 2nd order event
# EVENT_NORMAL2FALL2NORMAL = 2
# EVENT_FALL2NORMAL2FALL = 3

XI_FNF = 30 # the time priod to determine whether to convert the event, XI < real_action_time_period XI = 100/<spilt>
XI_NFN = 15 # need to change
XI_NFN_PREV = 3

events = [] # an event: (EVENT_TYPE, INDEX_OF_LABEL), index of label is the cdr index of an event
# events_2nd = []

def event_convert(result_label):
    # initialize the covertion result
    converted_result_label =  result_label
    
    # build up events
    events = build_events(converted_result_label)
    
    # convert FALL to NORMAL, filtering the sigular points
    for i in range(len(events)):
        if i > 0:
            delta_event = events[i][1] - events[i-1][1]
            if(events[i-1][0] == EVENT_NORMAL2FALL): # case N2F2N
                if(delta_event < XI_NFN_PREV):
                    converted_result_label[events[i-1][1]:events[i][1]] = [LABEL_NROMAL] * delta_event
    
    # re-build the events according to the new converted_result_label
    events.clear()
    events = build_events(converted_result_label)
    
    # convert NORMAL to FALL
    for i in range(len(events)):
        if i > 0:
            delta_event = events[i][1] - events[i-1][1]
            if(events[i-1][0] == EVENT_FALL2NORMAL): # case F2N2F
                if(delta_event < XI_FNF):
                    converted_result_label[events[i-1][1]:events[i][1]] = [LABEL_FALL] * delta_event
    
    #print("DEBUG: ", events, converted_result_label)
    
    # re-build the events according to the new converted_result_label
    events.clear()
    events = build_events(converted_result_label)
    
    # convert FALL to NORMAL
    for i in range(len(events)):
        if i > 0:
            delta_event = events[i][1] - events[i-1][1]
            if(events[i-1][0] == EVENT_NORMAL2FALL): # case N2F2N
                if(delta_event < XI_NFN):
                    converted_result_label[events[i-1][1]:events[i][1]] = [LABEL_NROMAL] * delta_event

    
    return result_label

def build_events(result_list):
    # build up events
    events_local = []
    for i in range(len(result_list)):
        if i > 0:
            if(result_list[i] == LABEL_NROMAL and result_list[i-1]==LABEL_FALL):
                events_local.append((EVENT_FALL2NORMAL, i))
            elif(result_list[i] == LABEL_FALL and result_list[i-1]==LABEL_NROMAL):
                events_local.append((EVENT_NORMAL2FALL, i))
    return events_local
###############################################convert


if __name__ == "__main__":
    labels=["Normal","Fall"]
    
    model = model_def()
    
    plot_model(model, to_file='model.png')
    #DONE: serialize model, validate data, and normalize data at first. 
    #TODO:life long learning.
    
    #model.load_weights(checkpoint_path)
    
    # uncomment to train the model
    historys = train(model, 
          epochs=10, 
          verbose=1, 
          enable_cb=False, # not save the checkpoints
          data_path=data_path, checkpoint_path=checkpoint_path)

    acc = []
    loss = []
    for result in historys:
        print(result.history)
        acc.extend(list(result.history['binary_accuracy']))
        loss.extend(list(result.history['val_loss']))
    epochs = range(1, len(acc) + 1)

    plt.title(f'Accuracy: {np.max(acc):.2f} and Loss: {np.min(loss):.2f}')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, loss, 'blue', label='Validation loss')
    plt.legend()
    plt.show()
    print(result.history)
    
    g = data_process(data_path, n_splits=n_splits)
    x, y, x_test, y_test = next(g)
    
    # evaluate the model
    loss, acc = model.evaluate(x_test.reshape(b, s, i), y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    #print("predict_classes: ", model.predict(x_test.reshape(b, s, i))[0][0])

    # TP = cm[1][1] #跌倒预测为跌倒
    # TN = cm[0][0] #ADL预测为ADL
    # FP = cm[0][1] #ADL预测为跌倒
    # FN = cm[1][0] #跌倒预测为ADL

    plot_confuse(model, x_test,y_test,labels,
                 func=cmprint)
    plot_confuse(model, x_test,y_test,labels,
                 convert=event_convert,
                 cmap=plt.cm.Greens,
                 func=cmprint,
                 title="event_convert(agrithm1) confusion matrix")
    plot_confuse(model, x_test,y_test,labels,
                 convert=filter_convert,
                 cmap=plt.cm.Oranges,
                 func=cmprint,
                 title="filter_convert(agrithm2) confusion matrix")
    #save to .pb model files
    #model.save(model_path)
    
    #print(model.predict(x[0, 0, :].reshape(1, 1, -1)))
    #plt.plot(np.arange(np.shape(x)[0]), list(map(lambda p: 0 if p[0] < 0.5 else 1, model.predict(x_test.reshape(b, s, i)))), color='y', label='prediction')
    plt.subplot(311)
    plt.title("Original")
    plt.plot(np.arange(np.shape(x)[0]), list(map(lambda p: 0 if p[0] < 0.5 else 1, model.predict(x_test.reshape(b, s, i)))), 'blue', label='prediction')
    plt.scatter(np.arange(np.shape(x)[0]), y_test, color='r', label='real')
    plt.legend()
    plt.subplot(312)
    plt.title("Algrithm 1")
    plt.plot(np.arange(np.shape(x)[0]), event_convert(list(map(lambda p: 0 if p[0] < 0.5 else 1, model.predict(x_test.reshape(b, s, i))))), 'g--', label='event_converted prediction')
    plt.scatter(np.arange(np.shape(x)[0]), y_test, color='r', label='real')
    plt.legend()
    plt.subplot(313)
    plt.title("Algrithm 2")
    plt.plot(np.arange(np.shape(x)[0]), filter_convert(list(map(lambda p: 0 if p[0] < 0.5 else 1, model.predict(x_test.reshape(b, s, i))))), 'y-.', label='filter_converted prediction')
    plt.scatter(np.arange(np.shape(x)[0]), y_test, color='r', label='real')
    plt.legend()
    plt.show()