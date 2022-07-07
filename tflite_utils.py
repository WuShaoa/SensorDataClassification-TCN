import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pre_process_data import data_process

saved_model_dir = './my_model3'
model_name = 'my_model3'
data_path = './data/my_data.csv'
# Convert the model.
def generate():
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    open(f"./lite_model/{model_name}.tflite", "wb").write(tflite_model)

def test(x):
    interpreter = tf.lite.Interpreter(model_path=f"./lite_model/{model_name}.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()#
    output_details = interpreter.get_output_details()
    
    # NxHxWxC, H:1, W:2
    batch = input_details[0]['shape'][0]
    step = input_details[0]['shape'][1]
    dim = input_details[0]['shape'][2]
    print(output_details)
    results = []
    
    for entry in x:
        input_data = np.resize(entry, (batch, step, dim)).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
    
        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        result = np.squeeze(output_data)

        #print("RESULT:", result)
        #print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
        
        results.append(1 if result >= 0.5 else 0)
    
    return results
    
    
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


if __name__ == "__main__":
    #generate()
    
    x, y, x_test, y_test = next(data_process(data_path))
    y_pred = event_convert(filter_convert(test(x_test)))
    
    plt.plot(np.arange(np.shape(x_test)[0]), y_pred, color='y')
    plt.scatter(np.arange(np.shape(x_test)[0]),y_test, color='r')
    plt.show()
    print(np.abs(y_pred - y_test).sum()/len(y_test)) # simple evaluate
    