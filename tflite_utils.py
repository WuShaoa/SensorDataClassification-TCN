import time
import numpy as np
import tensorflow as tf
from pre_process_data import data_process

saved_model_dir = './my_model2'
model_name = 'my_model2'
data_path = './data/my_data.csv'
# Convert the model.
def generate():
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    open(f"./lite_model/{model_name}.tflite", "wb").write(tflite_model)

def test(x):
    interpreter = tf.lite.Interpreter(model_path=f"./lite_model/{model_name}.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # NxHxWxC, H:1, W:2
    batch = input_details[0]['shape'][0]
    step = input_details[0]['shape'][1]
    dim = input_details[0]['shape'][2]
    
    results = []
    
    for entry in x:
        input_data = np.resize(entry, (batch, step, dim)).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
    
        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        result = np.squeeze(output_data)

        print("RESULT:", result)
        print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
        
        results.append(result)
    
    return results
    
if __name__ == "__main__":
    #generate()
    
    x, y, x_test, y_test = next(data_process(data_path))
    y_pred = test(x)
    print(np.abs(y_pred - y).sum()/len(y)) # simple evaluate
    