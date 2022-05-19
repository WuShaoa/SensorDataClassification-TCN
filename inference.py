import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pre_process_data import data_process

data_path = "./data/my_data.csv"
model_path = "./my_model"
b, s, i = -1, 1, 18 #batch_size, time_steps, input_dim
input_name='input'
output_name= 'output'

loaded = tf.saved_model.load(model_path)
infer = loaded.signatures["serving_default"]
# print(list(loaded.signatures.keys())) 
print(infer.structured_outputs)

x, y, x_test, y_test = next(data_process(data_path))
pred_out = infer(input=x_test.reshape(b, s, i).astype(np.float32))
print(pred_out[output_name])

plt.plot(np.arange(len(x_test)), pred_out[output_name].numpy(), color='y')
plt.scatter(np.arange(len(x_test)), y_test, color='r')
plt.show()