import json
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
import tensorflow as tf
import pickle


def read_from_file(json_file_name):
    with open(json_file_name, "r") as read_file:
        input_list = json.load(read_file)
        return [input_list]


def get_joints(json_file_name, previous_list_input, previous_list_output):
    data = read_from_file(json_file_name)
    # unzipping the data
    data2 = data[0]
    data3=data2["Frames"]
    # initiliasing lists of inputs and outputs
    input_data = previous_list_input
    output_data = previous_list_output

    # wczytanie listy jointow
    with open("ListOfJoints.txt", "rb") as fp:
        list_of_joints = pickle.load(fp)

    for t in data3:
        for dictionaries in t:
            if dictionaries == "Tag":
                output_data.append(t[dictionaries])
            body = t["Body"]
            if dictionaries == "Body":
                help_table = []
                for particular_joint_name in range(len(list_of_joints)):
                    help_table2 = []
                    particular_joint = body[list_of_joints[particular_joint_name]]
                    help_table2.append(particular_joint["X"])
                    help_table2.append(particular_joint["Y"])
                    help_table2.append(particular_joint["Z"])
                    help_table.append(help_table2)

                input_data.append(help_table)
    print()

    return input_data, output_data


input_data = []
output_data = []
input_data, output_data=get_joints("testexp.json", input_data, output_data)

# input_data, output_data=get_joints("testexp2.json", input_data, output_data)


print(np.shape(input_data), np.shape(output_data))

reshaped_input_data = np.reshape(input_data, (len(input_data), len(input_data[1][2]), len(input_data[1])))

print(np.shape(reshaped_input_data))


model= Sequential()
model.add(LSTM(25, input_shape=(3, 25), return_sequences=True))
model.add(LSTM(25, return_sequences=True))
model.add(LSTM(25))
model.add(Dense(1))
model.add(Activation('sigmoid')) #there may be a 'softmax' if needed
optimizer = Adam()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

print(model.summary())

model.fit(reshaped_input_data, output_data, batch_size=20, epochs=10)

loss, accuracy = model.evaluate(reshaped_input_data, output_data) #there should be used


