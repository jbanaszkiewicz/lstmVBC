import json
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
import tensorflow as tf
import pickle
import glob
import errno


def read_from_file(json_file_name):
    with open(json_file_name, "r") as read_file:
        input_list = json.load(read_file)
        return [input_list]


def read_many_json(path):
    # path = "lewa_razem\*.json"
    files = glob.glob(path)
    many_files = []
    for name in files:
        single_file = read_from_file(name)
        many_files.append(single_file)
    return many_files


def position_coordinate_system(dict_joints):
    for joint in dict_joints:
        if joint != 'SpineMid':
            dict_joints[joint][0] = dict_joints[joint][0] - dict_joints['SpineMid'][0]
            dict_joints[joint][1] = dict_joints[joint][1] - dict_joints['SpineMid'][1]
            dict_joints[joint][2] = dict_joints[joint][2] - dict_joints['SpineMid'][2]
    for joint in dict_joints:
        if joint == 'SpineMid':
            dict_joints['SpineMid'][0] = 0
            dict_joints['SpineMid'][1] = 0
            dict_joints['SpineMid'][2] = 0
    return dict_joints


def get_joints(json_file_name, previous_list_input, previous_list_output):
    data = read_many_json(json_file_name)
    for file_number in range(len(data)):
        # unzipping the data
        data1 = data[file_number]
        data2 = data1[0]
        data3=data2["Frames"]
        # initiliasing lists of inputs and outputs
        input_data = previous_list_input
        output_data = previous_list_output

        # wczytanie listy jointow
        with open("ListOfJoints.txt", "rb") as fp:
            list_of_joints = pickle.load(fp)

        for t in data3:
            for dictionaries in t:
                dict_with_coordinates = {}
                if dictionaries == "Tag":
                    output_data.append(t[dictionaries])
                body = t["Body"]
                if dictionaries == "Body":
                    help_table = []
                    dict_with_coordinates = {}
                    for particular_joint_nr in range(len(list_of_joints)):
                        help_table2 = []
                        particular_joint = body[list_of_joints[particular_joint_nr]]
                        # print(type(list_of_joints[particular_joint_nr]))
                        help_table2.append(particular_joint["X"])
                        help_table2.append(particular_joint["Y"])
                        help_table2.append(particular_joint["Z"])
                        dict_with_coordinates[list_of_joints[particular_joint_nr]] = help_table2
                        #print()
                        # print(type(dict_with_coordinates))

                    dict_with_coordinates = position_coordinate_system(dict_with_coordinates)


                    for particular_joint_nr in range(len(list_of_joints)):
                        help_table2 = []
                        particular_joint = dict_with_coordinates[list_of_joints[particular_joint_nr]]
                        help_table2.append(particular_joint[0])
                        help_table2.append(particular_joint[1])
                        help_table2.append(particular_joint[2])
                        help_table.append(help_table2)
                    input_data.append(help_table)
    return input_data, output_data


input_data = []
output_data = []
input_data, output_data=get_joints("lewa_razem\*.json", input_data, output_data)
print(np.shape(input_data), ", ", np.shape(output_data))

reshaped_input_data = np.reshape(input_data, (len(input_data), len(input_data[1][2]), len(input_data[1])))

print(np.shape(reshaped_input_data))


model = Sequential()
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


