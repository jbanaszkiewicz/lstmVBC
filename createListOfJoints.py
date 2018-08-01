import pickle
import json

def read_from_file(json_file_name):
    with open(json_file_name, "r") as read_file:
        input_list = json.load(read_file)
        return [input_list]


def createListOfJoints():
    list = read_from_file("testexp.json")
    data2 = list[0]
    data3 = data2["Frames"]
    # initiliasing lists of inputs and outputs
    joints_names_list = []
    for t in data3:
        for dictionaries in t:
            body = t["Body"]
            if dictionaries == "Body":
                help_table = []
                for joint in t["Body"]:
                    help_table2 = []
                    if joint == "SpineBase" or joint == "SpineMid" or joint == "Neck" or joint == "Head" or joint == "ShoulderLeft" or joint == "ElbowLeft" or joint == "WristLeft" or joint == "HandLeft" or joint == "ShoulderRight" or joint == "ElbowRight" or joint == "WristRight" or joint == "HandRight" or joint == "HipLeft" or joint == "KneeLeft" or joint == "AnkleLeft" or joint == "FootLeft" or joint == "HipRight" or joint == "KneeRight" or joint == "AnkleRight" or joint == "FootRight" or joint == "SpineShoulder" or joint == "HandTipLeft" or joint == "ThumbLeft" or joint == "HandTipRight" or joint == "ThumbRight":
                        if (len(joints_names_list) < 25):
                            joints_names_list.append(joint)
    return joints_names_list


# with open("ListOfJoints.txt", "wb") as fp:
#   pickle.dump(createListOfJoints(), fp)

b = []
with open("ListOfJoints.txt", "rb") as fp:
    b = pickle.load(fp)
print(b)
print(type(b))

