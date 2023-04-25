import datetime
import numpy as np

PTH_TO_SRC_FILE = "D:/Рабочий стол/stats2023-04-10_identity_13-45-17.txt"

def parse_line(line):
    parts = line.split("\t")
    parts[2] = parts[2].replace("[", "")
    parts[2] = parts[2].replace("]", "")

    parts[3] = parts[3].replace("[", "")
    parts[3] = parts[3].replace("]", "")
    parts[3] = parts[3].replace("\n", "")

    src_msg = [int(n) for n in parts[2].split(" ")]
    rec_msg = [int(n) for n in parts[3].split(" ")]

    return np.array(src_msg), np.array(rec_msg)


def get_errors_dict(src_msg, dec_msg):
    errors_dictionary = {1: 0, 2: 0, 3: 0, 4: 0}
    sequential = 0
    for i in range(0, np.shape(src_msg)[0], 1):
        if src_msg[i] == dec_msg[i]:
            if sequential != 0:
                try:
                    errors_dictionary[sequential] = errors_dictionary[sequential] + 1
                except KeyError:
                    errors_dictionary[sequential] = 1
            sequential = 0
        else:
            sequential += 1

    if sequential != 0:
        try:
            errors_dictionary[sequential] = errors_dictionary[sequential] + 1
        except KeyError:
            errors_dictionary[sequential] = 1

    return errors_dictionary


datetime_date_str = str(datetime.datetime.now().date())
datetime_time_str = str(datetime.datetime.now().time()).replace(" ", "")
datetime_time_str = datetime_time_str.split(".")[0]
datetime_time_str = datetime_time_str.replace(":", "-")
PTH_TO_SAVE_FILE = r"D:\Рабочий стол\analyze" + datetime_date_str + "_" + datetime_time_str + ".txt"

f = open(PTH_TO_SRC_FILE, 'r')
# skip title line
f.readline()
diction_global = {}
msg_count = 0
while True:
    line = f.readline()
    if not line:
        break
    if line == "DATA_BLOCK_FINISHED\n":
        break
    # getting src_msg and enc_msg as arrays
    src_msg_arr, dec_msg_arr = parse_line(line)
    dict_temp = get_errors_dict(src_msg_arr, dec_msg_arr)
    for key in dict_temp.keys():
        try:
            diction_global[key] = diction_global[key] + dict_temp[key]
        except KeyError:
            diction_global[key] = dict_temp[key]
    msg_count += 1

f = open(PTH_TO_SAVE_FILE, 'a')
for key in diction_global.keys():
    f.write(str(key) +": " + str(diction_global[key]))
    f.write('\n')
f.write("MSG count: " + str(msg_count) + "\n")
f.close()