import random
import numpy as np
from reedsolo import RSCodec

BITS_IN_ONE_BYTE = 8


def create_msg(msg_len):
    return np.random.choice([0, 1], msg_len)


def bits_arr_to_chr(bits_arr):
    result = 0
    for i in range(0, BITS_IN_ONE_BYTE, 1):
        result += bits_arr[i] * (2 ** i)
    return chr(result)


def charbytes_to_int(char):
    return ord(char)


def reverse_elem_value(value):
    if value == 0:
        return 1
    if value == 1:
        return 0
    return value - 5


def simulate_errors(np_array, errors_num):
    new_arr = np.copy(np_array)
    positions = random.sample(range(0, np_array.shape[0], 1), errors_num)
    print("Positions of errors", positions)
    for pos in positions:
        new_arr[pos] = reverse_elem_value(np_array[pos])
    return new_arr


def int_to_array_of_bits(int_value):
    cur_value = int_value
    bits_list = []
    res_length = BITS_IN_ONE_BYTE
    while cur_value > 0:
        # bits_list.insert(0, cur_value % 2)
        bits_list.append(cur_value % 2)
        cur_value = cur_value // 2
    while res_length > len(bits_list):
        bits_list.append(0)
    return np.array(bits_list)


def view_msg_as_bytes_arr(msg):
    bit_msg_len = msg.shape[0]
    # bytes = bit_msg_len // 8 if bit_msg_len % 8 == 0 else bit_msg_len // 8 + 1
    bytes_list = []
    for i in range(0, bit_msg_len, BITS_IN_ONE_BYTE):
        bytes_list.append(bits_arr_to_chr(msg[0 + i:BITS_IN_ONE_BYTE + i]))
    return np.array(bytes_list)


def view_bytes_arr_as_msg(charbytes_arr):
    result = np.empty(0)
    for i in range(0, charbytes_arr.shape[0], 1):
        result = np.concatenate((result, int_to_array_of_bits(charbytes_to_int(charbytes_arr[i]))), axis=0)
    return result


def view_bytearr_as_msg(bytesarr):
    result = list(bytesarr)
    return np.array(result)


def encode_batch_to_np_arrays(src_msg_len, batch_size, desired_output):
    message_arrays = np.random.choice([0, 1], (batch_size, src_msg_len))
    assert desired_output >= src_msg_len
    errors_to_repair = (desired_output - src_msg_len) // 2
    if desired_output != src_msg_len + 2 * errors_to_repair:
        raise ValueError('Desired output cannot be matched. Check arguments')
    codec_param = 2 * errors_to_repair
    rsc = RSCodec(codec_param)
    result = np.frombuffer(rsc.encode(message_arrays[0].astype(np.uint8)), dtype=np.uint8)
    result = np.reshape(result, (1, result.shape[0]))
    for i in range(1, batch_size, 1):
        tmp = np.frombuffer(rsc.encode(message_arrays[i].astype(np.uint8)), dtype=np.uint8)
        tmp = np.reshape(tmp, (1, tmp.shape[0]))
        result = np.concatenate((result, tmp), axis=0)
    return result
