import math
import os
import torchvision.transforms.functional as TF
import numpy as np
import torch.nn
from PIL import Image
import datetime

import utils
from helpers.solomon_reed_functional import encode_batch_to_np_arrays, decoded_msg_to_src_msg
from helpers.ssim_metrics import calculate_ssim
from model.hidden_copy import Hidden
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.dropout import Dropout
from noise_layers.noiser import Noiser
from test_model import centerCrop
ARCH_NUMBER = '1'
MSG_LEN = '64'
SRC_MSG_LENGTH_SR = 44
IMAGE_QUANTITY = '_1000_'
# PTH_IMAGES_DIRECTORY = r'D:\Рабочий стол\10000test'
PTH_IMAGES_DIRECTORY = r'D:\Рабочий стол\1000test'
ADD_TO_TITLE = "msg_" + MSG_LEN + "_a" + ARCH_NUMBER + IMAGE_QUANTITY
PTH_OPTIONS_FILE = r"D:\Рабочий стол\exp1_data\arch" + ARCH_NUMBER + "\options-and-config_msg" + MSG_LEN + "_arch" + ARCH_NUMBER + ".pickle"
PTH_CHCKPNT_FILE = r"D:\Рабочий стол\exp1_data\arch" + ARCH_NUMBER + "\identity_msg" + MSG_LEN + "--epoch-300.pyt"
NOISE_MODE = "identity"  # identity crop cropout dropout and jpeg are available

# PTH_OPTIONS_FILE = r"D:\Рабочий стол\runs_from_virt\tries_to improve\try_success\options-and-config.pickle"
# PTH_CHCKPNT_FILE = r"D:\Рабочий стол\runs_from_virt\tries_to improve\try_success\for_conf--epoch-398.pyt"
BATCH_SIZE_ONE_IMAGE = 1
# crop
CROP_HEIGHT_RATIO_RANGE_MIN = 0.2
CROP_HEIGHT_RATIO_RANGE_MAX = 0.3
CROP_WIDTH_RATIO_RANGE_MIN = 0.4
CROP_WIDTH_RATIO_RANGE_MAX = 0.5

# cropout
CROPOUT_HEIGHT_RATIO_RANGE_MIN = 0.11
CROPOUT_HEIGHT_RATIO_RANGE_MAX = 0.22
CROPOUT_WIDTH_RATIO_RANGE_MIN = 0.33
CROPOUT_WIDTH_RATIO_RANGE_MAX = 0.44

# dropout
DROPOUT_KEEP_RATIO_RANGE_MIN = 0.55
DROPOUT_KEEP_RATIO_RANGE_MAX = 0.6

NOISE_PARAM_1 = 0
NOISE_PARAM_2 = 0
NOISE_PARAM_3 = 0
NOISE_PARAM_4 = 0

def calculate_psnr_img(src_tensor, result_tensor):
    # img1 and img2 have range [0, 255]
    src_img = utils.tensor_to_image(src_tensor)
    src_img = np.squeeze(src_img, 0)
    res_img = utils.tensor_to_image(result_tensor)
    res_img = np.squeeze(res_img, 0)

    img1 = src_img.astype(np.float64)
    img2 = res_img.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# work with tensors
def calculate_psnr(src_tensor, result_tensor):
    # Calculate PSNR between the two images
    mse = torch.nn.functional.mse_loss(src_tensor, result_tensor)
    # psnr = 10 * torch.log10(1 / mse)
    return 10 * torch.log10(1 / mse)


def calc_ssim(src_tensor, result_tensor):
    src_img = utils.tensor_to_image(src_tensor)
    src_img = np.squeeze(src_img, 0)
    res_img = utils.tensor_to_image(result_tensor)
    res_img = np.squeeze(res_img, 0)
    return calculate_ssim(src_img, res_img)

def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def convert_msg_2_str(msg):
    msg = msg.astype(int)
    msg = np.squeeze(msg, 0)
    msg_str = str(msg)
    return msg_str.replace('\n ', ' ')


def main():
    global NOISE_PARAM_1, NOISE_PARAM_2, NOISE_PARAM_3, NOISE_PARAM_4
    datetime_date_str = str(datetime.datetime.now().date())
    datetime_time_str = str(datetime.datetime.now().time()).replace(" ", "")
    datetime_time_str = datetime_time_str.split(".")[0]
    datetime_time_str = datetime_time_str.replace(":", "-")

    PTH_BEP_STATS_FILE = r"D:\Рабочий стол\bep_stats" + ADD_TO_TITLE + datetime_date_str + "_" + NOISE_MODE + "_" + datetime_time_str + ".txt"
    PTH_LOSS_STATS_FILE = r"D:\Рабочий стол\loss_stats" + ADD_TO_TITLE + datetime_date_str + "_" + NOISE_MODE + "_" + datetime_time_str + ".txt"
    PTH_PSNR_STATS_FILE = r"D:\Рабочий стол\psnr_stats" + ADD_TO_TITLE + datetime_date_str + "_" + NOISE_MODE + "_" + datetime_time_str + ".txt"
    PTH_SSIM_STATS_FILE = r"D:\Рабочий стол\ssim_stats" + ADD_TO_TITLE + datetime_date_str + "_" + NOISE_MODE + "_" + datetime_time_str + ".txt"

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_options, hidden_config, noise_config = utils.load_options(PTH_OPTIONS_FILE)

    if NOISE_MODE == "crop":
        crop = Crop((CROP_HEIGHT_RATIO_RANGE_MIN, CROP_HEIGHT_RATIO_RANGE_MAX),
                    (CROP_WIDTH_RATIO_RANGE_MIN, CROP_WIDTH_RATIO_RANGE_MAX))
        noiser = Noiser([crop, ], device)
        NOISE_PARAM_1 = CROP_HEIGHT_RATIO_RANGE_MIN
        NOISE_PARAM_2 = CROP_HEIGHT_RATIO_RANGE_MAX
        NOISE_PARAM_3 = CROP_WIDTH_RATIO_RANGE_MIN
        NOISE_PARAM_4 = CROP_WIDTH_RATIO_RANGE_MAX
    if NOISE_MODE == "cropout":
        cropout = Cropout((CROPOUT_HEIGHT_RATIO_RANGE_MIN, CROPOUT_HEIGHT_RATIO_RANGE_MAX),
                          (CROPOUT_WIDTH_RATIO_RANGE_MIN, CROPOUT_WIDTH_RATIO_RANGE_MAX))
        noiser = Noiser([cropout, ], device)
        NOISE_PARAM_1 = CROPOUT_HEIGHT_RATIO_RANGE_MIN
        NOISE_PARAM_2 = CROPOUT_HEIGHT_RATIO_RANGE_MAX
        NOISE_PARAM_3 = CROPOUT_WIDTH_RATIO_RANGE_MIN
        NOISE_PARAM_4 = CROPOUT_WIDTH_RATIO_RANGE_MAX
    if NOISE_MODE == "dropout":
        dropout = Dropout((DROPOUT_KEEP_RATIO_RANGE_MIN, DROPOUT_KEEP_RATIO_RANGE_MAX))
        noiser = Noiser([dropout, ], device)
        NOISE_PARAM_1 = DROPOUT_KEEP_RATIO_RANGE_MIN
        NOISE_PARAM_2 = DROPOUT_KEEP_RATIO_RANGE_MAX
        NOISE_PARAM_3 = "-"
        NOISE_PARAM_4 = "-"
    if NOISE_MODE == "jpeg":
        noiser = Noiser(["JpegPlaceholder", ], device)
        NOISE_PARAM_1 = "-"
        NOISE_PARAM_2 = "-"
        NOISE_PARAM_3 = "-"
        NOISE_PARAM_4 = "-"

    # убираем слой identity без шума для тестирования лишь на одном шумовом слое
    if NOISE_MODE == "identity":
        noiser = Noiser([], device)
        NOISE_PARAM_1 = "-"
        NOISE_PARAM_2 = "-"
        NOISE_PARAM_3 = "-"
        NOISE_PARAM_4 = "-"
    else:
        noiser.remove_noise_layer(0)

    # noiser = Noiser(noise_config, device)
    checkpoint = torch.load(PTH_CHCKPNT_FILE)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)
    f = open(PTH_BEP_STATS_FILE, 'a')
    f.write("file_name" + '\t' + "error_value" + '\t\t\t\t' + "src_message" + '\t\t\t\t\t\t\t\t' + "recovered_message" + '\n')
    f.close()
    files = os.listdir(PTH_IMAGES_DIRECTORY)
    i = 0
    for file in files:
        src_img_pth = PTH_IMAGES_DIRECTORY + '\\' + file
        image_pil = Image.open(src_img_pth)
        try:
            image = randomCrop(np.array(image_pil), hidden_config.H, hidden_config.W)
        except AssertionError:
            # print(src_img_pth)
            continue
        except ValueError:
            continue
        try:
            image_tensor = TF.to_tensor(image).to(device)
        except ValueError:
            # print(src_img_pth)
            continue
        image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
        image_tensor.unsqueeze_(0)

        # for t in range(args.times):
        message = torch.Tensor(encode_batch_to_np_arrays(SRC_MSG_LENGTH_SR,
                                                         BATCH_SIZE_ONE_IMAGE,
                                                         hidden_config.message_length)).to(device)
        losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch(
            [image_tensor, message])
        decoded_rounded = decoded_messages.detach().cpu().numpy().round()
        message_detached = message.detach().cpu().numpy()
        recover_msg = decoded_msg_to_src_msg(decoded_rounded, SRC_MSG_LENGTH_SR, hidden_config.message_length)
        error_value = np.mean(np.abs(recover_msg[0:SRC_MSG_LENGTH_SR] - message_detached[0:SRC_MSG_LENGTH_SR]))
        f = open(PTH_BEP_STATS_FILE, 'a')
        f.write(file.replace('.jpg', '') + '\t')
        f.write('{:.4f}'.format(error_value) + '\t')
        f.write(convert_msg_2_str(message_detached))
        f.write('\t')
        f.write(convert_msg_2_str(decoded_rounded))
        f.write('\n')
        f.close()

        f = open(PTH_LOSS_STATS_FILE, 'a')
        f.write(file.replace('.jpg', '') + '\t')
        f.write('{:.4f}'.format(losses['loss           ']) + '\n')
        f.close()

        f = open(PTH_PSNR_STATS_FILE, 'a')
        f.write(file.replace('.jpg', '') + '\t')
        psnr_value = calculate_psnr_img(image_tensor, encoded_images)
        f.write('{:.4f}'.format(psnr_value) + '\n')
        f.close()

        f = open(PTH_SSIM_STATS_FILE, 'a')
        f.write(file.replace('.jpg', '') + '\t')
        ssim_value = calc_ssim(image_tensor, encoded_images)
        f.write('{:.4f}'.format(ssim_value) + '\n')
        f.close()

        print(file + str(i))
        i += 1


    f = open(PTH_BEP_STATS_FILE, 'r')
    # skip title line
    f.readline()
    ber_sum = 0
    msg_counter = 0
    while True:
        line = f.readline()
        if not line:
            break
        # extract error value from line
        parts = line.split("\t")
        ber_sum += float(parts[1])
        msg_counter += 1
    ber_result = float(ber_sum / msg_counter)
    print("BER result: ", ber_result)


    f = open(PTH_LOSS_STATS_FILE, 'r')
    # skip title line
    f.readline()
    loss_sum = 0
    msg_counter = 0
    while True:
        line = f.readline()
        if not line:
            break
        # extract error value from line
        parts = line.split("\t")
        loss_sum += float(parts[1])
        msg_counter += 1
    loss_result = float(loss_sum / msg_counter)
    print("LOSS result: ", loss_result)


    f = open(PTH_PSNR_STATS_FILE, 'r')
    # skip title line
    f.readline()
    psnr_sum = 0
    msg_counter = 0
    while True:
        line = f.readline()
        if not line:
            break
        # extract error value from line
        parts = line.split("\t")
        psnr_sum += float(parts[1])
        msg_counter += 1
    psnr_result = float(psnr_sum / msg_counter)
    print("PSNR result: ", psnr_result)


    f = open(PTH_SSIM_STATS_FILE, 'r')
    # skip title line
    f.readline()
    ssim_sum = 0
    msg_counter = 0
    while True:
        line = f.readline()
        if not line:
            break
        # extract error value from line
        parts = line.split("\t")
        ssim_sum += float(parts[1])
        msg_counter += 1
    ssim_result = float(ssim_sum / msg_counter)
    print("SSIM result: ", ssim_result)


    f = open(PTH_BEP_STATS_FILE, 'a')
    f.write("DATA_BLOCK_FINISHED" + '\n')
    f.write('NOISE TYPE ' + NOISE_MODE + ' ' + str(NOISE_PARAM_1) + ' ' +
            str(NOISE_PARAM_2) + ' ' + str(NOISE_PARAM_3) + ' ' + str(NOISE_PARAM_4) + '\n')
    f.write('Images evaluated\n')
    f.write('{:.3f}'.format(msg_counter) + '\n')
    f.write('BER MATH EXPECTATION\n')
    f.write('{:.8f}'.format(ber_result))
    f.close()


    f = open(PTH_LOSS_STATS_FILE, 'a')
    f.write("DATA_BLOCK_FINISHED" + '\n')
    f.write('NOISE TYPE ' + NOISE_MODE + ' ' + str(NOISE_PARAM_1) + ' ' +
            str(NOISE_PARAM_2) + ' ' + str(NOISE_PARAM_3) + ' ' + str(NOISE_PARAM_4) + '\n')
    f.write('Images evaluated\n')
    f.write('{:.3f}'.format(msg_counter) + '\n')
    f.write('LOSS MATH EXPECTATION\n')
    f.write('{:.8f}'.format(loss_result))
    f.close()


    f = open(PTH_PSNR_STATS_FILE, 'a')
    f.write("DATA_BLOCK_FINISHED" + '\n')
    f.write('NOISE TYPE ' + NOISE_MODE + ' ' + str(NOISE_PARAM_1) + ' ' +
            str(NOISE_PARAM_2) + ' ' + str(NOISE_PARAM_3) + ' ' + str(NOISE_PARAM_4) + '\n')
    f.write('Images evaluated\n')
    f.write('{:.3f}'.format(msg_counter) + '\n')
    f.write('PSNR MATH EXPECTATION\n')
    f.write('{:.8f}'.format(psnr_result))
    f.close()


    f = open(PTH_SSIM_STATS_FILE, 'a')
    f.write("DATA_BLOCK_FINISHED" + '\n')
    f.write('NOISE TYPE ' + NOISE_MODE + ' ' + str(NOISE_PARAM_1) + ' ' +
            str(NOISE_PARAM_2) + ' ' + str(NOISE_PARAM_3) + ' ' + str(NOISE_PARAM_4) + '\n')
    f.write('Images evaluated\n')
    f.write('{:.3f}'.format(msg_counter) + '\n')
    f.write('SSIM MATH EXPECTATION\n')
    f.write('{:.8f}'.format(ssim_result))
    f.close()


if __name__ == '__main__':
    main()
