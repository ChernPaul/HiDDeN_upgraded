import os
import torchvision.transforms.functional as TF
import numpy as np
import torch.nn
from PIL import Image
import datetime


import utils
from model.hidden import Hidden
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.dropout import Dropout
from noise_layers.noiser import Noiser

PTH_IMAGES_DIRECTORY = r'D:\Рабочий стол\10000test'
# PTH_SRC_IMG_FILE = r"D:\Рабочий стол\val2014\val2014\COCO_val2014_000000000073.jpg"
PTH_OPTIONS_FILE = r"C:\Users\Pavel\PycharmProjects\HiDDeN_upd\runs\run_with_noises_crop((0.2,0.3),(0.4,0.5))+cropout((0.11,0.22),(0.33,0.44))+dropout(0.55,0.6)+jpeg() 2023.03.03--14-10-28\options-and-config.pickle"
PTH_CHCKPNT_FILE = r"C:\Users\Pavel\PycharmProjects\HiDDeN_upd\runs\run_with_noises_crop((0.2,0.3),(0.4,0.5))+cropout((0.11,0.22),(0.33,0.44))+dropout(0.55,0.6)+jpeg() 2023.03.03--14-10-28\checkpoints\run_with_noises_crop((0.2,0.3),(0.4,0.5))+cropout((0.11,0.22),(0.33,0.44))+dropout(0.55,0.6)+jpeg()--epoch-300.pyt"
NOISE_MODE = "jpeg"  # identity crop cropout dropout and jpeg are available

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

def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def convert_msg_2_str(msg):
    msg_str = str(msg)
    return msg_str.replace('\n ', '')


def main():
    global NOISE_PARAM_1, NOISE_PARAM_2, NOISE_PARAM_3, NOISE_PARAM_4
    datetime_date_str = str(datetime.datetime.now().date())
    datetime_time_str = str(datetime.datetime.now().time()).replace(" ", "")
    datetime_time_str = datetime_time_str.split(".")[0]
    datetime_time_str = datetime_time_str.replace(":", "-")
    PTH_STATS_FILE = r"D:\Рабочий стол\stats" + datetime_date_str + "_" + NOISE_MODE + "_" + datetime_time_str + ".txt"

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
    f = open(PTH_STATS_FILE, 'a')
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
            continue
        try:
            image_tensor = TF.to_tensor(image).to(device)
        except ValueError:
            continue
        image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
        image_tensor.unsqueeze_(0)

        # for t in range(args.times):
        message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
                                                         hidden_config.message_length))).to(device)
        losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch(
            [image_tensor, message])
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        message_detached = message.detach().cpu().numpy()
        error_value = np.mean(np.abs(decoded_rounded - message_detached))
        f = open(PTH_STATS_FILE, 'a')
        f.write(file.replace('.jpg', '') + '\t')
        f.write('{:.3f}'.format(error_value) + '\t')
        f.write(convert_msg_2_str(message_detached))
        f.write('\t')
        f.write(convert_msg_2_str(decoded_rounded))
        f.write('\n')
        f.close()
        print(file + str(i))
        i += 1


    f = open(PTH_STATS_FILE, 'r')
    # skip title line
    f.readline()
    sum = 0
    counter = 0
    while True:
        line = f.readline()
        if not line:
            break
        # extract error value from line
        parts = line.split("\t")
        sum += float(parts[1])
        counter += 1

    result = float(sum / counter)
    print(result)

    f = open(PTH_STATS_FILE, 'a')
    f.write('NOISE TYPE ' + NOISE_MODE + ' ' + str(NOISE_PARAM_1) + ' ' +
            str(NOISE_PARAM_2) + ' ' + str(NOISE_PARAM_3) + ' ' + str(NOISE_PARAM_4) + '\n')
    f.write('MATH EXPECTATION\n')
    f.write('{:.8f}'.format(result))
    f.close()


if __name__ == '__main__':
    main()
