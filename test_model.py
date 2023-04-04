import argparse
import datetime
import torch.nn
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
import utils
import os
from model.hidden import *
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.dropout import Dropout
from noise_layers.identity import Identity
from noise_layers.noiser import Noiser

# --source-image "D:\Рабочий стол\val2014\val2014\COCO_val2014_000000000073"
# --options-file "C:\Users\Pavel\PycharmProjects\HiDDeN_Ando\runs\run_with_noises_crop((0.2,0.3),(0.4,0.5))+cropout((0.11,0.22),(0.33,0.44))+dropout(0.55,0.6)+jpeg() 2023.03.03--14-10-28\options-and-config.pickle"
# --checkpoint-file "C:\Users\Pavel\PycharmProjects\HiDDeN_Ando\runs\run_with_noises_crop((0.2,0.3),(0.4,0.5))+cropout((0.11,0.22),(0.33,0.44))+dropout(0.55,0.6)+jpeg() 2023.03.03--14-10-28\checkpoints\run_with_noises_crop((0.2,0.3),(0.4,0.5))+cropout((0.11,0.22),(0.33,0.44))+dropout(0.55,0.6)+jpeg()--epoch-300.pyt"

PTH_TO_SAVE_IMG = r"D:\Рабочий стол"

DIFF_IMAGE_MULTIPLAYER = 10
MAX_PIXEL_VALUE = 255
MIN_PIXEL_VALUE = 0
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

def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img

# for testing my implementation
def centerCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = img.shape[1]//2
    y = img.shape[0]//2
    img = img[y-height//2:y+height//2, x-width//2:x+width//2]
    return img


def create_diff_image(encoded_images, noised_images, filename):
    np_en_image = utils.tensor_to_image(encoded_images)
    np_noise_image = utils.tensor_to_image(noised_images)

    if np_noise_image.shape[2] < encoded_images.shape[2]:
        print()

    diff_image = DIFF_IMAGE_MULTIPLAYER * np.abs(np_en_image - np_noise_image)

    # border processing
    diff_image = np.where(diff_image < 255, diff_image, 255)
    diff_image = np.where(diff_image < 0, 0, diff_image)

    diff_image = np.squeeze(diff_image, 0)
    diff_tensor = utils.image_to_tensor(diff_image)
    stacked_images = torch.cat([encoded_images.cpu(), noised_images.cpu(), diff_tensor.cpu()], dim=0)

    # create path and filename with actual time
    datetime_date_str = str(datetime.datetime.now().date())
    datetime_time_str = str(datetime.datetime.now().time()).replace(" ", "")
    datetime_time_str = datetime_time_str.split(".")[0]
    datetime_time_str = datetime_time_str.replace(":", "-")
    filename = os.path.join(filename, datetime_date_str + datetime_time_str + 'test.png')

    torchvision.utils.save_image(stacked_images, filename, encoded_images.shape[0], normalize=False)


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--source-image', '-s', required=True, type=str,
                        help='The image to watermark')

    # parser.add_argument('--times', '-t', default=10, type=int,
    #                     help='Number iterations (insert watermark->extract).')

    args = parser.parse_args()

    train_options, hidden_config, noise_config = utils.load_options(args.options_file)

    crop = Crop((CROP_HEIGHT_RATIO_RANGE_MIN, CROP_HEIGHT_RATIO_RANGE_MAX),
                (CROP_WIDTH_RATIO_RANGE_MIN, CROP_WIDTH_RATIO_RANGE_MAX))
    noiser = Noiser([crop, ], device)

    # cropout = Cropout((CROPOUT_HEIGHT_RATIO_RANGE_MIN, CROPOUT_HEIGHT_RATIO_RANGE_MAX),
    #                   (CROPOUT_WIDTH_RATIO_RANGE_MIN, CROPOUT_WIDTH_RATIO_RANGE_MAX))
    # noiser = Noiser([cropout, ], device)

    # dropout = Dropout((DROPOUT_KEEP_RATIO_RANGE_MIN, DROPOUT_KEEP_RATIO_RANGE_MAX))
    # noiser = Noiser([dropout, ], device)

    # noiser = Noiser(["JpegPlaceholder", ], device)

    # source version of noiser
    # noiser = Noiser(noise_config, device)
    # noiser = Noiser(noise_config)

    # убираем бесшумовой слой для тестирования только одного шума
    noiser.remove_noise_layer(0)

    checkpoint = torch.load(args.checkpoint_file)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)

    image_pil = Image.open(args.source_image)
    image = randomCrop(np.array(image_pil), hidden_config.H, hidden_config.W)
    # image = centerCrop(np.array(image_pil), hidden_config.H, hidden_config.W)
    image_tensor = TF.to_tensor(image).to(device)
    image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
    image_tensor.unsqueeze_(0)

    # msg = np.array([[0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0.,
    #                 0., 1., 0., 0., 0., 0., 1.]])
    # message = torch.Tensor(msg).to(device)

    # for t in range(args.times):
    message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
                                                    hidden_config.message_length))).to(device)

    losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([image_tensor, message])
    decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
    message_detached = message.detach().cpu().numpy()
    print('original: {}'.format(message_detached))
    print('decoded : {}'.format(decoded_rounded))
    print('error : {:.3f}'.format(np.mean(np.abs(decoded_rounded - message_detached))))



    create_diff_image(encoded_images, noised_images, PTH_TO_SAVE_IMG)

    # utils.save_images(image_tensor.cpu(), noised_images.cpu(), 'test' + str(datetime.datetime.now().date()) +
    #                   datetime_time_str, PTH_TO_SAVE_IMG, resize_to=(256, 256))

    # bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy()))/(image_tensor.shape[0] * messages.shape[1])



if __name__ == '__main__':
    main()
