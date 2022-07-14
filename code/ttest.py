import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from models.dfanet import dfanet
from models.enet import enet
from models.fastscnn import Fast_SCNN
from models.icnet import IcNet
from models.mobilenet import MobileNet
from models.shuffleseg import ShuffleSeg
from models.unet import unet

from models.shuffleseg_v2 import ShuffleSeg_v2
from models.enet_unet import enet_unet
from models.shufflenet_unet import shufflenet_unet
from misc.unet_v2 import unet_v2
from utilities import class_wise_metrics
from error import *
from models.fastscnn import resize_image
import statistics
from scipy import stats


def get_model(name):
    IMG_SIZE = 256
    cha_num = 1
    cls_num = 2
    if name == 'dfanet':
        model = dfanet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num, size_factor=2)
    elif name == 'enet':
        model = enet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'fastscnn':
        model = Fast_SCNN(num_classes=cls_num, input_shape=(IMG_SIZE, IMG_SIZE, cha_num)).model()
    elif name == 'icnet':
        model = IcNet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'mobilenet':
        model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'shuffleseg':
        model = ShuffleSeg(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'unet':
        model = unet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    #
    elif name == 'unet_v2':
        model = unet_v2(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'shuffleseg_v2':
        model = ShuffleSeg_v2(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'enet_unet':
        model = enet_unet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'shufflenet_unet':
        model = shufflenet_unet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    else:
        raise NameError("No Model Found!")
    return model

def load_tf_model(path):
    #model = tf.keras.models.load_model('ICNet.model', custom_objects={ 'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef })
    try:
        with tf.keras.utils.CustomObjectScope({'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef,
                                               'resize_image': resize_image }):
            model = tf.keras.models.load_model(path+'.model')
            return model
    except:
        model = get_model(os.path.basename(path))
        model.load_weights(path+'_weights.h5')
        return model
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Load Data """
    X_test_normal = np.load("../data/X_test_normal.npy")
    X_test_normal= np.divide(X_test_normal.astype(float), np.std(X_test_normal.astype(float), axis=0),
                              out=np.zeros_like(X_test_normal.astype(float)),
                              where=np.std(X_test_normal.astype(float), axis=0) != 0)
    X_test_Ch = np.load("../data/X_test_challenging.npy")
    X_test_Ch= np.divide(X_test_Ch.astype(float), np.std(X_test_Ch.astype(float), axis=0),
                              out=np.zeros_like(X_test_Ch.astype(float)),
                              where=np.std(X_test_Ch.astype(float), axis=0) != 0)

    X = np.concatenate((X_test_normal, X_test_Ch))


    y_test_normal = np.load("../data/y_test_normal.npy")
    y_test_CH = np.load("../data/y_test_challenging.npy")
    y = np.concatenate((y_test_normal, y_test_CH))


    """ Hyperparamaters """
    model_path = "../misc/saved_models/unet"
    """ Load the Model """
    model = load_tf_model(model_path)
    """ Predict """
    batch_size = 1
    mask = model.predict(X, batch_size=batch_size)
    mask = np.argmax(mask, axis=3)
    mask = mask[..., tf.newaxis]
    # compute the class wise metrics
    dice = []
    iou = []
    for i in range(mask.shape[0]):
        cls_wise_iou, cls_wise_dice_score = class_wise_metrics(y[i,:,:,0], mask[i,:,:,0])
        dice.append(statistics.mean(cls_wise_dice_score))
        iou.append(statistics.mean(cls_wise_iou))

    np.save('../results/ttest_all/'+os.path.basename(model_path)+'_Dice.npy', np.array(dice))
    np.save('../results/ttest_all/'+os.path.basename(model_path)+'_IoU.npy', np.array(iou))
    print(np.array(dice).mean())
#
path = '../misc/ttest_all/'
RFBSNET_Dice = np.load(path+'unet_v4_Dice.npy')
RFBSNET_IoU = np.load(path+'unet_v4_IoU.npy')

for D, I in zip(os.listdir(path+'dice'), os.listdir(path+'iou')):
    D = os.path.join(path+'dice', D)
    I = os.path.join(path+'iou', I)

    Dice = np.load(D)
    IoU = np.load(I)

    my_test = stats.ttest_rel(RFBSNET_Dice, Dice)
    p_value_D = my_test[1]

    print(os.path.basename(os.path.splitext(D)[0]))
    print("{} ".format(p_value_D))

    my_test = stats.ttest_rel(RFBSNET_IoU, IoU)
    p_value_I = my_test[1]

    print(os.path.basename(os.path.splitext(I)[0]))
    print("{} ".format(p_value_I))

