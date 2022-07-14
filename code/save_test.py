import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from models.dfanet import dfanet
from models.enet import enet
from models.fastscnn import Fast_SCNN
from models.icnet import IcNet
from models.mobilenet import MobileNet
from models.shuffleseg import ShuffleSeg
from models.unet import unet
import matplotlib.pyplot as plt
from models.shuffleseg_v2 import ShuffleSeg_v2
from models.enet_unet import enet_unet
from models.shufflenet_unet import shufflenet_unet
from misc.unet_v2 import unet_v2
from utilities import class_wise_metrics
from error import *
from models.fastscnn import resize_image
from medpy.io import load
import matplotlib
matplotlib.use('TkAgg')

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
    X_test_normal_, _ = load("../data/fetal_MRI_dataset_test_challenging/Challenging/fetus_37.nii")
    y_test_normal, _ = load("../data/fetal_MRI_dataset_test_challenging/Challenging/goodmask/maskfetus_37.nii")
    X_test_normal_ = np.moveaxis(X_test_normal_, -1, 0) # Bring the last dim to the first
    X_test_normal_ = X_test_normal_[..., np.newaxis] # Add one axis to the end

    y_test_normal = np.moveaxis(y_test_normal, -1, 0)
    y_test_normal = y_test_normal[..., np.newaxis]

    X_test_normal = np.divide(X_test_normal_.astype(float), np.std(X_test_normal_.astype(float), axis=0),
                              out=np.zeros_like(X_test_normal_.astype(float)),
                              where=np.std(X_test_normal_.astype(float), axis=0) != 0)

    """ Hyperparamaters """
    model_path = "../misc/saved_models/icnet"

    """ Load the Model """
    model = load_tf_model(model_path)

    """ Predict """

    mask = model.predict(X_test_normal, batch_size=1)
    mask = np.argmax(mask, axis=3)
    mask = mask[..., tf.newaxis]

    # np.save('../results/results_challenging/' + os.path.basename(model_path) + '_mask.npy', mask)

    # compute the class wise metrics
    i=23
    cls_wise_iou, cls_wise_dice_score = class_wise_metrics(y_test_normal, mask)
    print(np.mean(cls_wise_dice_score))
    print(np.mean(cls_wise_iou))
    # np
    # .save('../results/results_challenging/' + os.path.basename(model_path) + '_out.npy',
    #         np.array([(np.array(cls_wise_dice_score)).mean(), (np.array(cls_wise_iou)).mean()]))
    save_path = '../misc/figures/'
    img = X_test_normal_[i,:,:,0]
    # m = y_test_normal[i,:,:,0]
    m = mask[i, :, :, 0]
    img_masked = np.ma.masked_where(m == 0, m)
    plt.imshow(img, cmap='gray')
    plt.imshow(img_masked, cmap='rainbow',vmin=0,vmax=1, alpha=0.3)
    plt.xticks([])
    plt.yticks([])
    # plt.savefig(save_path+"27.png", bbox_inches='tight',
    #             pad_inches=0)
    plt.show()