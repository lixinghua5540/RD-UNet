import keras
import keras.backend as K
from tqdm import tqdm

class ADAMLearningRateTracker(keras.callbacks.Callback):


    def __init__(self, end_lr):
        super(ADAMLearningRateTracker, self).__init__()
        self.end_lr = end_lr

    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        print('\n***The last Basic Learning rate in this epoch is:', K.eval(optimizer.lr), '***\n')
        if K.eval(optimizer.lr) <= self.end_lr:
            print("training is finished")
            self.model.stop_training = True


def get_input_image_names(list_names, directory_name, if_train=True):
    list_img = []
    list_msk = []
    list_test_ids = []

    for filenames in tqdm(list_names['name'], miniters=1000):
        img=filenames+'data'
        if if_train:
            dir_type_name = "img"
            fl_img = []
            nmask = filenames+'qmask'
            fl_msk = directory_name + '/mask/' + '{}.tif'.format(nmask)
            list_msk.append(fl_msk)

        else:
            dir_type_name = "img"
            fl_img = []
            fl_id = '{}.TIF'.format(filenames)
            list_test_ids.append(fl_id)
        fl_img.append(directory_name + '/' + dir_type_name + '/' +'{}.tif'.format(img))
        list_img.append(fl_img)

    if if_train:
        return list_img, list_msk
    else:
        return list_img, list_test_ids
