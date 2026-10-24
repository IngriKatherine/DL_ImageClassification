#------------------------------------------------------------------------------------------------------------------
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical

#------------------------------------------------------------------------------------------------------------------

'''
LAST UPDATE 10/20/2021 LSDR
last update 10/21/2021 lsdr
02/14/2022 am LSDR CHECK CONSISTENCY
02/14/2022 pm LSDR Change result for results
10/23/2026 REVIEWED: pixels scaled to 0-1, training data shuffled, batch size 32 with prefetch,
           metrics average fixed, multilabel branch fixed, clearer errors for missing files
'''
#------------------------------------------------------------------------------------------------------------------

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file
##
## Layout:   <project>/Data/*.jpg
##           <project>/excel/*.xlsx
##           <project>/<code folder>/train_Matthew.py   <- this file
## The project folder is the parent of the folder where this file lives (override with --path).
## Output files (model, summary, results) are saved next to this file.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR
PATH = os.path.dirname(SCRIPT_DIR)
DATA_DIR = PATH + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep

n_epoch = 1
BATCH_SIZE = 32          # images are read from disk batch by batch, never all at once

## Image processing
CHANNELS = 3
IMAGE_SIZE = 200

NICKNAME = 'Matthew'

def out_file(name):
    return os.path.join(OUT_DIR, name)
#------------------------------------------------------------------------------------------------------------------

def find_excel():
    '''
        returns the path of the .xlsx file inside the "excel" folder
    '''
    folder = PATH + os.path.sep + "excel"
    if not os.path.isdir(folder):
        raise FileNotFoundError('Folder not found: {}'.format(folder))

    files = sorted(f for f in os.listdir(folder) if f.endswith('.xlsx') and not f.startswith('~$'))
    if len(files) == 0:
        raise FileNotFoundError('No .xlsx file found in {}'.format(folder))
    if len(files) > 1:
        print('More than one .xlsx file found, using', files[0])

    return folder + os.path.sep + files[0]
#------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    '''
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Binary   target = (1,0)

    :return: class_names (sorted)
    '''

    if target_type == 1:
        class_names = np.sort(xdf_data['target'].unique())

        class_to_idx = {c: i for i, c in enumerate(class_names)}
        final_target = xdf_data['target'].map(class_to_idx).to_numpy()

        final_target = to_categorical(final_target, num_classes=len(class_names))

        # one-hot vector stored as text "0,0,1,0,0,0" so it fits in one cell
        xdf_data['target_class'] = [",".join(str(int(e)) for e in row) for row in final_target]

    elif target_type == 2:
        labels = xdf_data['target'].astype(str).apply(lambda x: [t.strip() for t in x.split(",")])

        class_names = np.sort(np.unique([t for row in labels for t in row]))
        class_to_idx = {c: i for i, c in enumerate(class_names)}

        multi_hot = np.zeros((len(labels), len(class_names)), dtype=int)
        for r, row in enumerate(labels):
            for t in row:
                multi_hot[r, class_to_idx[t]] = 1

        xdf_data['target_class'] = [",".join(str(e) for e in row) for row in multi_hot]

    elif target_type == 3:
        # target_class is already done
        class_names = np.sort(xdf_data['target'].unique())

    else:
        raise ValueError('target_type must be 1, 2 or 3')

    return class_names
#------------------------------------------------------------------------------------------------------------------

def process_path(feature, target):
    '''
          feature is the path and id of the image
          target is the result
          returns the image and the target as label
    '''

    label = target

    file_path = feature
    img = tf.io.read_file(file_path)

    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)

    img = tf.image.resize( img, [IMAGE_SIZE, IMAGE_SIZE])

    # scale pixels from 0-255 to 0-1 (raw 0-255 values make the loss explode)
    img = img / 255.0

    # augmentation

    img = tf.reshape(img, [IMAGE_SIZE * IMAGE_SIZE * CHANNELS])

    return img, label
#------------------------------------------------------------------------------------------------------------------

def get_target(num_classes):
    '''
    Get the target from the dataset
    1 = multiclass
    2 = multilabel
    3 = binary
    '''

    y_target = np.array([[int(i) for i in str(x).split(",")] for x in xdf_dset['target_class']],
                        dtype=np.float32)

    if y_target.ndim != 2 or y_target.shape[1] != num_classes:
        raise ValueError('target_class has shape {} but {} classes were expected'.format(y_target.shape, num_classes))

    return y_target
#------------------------------------------------------------------------------------------------------------------


def read_data(num_classes, training=False):
    '''
          reads the dataset and process the target
          only the file names are kept in memory, each image is read from disk when its batch is needed
          training=True shuffles the order of the images on every epoch
    '''

    ds_inputs = (DATA_DIR + xdf_dset['id']).tolist()
    ds_targets = get_target(num_classes)

    missing = [p for p in ds_inputs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('{} images not found, for example {} . Check the Data folder: {}'.format(
            len(missing), missing[:3], DATA_DIR))

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs, ds_targets)) # creates a tensor from the image paths and targets

    if training:
        # shuffling only file names and labels (tiny), before the images are loaded
        list_ds = list_ds.shuffle(len(ds_inputs), reshuffle_each_iteration=True)

    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return final_ds
#------------------------------------------------------------------------------------------------------------------

def save_model(model):
    '''
         receives the model and print the summary into a .txt file
    '''
    with open(out_file('summary_{}.txt'.format(NICKNAME)), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x, **kwargs: fh.write(x + '\n'))
#------------------------------------------------------------------------------------------------------------------

def model_definition():
    # Define a Keras sequential model
    model = tf.keras.Sequential()

    # Define the input and the first dense layer
    model.add(tf.keras.layers.Input(shape=(INPUTS_r,)))
    model.add(tf.keras.layers.Dense(300, activation='relu'))
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(80, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))

    model.add(tf.keras.layers.Dense(OUTPUTS_a, activation='softmax')) #final layer , outputs_a is the number of targets

    model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

    save_model(model) #print Summary
    return model
#------------------------------------------------------------------------------------------------------------------

def train_func(train_ds):
    '''
        train the model
    '''

    model_file = out_file('model_{}.keras'.format(NICKNAME))

    #early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience =100)
    check_point = tf.keras.callbacks.ModelCheckpoint(model_file, monitor='accuracy', save_best_only=True)
    final_model = model_definition()

    #final_model.fit(train_ds,  epochs=n_epoch, callbacks=[early_stop, check_point])
    # shuffle=False because the dataset already shuffles itself in read_data(training=True)
    final_model.fit(train_ds,  epochs=n_epoch, shuffle=False, callbacks=[check_point])

    # safety net: if the checkpoint never saved (for example accuracy was NaN), save the final model
    if not os.path.exists(model_file):
        final_model.save(model_file)
#------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    '''
        predict fumction
    '''

    model_file = out_file('model_{}.keras'.format(NICKNAME))
    final_model = tf.keras.models.load_model(model_file)
    res = final_model.predict(test_ds)

    if len(res) != len(xdf_dset):
        raise ValueError('{} predictions for {} rows'.format(len(res), len(xdf_dset)))

    # res holds probabilities, keep the index of the most probable class
    xdf_dset['results'] = np.argmax(res, axis=1)
    xdf_dset.to_excel(out_file('results_{}.xlsx'.format(NICKNAME)), index=False)
#------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=None):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_weighted, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    if aggregates is None:
        aggregates = []

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        print("f1_score {}".format(type), res)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        print("cohen_kappa_score", res)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        print("accuracy_score", res)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        print('mattews_coef', res)
        return res


    # For multiclass

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_true = xdf_dset['target'].map(class_to_idx).to_numpy()
    y_pred = xdf_dset['results'].to_numpy()

    # End of Multiclass

    xcont = 0   # number of metrics that were actually computed
    xsum = 0

    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average = weighted
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet = accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet = matthews_metric(y_true, y_pred)
        else:
            print('Metric does not exist:', xm)
            continue

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum )
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum/xcont)
    # Ask for arguments for each metric
#------------------------------------------------------------------------------------------------------------------

def main():
    global xdf_data, class_names, INPUTS_r, OUTPUTS_a, xdf_dset, PATH, DATA_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, required=False)  # project folder (optional)
    args = parser.parse_args()
    if args.path:
        PATH = os.path.abspath(args.path)
        DATA_DIR = PATH + os.path.sep + 'Data' + os.path.sep

    FILE_NAME = find_excel()

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    class_names= process_target(1)  # 1: Multiclass 2: Multilabel 3:Binary

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    ## Processing Train dataset

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
    if len(xdf_dset) == 0:
        raise ValueError('No rows with split == "train" in {}'.format(FILE_NAME))

    train_ds = read_data( OUTPUTS_a, training=True )
    train_func(train_ds)

    # Preprocessing Test dataset

    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()
    if len(xdf_dset) == 0:
        raise ValueError('No rows with split == "test" in {}'.format(FILE_NAME))

    test_ds= read_data(OUTPUTS_a)
    predict_func(test_ds)

    ## Metrics Function over the result of the test dataset
    list_of_metrics = ['f1_macro', 'coh']
    list_of_agg = ['avg']
    metrics_func(list_of_metrics, list_of_agg)
# ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    main()
#------------------------------------------------------------------------------------------------------------------