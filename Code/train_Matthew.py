#------------------------------------------------------------------------------------------------------------------
import os
import csv
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef, confusion_matrix
from tensorflow.keras.utils import to_categorical

#------------------------------------------------------------------------------------------------------------------

'''
Based on train_example.py (LSDR) and train_Matthew.py
10/24/2026 v2:
  - validation split carved out of the "train" rows (stratified, fixed seed); the "test" rows are never used for training
  - input normalization selectable at the top: none / scale01 / minmax [-1,1] / zscore (mean 0, std 1)
    the normalization is stored INSIDE the model as its first layer, so the test script feeds the same
    0-1 pixels as before (img / 255) and never needs the training statistics
  - optional Dropout, Batch Normalization and Weight Decay (L2) selectable at the top
  - best epoch selected on the VALIDATION set (not on training accuracy)
  - prints macro F1, Cohen kappa and accuracy for validation and test at the end
  - no shuffling, no augmentation (the Excel rows are already in random class order)
  - output files unchanged: model_<NICKNAME>.keras, summary_<NICKNAME>.txt, results_<NICKNAME>.xlsx
'''

#==================================================================================================================
#  EXPERIMENT SETTINGS  -  change these and re-run
#==================================================================================================================

NICKNAME = 'Matthew'

## ---- Input normalization ---------------------------------------------------------------------------------------
##  'none'    : raw pixels 0-255, no normalization at all
##  'scale01' : pixels divided by 255 -> range [0,1]   (what train_Matthew.py v1 did)
##  'minmax'  : book option (i)   p^ = 2(p - pmin)/(pmax - pmin) - 1   element by element  -> range [-1,1]
##  'zscore'  : book option (ii)  p^ = (p - pmean)/pstd               element by element  -> mean 0, std 1
##  pmin, pmax, pmean, pstd are vectors (one value per input element = per pixel per channel),
##  computed ONLY on the training portion (never on validation or test).
NORM_MODE = 'zscore'

## ---- Network ---------------------------------------------------------------------------------------------------
HIDDEN_LAYERS = [300, 200, 100, 100, 80, 50]   # neurons of each hidden Dense layer
ACTIVATION = 'relu'

## ---- Regularization (set to 0 / False to switch off) -----------------------------------------------------------
DROPOUT_RATE = 0.0       # e.g. 0.2 - 0.5 ; fraction of neurons dropped after each hidden layer
USE_BATCHNORM = False    # True -> Dense -> BatchNorm -> activation (-> Dropout)
WEIGHT_DECAY = 0.0       # e.g. 1e-4 ; L2 penalty  alpha * sum(w^2)  added to the loss for every Dense kernel

## ---- Training --------------------------------------------------------------------------------------------------
n_epoch = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001    # RMSprop
VAL_FRACTION = 0.20      # fraction of the "train" rows kept aside for validation
SEED = 42                # same seed -> same validation split and same initial weights

## Which validation number decides the "best" epoch saved to model_<NICKNAME>.keras
##  'val_f1_macro' | 'val_kappa' | 'val_accuracy' | 'val_loss'
CHECKPOINT_MONITOR = 'val_f1_macro'
EARLY_STOP_PATIENCE = 0  # 0 = off ; e.g. 8 -> stop if the monitored value has not improved in 8 epochs

## ---- Speed -----------------------------------------------------------------------------------------------------
## The JPGs are 4000x2672, decoding them is the slowest part of every epoch.
##  None     : decode from disk every epoch (lowest RAM)
##  'memory' : keep the resized images in RAM after the first pass
##             needs about IMAGE_SIZE*IMAGE_SIZE*3*4 bytes per image -> ~5.6 GB for 200x200 (train + validation)
CACHE_MODE = None

## ---- Image processing (must match the test script) -------------------------------------------------------------
CHANNELS = 3
IMAGE_SIZE = 200

#==================================================================================================================

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

## Layout:   <project>/Data/*.jpg
##           <project>/excel/*.xlsx
##           <project>/Code/train_Matthew_v2.py   <- this file
## The project folder is the parent of the folder where this file lives (override with --path).
## Output files (model, summary, results) are saved next to this file.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR
PATH = os.path.dirname(SCRIPT_DIR)
DATA_DIR = PATH + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep

def out_file(name):
    return os.path.join(OUT_DIR, name)
#------------------------------------------------------------------------------------------------------------------

def check_settings():
    '''
        stops early with a clear message if a setting at the top is mistyped
    '''
    if NORM_MODE not in ('none', 'scale01', 'minmax', 'zscore'):
        raise ValueError("NORM_MODE must be 'none', 'scale01', 'minmax' or 'zscore', got {}".format(NORM_MODE))
    if CHECKPOINT_MONITOR not in ('val_f1_macro', 'val_kappa', 'val_accuracy', 'val_loss'):
        raise ValueError('CHECKPOINT_MONITOR not recognised: {}'.format(CHECKPOINT_MONITOR))
    if CACHE_MODE not in (None, 'memory'):
        raise ValueError("CACHE_MODE must be None or 'memory'")
    if not 0.0 <= DROPOUT_RATE < 1.0:
        raise ValueError('DROPOUT_RATE must be in [0, 1)')
    if not 0.0 < VAL_FRACTION < 1.0:
        raise ValueError('VAL_FRACTION must be in (0, 1)')
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
          returns the image as a flat vector of pixels in [0,1] and the target as label
          (identical to the test script; any further normalization happens inside the model)
    '''

    label = target

    file_path = feature
    img = tf.io.read_file(file_path)

    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)

    img = tf.image.resize( img, [IMAGE_SIZE, IMAGE_SIZE])

    # unit conversion 0-255 -> 0-1 so the input matches test_Matthew.py
    # (NORM_MODE 'none' multiplies back by 255 inside the model)
    img = img / 255.0

    img = tf.reshape(img, [IMAGE_SIZE * IMAGE_SIZE * CHANNELS])

    return img, label
#------------------------------------------------------------------------------------------------------------------

def get_target(df, num_classes):
    '''
    Get the one-hot target of the rows in df
    1 = multiclass
    2 = multilabel
    3 = binary
    '''

    y_target = np.array([[int(i) for i in str(x).split(",")] for x in df['target_class']],
                        dtype=np.float32)

    if y_target.ndim != 2 or y_target.shape[1] != num_classes:
        raise ValueError('target_class has shape {} but {} classes were expected'.format(y_target.shape, num_classes))

    return y_target
#------------------------------------------------------------------------------------------------------------------

def read_data(df, num_classes, cache=None):
    '''
          builds the tf.data pipeline for the rows in df
          only the file names are kept in memory, each image is read from disk when its batch is needed
          NO shuffling: the order of the rows in df is kept (needed so predictions line up with the rows)
    '''

    ds_inputs = (DATA_DIR + df['id']).tolist()
    ds_targets = get_target(df, num_classes)

    missing = [p for p in ds_inputs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('{} images not found, for example {} . Check the Data folder: {}'.format(
            len(missing), missing[:3], DATA_DIR))

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs, ds_targets)) # creates a tensor from the image paths and targets

    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    if cache == 'memory':
        final_ds = final_ds.cache()

    final_ds = final_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return final_ds
#------------------------------------------------------------------------------------------------------------------

def compute_input_stats(ds):
    '''
        one pass over the TRAINING images to get, for every input element (pixel x channel):
        pmin, pmax, pmean, pstd   (vectors of length INPUTS_r)
        float64 accumulators so the sums stay exact
    '''
    n = 0
    s1 = np.zeros(INPUTS_r, dtype=np.float64)
    s2 = np.zeros(INPUTS_r, dtype=np.float64)
    pmin = np.full(INPUTS_r, np.inf)
    pmax = np.full(INPUTS_r, -np.inf)

    print('Computing normalization statistics on the training images ...')
    for xb, _ in ds:
        xb = xb.numpy().astype(np.float64)
        n += xb.shape[0]
        s1 += xb.sum(axis=0)
        s2 += (xb ** 2).sum(axis=0)
        pmin = np.minimum(pmin, xb.min(axis=0))
        pmax = np.maximum(pmax, xb.max(axis=0))

    pmean = s1 / n
    pvar = np.maximum(s2 / n - pmean ** 2, 0.0)   # population variance, clipped at 0 for rounding
    pstd = np.sqrt(pvar)

    print('  images used: {}   pixel range [{:.3f}, {:.3f}]   mean of pmean {:.3f}   mean of pstd {:.3f}'.format(
        n, pmin.min(), pmax.max(), pmean.mean(), pstd.mean()))

    return {'pmin': pmin, 'pmax': pmax, 'pmean': pmean, 'pstd': pstd}
#------------------------------------------------------------------------------------------------------------------

def normalization_layer(stats):
    '''
        returns the first layer of the model that applies NORM_MODE.
        Both book formulas are written as  p^ = (p - center) / scale , element by element,
        which is exactly what the built-in Keras Normalization layer computes
        (mean = center, variance = scale^2). Being a built-in layer, it is saved inside
        model_<NICKNAME>.keras and the test script needs no extra code to load it.

          minmax : p^ = 2(p-pmin)/(pmax-pmin) - 1  =  (p - (pmax+pmin)/2) / ((pmax-pmin)/2)
          zscore : p^ = (p - pmean) / pstd
    '''

    if NORM_MODE == 'none':
        # inputs arrive as 0-1, bring them back to the raw 0-255 values
        return tf.keras.layers.Rescaling(255.0, name='no_normalization_raw_0_255')

    if NORM_MODE == 'scale01':
        # inputs already arrive as 0-1
        return None

    if NORM_MODE == 'minmax':
        center = (stats['pmax'] + stats['pmin']) / 2.0
        scale = (stats['pmax'] - stats['pmin']) / 2.0
        name = 'minmax_norm_minus1_to_1'
    else:   # zscore
        center = stats['pmean']
        scale = stats['pstd']
        name = 'zscore_norm_mean0_std1'

    # an element that never changes in the training set (scale 0) is only centred, not divided by 0
    n_const = int(np.sum(scale < 1e-12))
    if n_const > 0:
        print('  {} input elements are constant in the training set, they will only be centred'.format(n_const))
    scale = np.where(scale < 1e-12, 1.0, scale)

    return tf.keras.layers.Normalization(axis=-1,
                                         mean=center.astype(np.float32),
                                         variance=(scale ** 2).astype(np.float32),
                                         name=name)
#------------------------------------------------------------------------------------------------------------------

def save_model(model):
    '''
         receives the model and print the summary into a .txt file
    '''
    with open(out_file('summary_{}.txt'.format(NICKNAME)), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x, **kwargs: fh.write(x + '\n'))
#------------------------------------------------------------------------------------------------------------------

def model_definition(stats):
    # Define a Keras sequential model
    model = tf.keras.Sequential(name='MLP_{}'.format(NICKNAME))

    model.add(tf.keras.layers.Input(shape=(INPUTS_r,)))

    # 1) input normalization (selected with NORM_MODE)
    norm = normalization_layer(stats)
    if norm is not None:
        model.add(norm)

    # 2) hidden layers:  Dense -> [BatchNorm] -> activation -> [Dropout]
    reg = tf.keras.regularizers.L2(WEIGHT_DECAY) if WEIGHT_DECAY > 0 else None

    for units in HIDDEN_LAYERS:
        model.add(tf.keras.layers.Dense(units, kernel_regularizer=reg))
        if USE_BATCHNORM:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(ACTIVATION))
        if DROPOUT_RATE > 0:
            model.add(tf.keras.layers.Dropout(DROPOUT_RATE))

    # 3) output layer, OUTPUTS_a is the number of targets
    model.add(tf.keras.layers.Dense(OUTPUTS_a, activation='softmax', kernel_regularizer=reg))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    save_model(model) #print Summary
    return model
#------------------------------------------------------------------------------------------------------------------

class ValidationMetrics(tf.keras.callbacks.Callback):
    '''
        at the end of every epoch predicts the validation images and adds to the epoch log:
        val_loss (cross-entropy), val_accuracy, val_f1_macro, val_kappa
        ModelCheckpoint / EarlyStopping (placed after this callback) can then monitor any of them.
    '''
    def __init__(self, val_ds, y_val):
        super().__init__()
        self.val_ds = val_ds
        self.y_val = y_val     # integer class index of every validation image

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        prob = self.model.predict(self.val_ds, verbose=0)
        pred = np.argmax(prob, axis=1)

        p_true = np.clip(prob[np.arange(len(self.y_val)), self.y_val], 1e-7, 1.0)
        logs['val_loss'] = float(-np.mean(np.log(p_true)))
        logs['val_accuracy'] = float(accuracy_score(self.y_val, pred))
        logs['val_f1_macro'] = float(f1_score(self.y_val, pred, average='macro'))
        logs['val_kappa'] = float(cohen_kappa_score(self.y_val, pred))

        print('  epoch {:3d} | train loss {:.4f} acc {:.4f} | val loss {:.4f} acc {:.4f} f1_macro {:.4f} kappa {:.4f}'.format(
            epoch + 1, logs.get('loss', np.nan), logs.get('accuracy', np.nan),
            logs['val_loss'], logs['val_accuracy'], logs['val_f1_macro'], logs['val_kappa']))
#------------------------------------------------------------------------------------------------------------------

def train_func(train_ds, val_ds, y_val, stats):
    '''
        train the model, keep the epoch with the best validation score
    '''

    model_file = out_file('model_{}.keras'.format(NICKNAME))
    mode = 'min' if CHECKPOINT_MONITOR == 'val_loss' else 'max'

    callbacks = [ValidationMetrics(val_ds, y_val)]   # must come first: it adds the val_* values to the log
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(model_file, monitor=CHECKPOINT_MONITOR, mode=mode,
                                                        save_best_only=True, verbose=1))
    if EARLY_STOP_PATIENCE > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=CHECKPOINT_MONITOR, mode=mode,
                                                          patience=EARLY_STOP_PATIENCE))

    final_model = model_definition(stats)

    # shuffle=False: the order of the Excel rows is kept (rows are already in random class order)
    history = final_model.fit(train_ds, epochs=n_epoch, shuffle=False, callbacks=callbacks, verbose=2)

    # safety net: if the checkpoint never saved (for example the score was NaN), save the final model
    if not os.path.exists(model_file):
        final_model.save(model_file)

    return history
#------------------------------------------------------------------------------------------------------------------

def predict_labels(ds):
    '''
        loads the best saved model and returns the predicted class index of every image in ds
    '''
    final_model = tf.keras.models.load_model(out_file('model_{}.keras'.format(NICKNAME)))
    res = final_model.predict(ds, verbose=0)
    # res holds probabilities, keep the index of the most probable class
    return np.argmax(res, axis=1)
#------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    '''
        predict function: writes results_<NICKNAME>.xlsx (same format as the template)
    '''

    pred = predict_labels(test_ds)

    if len(pred) != len(xdf_dset):
        raise ValueError('{} predictions for {} rows'.format(len(pred), len(xdf_dset)))

    xdf_dset['results'] = pred
    xdf_dset.to_excel(out_file('results_{}.xlsx'.format(NICKNAME)), index=False)
#------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=None):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_weighted, coh, acc, mat
    list of aggregates : avg, sum
    :return: dict with the value of each metric
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
    values = {}

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

        values[xm] = float(xmet)
        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum )
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum/xcont)

    # rows = true class, columns = predicted class
    print('Confusion matrix (rows = true, columns = predicted), classes:', list(class_names))
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))))

    return values
#------------------------------------------------------------------------------------------------------------------

def settings_text():
    return ('NORM_MODE={} | HIDDEN={} | DROPOUT={} | BATCHNORM={} | WEIGHT_DECAY={} | LR={} | EPOCHS={} | '
            'BATCH={} | IMAGE_SIZE={} | VAL_FRACTION={} | MONITOR={} | SEED={}').format(
        NORM_MODE, HIDDEN_LAYERS, DROPOUT_RATE, USE_BATCHNORM, WEIGHT_DECAY, LEARNING_RATE, n_epoch,
        BATCH_SIZE, IMAGE_SIZE, VAL_FRACTION, CHECKPOINT_MONITOR, SEED)
#------------------------------------------------------------------------------------------------------------------

def log_experiment(history, val_values, test_values):
    '''
        appends one line per run to experiments_<NICKNAME>.csv so different combinations can be compared
        (an extra file; the official output files are not changed)
    '''
    monitored = history.history.get(CHECKPOINT_MONITOR, [])
    best_epoch = ''
    if len(monitored) > 0:
        best_epoch = int(np.argmin(monitored) if CHECKPOINT_MONITOR == 'val_loss' else np.argmax(monitored)) + 1

    row = {'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
           'norm': NORM_MODE, 'hidden': str(HIDDEN_LAYERS), 'dropout': DROPOUT_RATE,
           'batchnorm': USE_BATCHNORM, 'weight_decay': WEIGHT_DECAY, 'lr': LEARNING_RATE,
           'epochs_run': len(history.history.get('loss', [])), 'best_epoch': best_epoch,
           'batch': BATCH_SIZE, 'image_size': IMAGE_SIZE, 'monitor': CHECKPOINT_MONITOR, 'seed': SEED,
           'val_f1_macro': val_values.get('f1_macro'), 'val_kappa': val_values.get('coh'),
           'val_acc': val_values.get('acc'),
           'test_f1_macro': test_values.get('f1_macro'), 'test_kappa': test_values.get('coh'),
           'test_acc': test_values.get('acc')}

    file_name = out_file('experiments_{}.csv'.format(NICKNAME))
    new_file = not os.path.exists(file_name)
    with open(file_name, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print('Run logged in', file_name)
#------------------------------------------------------------------------------------------------------------------

def main():
    global xdf_data, class_names, INPUTS_r, OUTPUTS_a, xdf_dset, PATH, DATA_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, required=False)  # project folder (optional)
    args = parser.parse_args()
    if args.path:
        PATH = os.path.abspath(args.path)
        DATA_DIR = PATH + os.path.sep + 'Data' + os.path.sep

    check_settings()
    tf.keras.utils.set_random_seed(SEED)   # python, numpy and tensorflow seeds -> repeatable runs
    print('SETTINGS:', settings_text())

    FILE_NAME = find_excel()

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    class_names= process_target(1)  # 1: Multiclass 2: Multilabel 3:Binary

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    ## Splitting the "train" rows into training and validation (the "test" rows are left untouched)
    ## stratify keeps the same class proportions in both parts, random_state makes it the same split every run

    xdf_train_all = xdf_data[xdf_data["split"] == 'train'].copy()
    if len(xdf_train_all) == 0:
        raise ValueError('No rows with split == "train" in {}'.format(FILE_NAME))

    xdf_tr, xdf_val = train_test_split(xdf_train_all, test_size=VAL_FRACTION,
                                       stratify=xdf_train_all['target'], random_state=SEED)
    # keep the original Excel order inside each part (no shuffling)
    xdf_tr = xdf_tr.sort_index()
    xdf_val = xdf_val.sort_index()
    print('Training images: {}   Validation images: {}'.format(len(xdf_tr), len(xdf_val)))

    train_ds = read_data(xdf_tr, OUTPUTS_a, cache=CACHE_MODE)
    val_ds = read_data(xdf_val, OUTPUTS_a, cache=CACHE_MODE)
    y_val = xdf_val['target'].map(class_to_idx).to_numpy()

    ## Normalization statistics from the training part only
    stats = None
    if NORM_MODE in ('minmax', 'zscore'):
        stats = compute_input_stats(train_ds)

    history = train_func(train_ds, val_ds, y_val, stats)

    ## Metrics of the best saved model on the validation part

    print('\n================ VALIDATION (best saved epoch) ================')
    xdf_dset = xdf_val.copy()
    xdf_dset['results'] = predict_labels(val_ds)
    val_values = metrics_func(['f1_macro', 'coh', 'acc'], ['avg'])

    ## Test dataset -> results_<NICKNAME>.xlsx

    print('\n================ TEST ================')
    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()
    if len(xdf_dset) == 0:
        raise ValueError('No rows with split == "test" in {}'.format(FILE_NAME))

    test_ds= read_data(xdf_dset, OUTPUTS_a)
    predict_func(test_ds)

    ## Metrics Function over the result of the test dataset
    list_of_metrics = ['f1_macro', 'coh', 'acc']
    list_of_agg = ['avg']
    test_values = metrics_func(list_of_metrics, list_of_agg)

    print('\nSETTINGS:', settings_text())
    log_experiment(history, val_values, test_values)
# ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    main()
#------------------------------------------------------------------------------------------------------------------
