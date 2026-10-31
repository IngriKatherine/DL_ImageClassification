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
10/25/2026 review: antialiased resize + fast JPEG decode (recorded in the model name for the test script),
                   no Dense bias before BatchNorm, NaN epochs never saved as "best", sudo-free checks,
                   automatic RAM cache (g5.xlarge), summary file written once after training, best model loaded once
10/25/2026 pm: deeper/narrower net (10 layers), minmax, BatchNorm without Dropout, macro-F1 metric, shuffle each epoch
10/25/2026: Adam with Kingma-Ba defaults, dropout 0.2, BatchNorm on, Kaiming init, gradient clipping, NaN stop
10/25/2026: OPTIMIZER option (Adam default) and early stopping with min_delta / restore_best_weights
10/24/2026 v2:
  - validation split carved out of the "train" rows (stratified, fixed seed); the "test" rows are never used for training
  - input normalization selectable at the top: none / scale01 / minmax [-1,1] / zscore (mean 0, std 1)
    the normalization is stored INSIDE the model as its first layer, so the test script feeds the same
    0-1 pixels as before (img / 255) and never needs the training statistics
  - optional Dropout, Batch Normalization and Weight Decay (L2) selectable at the top
  - best epoch selected on the VALIDATION set (not on training accuracy)
  - prints macro F1, Cohen kappa and accuracy for validation and test at the end
  - no augmentation
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
NORM_MODE = 'minmax'

## ---- Network ---------------------------------------------------------------------------------------------------
## Deeper and narrower: a Dense layer has S(m-1)*S(m) + S(m) parameters, so widening layers multiplies the count,
## while adding one more narrow layer only adds S^2 + S. The first layer (120,000 inputs x S1) dominates the total.
## [128, 128, 96, 96, 64, 64, 64, 32, 32, 32]: 10 hidden layers, ~15.4M parameters (vs 6 layers / 36.1M before)
HIDDEN_LAYERS = [128, 128, 96, 96, 64, 64, 64, 32, 32, 32]   # neurons of each hidden Dense layer
ACTIVATION = 'relu'

## ---- Regularization (set to 0 / False to switch off) -----------------------------------------------------------
DROPOUT_RATE = 0.2       # e.g. 0.2 - 0.5 ; used ONLY when USE_BATCHNORM = False (BatchNorm and Dropout are not combined)
USE_BATCHNORM = True     # True -> Dense -> BatchNorm -> activation (-> Dropout).  KEEP True with Adam alpha=0.001 (see Stability)
WEIGHT_DECAY = 0.0       # e.g. 1e-4 ; L2 penalty  alpha * sum(w^2)  added to the loss for every Dense kernel

## ---- Training --------------------------------------------------------------------------------------------------
n_epoch = 100            # upper limit; with early stopping the run usually ends well before
BATCH_SIZE = 32
LEARNING_RATE = 0.001    # alpha. Kingma and Ba [2014] default for Adam (book Ch.3)

## ---- Optimizer (gradient descent algorithm) -------------------------------------------------------------------
##  'adam'    : Adam - momentum (running average of the gradient, beta_1) combined with a per-weight step size
##              (running average of the squared gradient, beta_2), both bias-corrected
##  'rmsprop' : RMSprop - per-weight step size only (what the template used)
##  'sgd'     : plain stochastic gradient descent with momentum (book: SDBP with momentum)
OPTIMIZER = 'adam'
## Kingma and Ba [2014] defaults (book Ch.3, Eq. 3.29-3.33): alpha = 0.001, beta1 = 0.9, beta2 = 0.999, e = 1e-8
ADAM_BETA_1 = 0.9        # decay of the smoothed gradient m (momentum)
ADAM_BETA_2 = 0.999      # decay of the smoothed squared gradient v
ADAM_EPSILON = 1e-8      # e, avoids division by 0  (Keras default is 1e-7, set to the paper value)
SGD_MOMENTUM = 0.9

## ---- Stability: keeping the training from exploding (book Ch.4, Supplemental Training Procedures) ------------
## 1) input normalization  -> NORM_MODE above ('zscore' or 'minmax'; 'none' feeds raw 0-255 values and is the risky one)
## 2) weight initialization -> Kaiming (He) for ReLU layers: weights with variance 2 / S^(m-1)   (book Eq. 4.34)
##                             'glorot_uniform' is the Keras default (Xavier), better suited to tansig/sigmoid
## 3) gradient clipping     -> [Pascanu et al., 2013]: if the norm of the whole gradient goes beyond CLIP_NORM it is
##                             rescaled so its norm equals CLIP_NORM (same direction, shorter step). 0 = off
##                             Fully effective with SGD/momentum; with Adam/RMSprop it only tames single bad batches
## 4) batch normalization -> USE_BATCHNORM above. With Adam every weight moves about alpha per step whatever the size
##                             of the gradient (the step is divided by sqrt(v^)), so clipping alone cannot stop it: with
##                             120,000 inputs the net input of each first-layer neuron jumps by ~alpha*120,000 per step and
##                             the loss blows up (tested: loss 12-37 with 12k inputs, 50-120 with 49k inputs, Adam 1e-3).
##                             BatchNorm re-normalizes the net input of every layer each batch -> loss 2.2 -> 0.7, stable.
##                             Without BatchNorm, use LEARNING_RATE = 1e-5 or smaller.
## 5) a NaN/inf loss stops the run immediately (the best epoch saved so far is kept)
KERNEL_INIT = 'he_normal'   # 'he_normal' (Kaiming) | 'glorot_uniform' (Xavier, Keras default)
CLIP_NORM = 1.0
VAL_FRACTION = 0.20      # fraction of the "train" rows kept aside for validation
SEED = 42                # same seed -> same validation split and same initial weights

## Shuffling: the order of the TRAINING images is reshuffled at the start of every epoch (seeded -> repeatable).
## Validation and test are never shuffled (their predictions must stay aligned with the Excel rows).
SHUFFLE_EACH_EPOCH = True

## Which validation number decides the "best" epoch saved to model_<NICKNAME>.keras
##  'val_f1_macro' | 'val_kappa' | 'val_accuracy' | 'val_loss'
CHECKPOINT_MONITOR = 'val_f1_macro'

## ---- Early stopping ------------------------------------------------------------------------------------------
## After every epoch the monitored validation value (CHECKPOINT_MONITOR) is checked. If it has not improved by at
## least EARLY_STOP_MIN_DELTA for EARLY_STOP_PATIENCE epochs in a row, training stops. The weights of the best
## epoch are the ones saved in model_<NICKNAME>.keras (and restored in memory).
EARLY_STOP_PATIENCE = 10     # 0 = off
EARLY_STOP_MIN_DELTA = 0.002 # smaller changes do not count as an improvement (F1 / kappa / accuracy units)

## ---- Speed -----------------------------------------------------------------------------------------------------
## Decoding the 4000x2672 JPGs is the slowest part of training (the g5.xlarge has only 4 vCPUs for it; the GPU waits).
## With the cache, every image is decoded once and kept in RAM, later epochs (and the validation of every epoch)
## read it from memory. Shuffling still works on the cached images (tested: no extra copy in memory).
##  'auto'   : use the RAM cache if it fits (needed RAM vs free RAM, keeping CACHE_HEADROOM_GB for TensorFlow/CUDA)
##  'memory' : always use the RAM cache
##  None     : decode from disk every epoch (lowest RAM, slowest)
## RAM needed = IMAGE_SIZE*IMAGE_SIZE*3*4 bytes per image: 200x200 -> ~4.6 GB train + ~1.2 GB validation = ~5.8 GB
## (g5.xlarge: 16 GB RAM -> fits)
CACHE_MODE = 'auto'
CACHE_HEADROOM_GB = 4.0

## ---- Image processing ------------------------------------------------------------------------------------------
## These choices are written into the model name (e.g. MLP_Matthew_prep_r8_aa), so test_Matthew.py reads them from the
## model and always prepares the images exactly as in training. Models without that tag use the old preparation.
CHANNELS = 3
IMAGE_SIZE = 200
## JPEG_DECODE_RATIO: decode the JPEG directly at 1/ratio of its size (1, 2, 4 or 8). The photos are 4000x2672, so
## ratio 8 gives 500x334 -> about 2x faster per image, same final quality. Keep 2672/ratio >= IMAGE_SIZE.
JPEG_DECODE_RATIO = 8
## ANTIALIAS: average all the pixels that fall into each output pixel when shrinking. Without it, bilinear resizing
## from 4000 px to 200 px samples only ~1 pixel in 20 (aliasing). Tested on the real images: mean error vs a true
## area average 3.9 (no antialias) vs 1.2 (antialias), on the 0-255 scale.
ANTIALIAS = True

#==================================================================================================================

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

## Layout:   <project>/Data/*.jpg
##           <project>/excel/*.xlsx
##           <project>/Code/train_Matthew.py   <- this file
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
    if CACHE_MODE not in (None, 'memory', 'auto'):
        raise ValueError("CACHE_MODE must be 'auto', 'memory' or None")
    if not 0.0 <= DROPOUT_RATE < 1.0:
        raise ValueError('DROPOUT_RATE must be in [0, 1)')
    if KERNEL_INIT not in ('he_normal', 'glorot_uniform'):
        raise ValueError("KERNEL_INIT must be 'he_normal' or 'glorot_uniform'")
    if OPTIMIZER not in ('adam', 'rmsprop', 'sgd'):
        raise ValueError("OPTIMIZER must be 'adam', 'rmsprop' or 'sgd', got {}".format(OPTIMIZER))
    if not 0.0 < VAL_FRACTION < 1.0:
        raise ValueError('VAL_FRACTION must be in (0, 1)')
    if JPEG_DECODE_RATIO not in (1, 2, 4, 8):
        raise ValueError('JPEG_DECODE_RATIO must be 1, 2, 4 or 8')
    if 2672 // JPEG_DECODE_RATIO < IMAGE_SIZE:
        print('WARNING: 2672/JPEG_DECODE_RATIO = {} is smaller than IMAGE_SIZE = {}: images would be enlarged, '
              'use a smaller JPEG_DECODE_RATIO'.format(2672 // JPEG_DECODE_RATIO, IMAGE_SIZE))
#------------------------------------------------------------------------------------------------------------------

def resolve_cache_mode(n_images):
    '''
        decides whether the decoded images of training + validation are kept in RAM
        returns 'memory' or None
    '''
    if CACHE_MODE != 'auto':
        return CACHE_MODE

    need_gb = n_images * IMAGE_SIZE * IMAGE_SIZE * CHANNELS * 4 / 1e9
    try:
        with open('/proc/meminfo') as fh:
            info = {l.split(':')[0]: int(l.split()[1]) for l in fh if ':' in l}
        free_gb = info['MemAvailable'] / 1e6
    except Exception:
        print('Cache: free RAM unknown (not Linux?), images will be read from disk every epoch')
        return None

    if need_gb <= free_gb - CACHE_HEADROOM_GB:
        print('Cache: ON  - {:.1f} GB of images kept in RAM ({:.1f} GB free)'.format(need_gb, free_gb))
        return 'memory'
    print('Cache: OFF - {:.1f} GB needed but only {:.1f} GB free (minus {} GB headroom); '
          'images are read from disk every epoch'.format(need_gb, free_gb, CACHE_HEADROOM_GB))
    return None
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

    if JPEG_DECODE_RATIO > 1:
        # the JPEG decoder itself shrinks the image by JPEG_DECODE_RATIO (much less work than a full decode)
        img = tf.io.decode_jpeg(img, channels=CHANNELS, ratio=JPEG_DECODE_RATIO)
    else:
        img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)

    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], antialias=ANTIALIAS)

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

def read_data(df, num_classes, cache=None, shuffle=False):
    '''
          builds the tf.data pipeline for the rows in df
          only the file names are kept in memory, each image is read from disk when its batch is needed
          shuffle=False keeps the order of the rows in df (needed so predictions line up with the rows)
          shuffle=True  (training only) gives a new random order of the images at every epoch
    '''

    ds_inputs = (DATA_DIR + df['id']).tolist()
    ds_targets = get_target(df, num_classes)

    missing = [p for p in ds_inputs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('{} images not found, for example {} . Check the Data folder: {}'.format(
            len(missing), missing[:3], DATA_DIR))

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs, ds_targets)) # creates a tensor from the image paths and targets

    n = len(ds_inputs)

    if cache == 'memory':
        # decode once, keep in RAM, then reshuffle the cached images every epoch
        final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).cache()
        if shuffle:
            final_ds = final_ds.shuffle(n, seed=SEED, reshuffle_each_iteration=True)
    else:
        # shuffle only the file names (tiny) before the images are read
        if shuffle:
            list_ds = list_ds.shuffle(n, seed=SEED, reshuffle_each_iteration=True)
        final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

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

    # an element that (almost) never changes in the training set would be divided by ~0 and blow up to huge
    # values on a new image; its scale is floored at MIN_SCALE (1/1000 of the 0-1 pixel range)
    MIN_SCALE = 1e-3
    n_small = int(np.sum(scale < MIN_SCALE))
    if n_small > 0:
        print('  {} input elements have almost no spread in the training set, their scale is floored at {}'.format(
            n_small, MIN_SCALE))
    scale = np.maximum(scale, MIN_SCALE)

    return tf.keras.layers.Normalization(axis=-1,
                                         mean=center.astype(np.float32),
                                         variance=(scale ** 2).astype(np.float32),
                                         name=name)
#------------------------------------------------------------------------------------------------------------------

def save_model(model):
    '''
         receives the model and print the summary into a .txt file
    '''
    os.environ['COLUMNS'] = '80'   # same summary width as an automated run (no terminal)
    with open(out_file('summary_{}.txt'.format(NICKNAME)), 'w', encoding='utf-8', newline='\n') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x, **kwargs: fh.write(x + '\n'))
#------------------------------------------------------------------------------------------------------------------

def make_optimizer():
    '''
        returns the optimizer selected with OPTIMIZER

        Adam, for every weight w with gradient g at step t:
            m = beta_1 * m + (1 - beta_1) * g          (average gradient  -> momentum)
            v = beta_2 * v + (1 - beta_2) * g^2        (average squared gradient -> per-weight scale)
            m^ = m / (1 - beta_1^t),  v^ = v / (1 - beta_2^t)   (bias correction for the first steps)
            w = w - LEARNING_RATE * m^ / (sqrt(v^) + epsilon)
    '''
    # gradient clipping on the norm of the whole gradient vector (all weights together), as in Pascanu et al.
    clip = {'global_clipnorm': CLIP_NORM} if CLIP_NORM > 0 else {}

    if OPTIMIZER == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1,
                                        beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON, **clip)
    if OPTIMIZER == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, **clip)
    return tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=SGD_MOMENTUM, **clip)
#------------------------------------------------------------------------------------------------------------------

def model_name():
    return 'MLP_{}_prep_r{}{}'.format(NICKNAME, JPEG_DECODE_RATIO, '_aa' if ANTIALIAS else '')
#------------------------------------------------------------------------------------------------------------------

def model_definition(stats):
    # Define a Keras sequential model
    # the name records how the images were prepared, test_Matthew.py reads it back (see Image processing settings)
    model = tf.keras.Sequential(name=model_name())

    model.add(tf.keras.layers.Input(shape=(INPUTS_r,)))

    # 1) input normalization (selected with NORM_MODE)
    norm = normalization_layer(stats)
    if norm is not None:
        model.add(norm)

    # 2) hidden layers:  Dense -> [BatchNorm] -> activation -> [Dropout]
    reg = tf.keras.regularizers.L2(WEIGHT_DECAY) if WEIGHT_DECAY > 0 else None

    for units in HIDDEN_LAYERS:
        # with BatchNorm the Dense bias is redundant (BatchNorm subtracts the mean and adds its own shift beta)
        model.add(tf.keras.layers.Dense(units, use_bias=not USE_BATCHNORM,
                                        kernel_initializer=KERNEL_INIT, kernel_regularizer=reg))
        if USE_BATCHNORM:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(ACTIVATION))
        if DROPOUT_RATE > 0 and not USE_BATCHNORM:
            model.add(tf.keras.layers.Dropout(DROPOUT_RATE))

    # 3) output layer, OUTPUTS_a is the number of targets
    model.add(tf.keras.layers.Dense(OUTPUTS_a, activation='softmax', kernel_regularizer=reg))

    # metric shown during training: macro F1 (each class counts the same, as in the leaderboard metrics) instead of
    # accuracy, which looks good just by favouring the large classes. Metrics are only reported, the weights are
    # trained on the loss (cross-entropy).
    model.compile(optimizer=make_optimizer(), loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.F1Score(average='macro', name='f1_macro')])

    # shown on screen only; summary_<NICKNAME>.txt is written once after training, from the saved model
    model.summary()
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

        if not np.all(np.isfinite(prob)):
            # the weights exploded: report the worst possible values so this epoch is never saved as the best one
            logs.update({'val_loss': np.inf, 'val_accuracy': -1.0, 'val_f1_macro': -1.0, 'val_kappa': -1.0})
            print('  epoch {:3d} | NaN/inf in the predictions, this epoch is discarded'.format(epoch + 1))
            return

        pred = np.argmax(prob, axis=1)

        p_true = np.clip(prob[np.arange(len(self.y_val)), self.y_val], 1e-7, 1.0)
        logs['val_loss'] = float(-np.mean(np.log(p_true)))
        logs['val_accuracy'] = float(accuracy_score(self.y_val, pred))
        logs['val_f1_macro'] = float(f1_score(self.y_val, pred, average='macro'))
        logs['val_kappa'] = float(cohen_kappa_score(self.y_val, pred))

        print('  epoch {:3d} | train loss {:.4f} f1_macro {:.4f} | val loss {:.4f} acc {:.4f} f1_macro {:.4f} kappa {:.4f}'.format(
            epoch + 1, logs.get('loss', np.nan), logs.get('f1_macro', np.nan),
            logs['val_loss'], logs['val_accuracy'], logs['val_f1_macro'], logs['val_kappa']))
#------------------------------------------------------------------------------------------------------------------

def train_func(train_ds, val_ds, y_val, stats):
    '''
        train the model, keep the epoch with the best validation score
    '''

    model_file = out_file('model_{}.keras'.format(NICKNAME))
    mode = 'min' if CHECKPOINT_MONITOR == 'val_loss' else 'max'

    callbacks = [tf.keras.callbacks.TerminateOnNaN(),   # stops at once if the loss explodes to NaN / inf
                 ValidationMetrics(val_ds, y_val)]     # must come before the checkpoint: it adds the val_* values
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(model_file, monitor=CHECKPOINT_MONITOR, mode=mode,
                                                        save_best_only=True, verbose=1))
    if EARLY_STOP_PATIENCE > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=CHECKPOINT_MONITOR, mode=mode,
                                                          patience=EARLY_STOP_PATIENCE,
                                                          min_delta=EARLY_STOP_MIN_DELTA,
                                                          restore_best_weights=True, verbose=1))

    final_model = model_definition(stats)

    # shuffle=False here because shuffling is done by the tf.data pipeline (read_data); Keras cannot shuffle a tf.data
    history = final_model.fit(train_ds, epochs=n_epoch, shuffle=False, callbacks=callbacks, verbose=2)

    # no epoch was ever saved (every epoch exploded): stop with a clear message instead of saving a broken model
    if not os.path.exists(model_file):
        raise RuntimeError('No valid epoch was saved (loss NaN/inf from the start). Lower LEARNING_RATE or turn on '
                           'USE_BATCHNORM / a NORM_MODE, then run again.')

    # the summary written before training has no optimizer state; the grader compares against the summary of
    # the SAVED model (which includes "Optimizer params"), so rewrite it from the file that will be submitted
    best_model = tf.keras.models.load_model(model_file)
    save_model(best_model)

    return history, best_model
#------------------------------------------------------------------------------------------------------------------

def predict_labels(ds, final_model):
    '''
        returns the predicted class index of every image in ds
    '''
    res = final_model.predict(ds, verbose=0)
    # res holds probabilities, keep the index of the most probable class
    return np.argmax(res, axis=1)
#------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds, final_model):
    '''
        predict function: writes results_<NICKNAME>.xlsx (same format as the template)
    '''

    pred = predict_labels(test_ds, final_model)

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

def dropout_used():
    # Dropout is not combined with BatchNorm
    return DROPOUT_RATE if not USE_BATCHNORM else 0.0

def settings_text():
    return ('NORM_MODE={} | HIDDEN={} | DROPOUT={} | BATCHNORM={} | WEIGHT_DECAY={} | OPTIMIZER={} | LR={} | '
            'INIT={} | CLIP={} | EPOCHS={} | EARLY_STOP={}/{} | SHUFFLE={} | BATCH={} | IMAGE_SIZE={} | '
            'JPEG_RATIO={} | ANTIALIAS={} | VAL_FRACTION={} | MONITOR={} | SEED={}').format(
        NORM_MODE, HIDDEN_LAYERS, dropout_used(), USE_BATCHNORM, WEIGHT_DECAY, OPTIMIZER, LEARNING_RATE, KERNEL_INIT, CLIP_NORM, n_epoch,
        EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA, SHUFFLE_EACH_EPOCH, BATCH_SIZE, IMAGE_SIZE,
        JPEG_DECODE_RATIO, ANTIALIAS, VAL_FRACTION, CHECKPOINT_MONITOR, SEED)
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
           'norm': NORM_MODE, 'hidden': str(HIDDEN_LAYERS), 'dropout': dropout_used(),
           'batchnorm': USE_BATCHNORM, 'weight_decay': WEIGHT_DECAY,
           'optimizer': OPTIMIZER, 'lr': LEARNING_RATE, 'init': KERNEL_INIT, 'clip': CLIP_NORM, 'shuffle': SHUFFLE_EACH_EPOCH, 'patience': EARLY_STOP_PATIENCE,
           'epochs_run': len(history.history.get('loss', [])), 'best_epoch': best_epoch,
           'batch': BATCH_SIZE, 'image_size': IMAGE_SIZE, 'jpeg_ratio': JPEG_DECODE_RATIO, 'antialias': ANTIALIAS, 'monitor': CHECKPOINT_MONITOR, 'seed': SEED,
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
    # keep the original Excel order inside each part (training order is reshuffled every epoch if SHUFFLE_EACH_EPOCH)
    xdf_tr = xdf_tr.sort_index()
    xdf_val = xdf_val.sort_index()
    print('Training images: {}   Validation images: {}'.format(len(xdf_tr), len(xdf_val)))

    cache = resolve_cache_mode(len(xdf_tr) + len(xdf_val))
    train_ds = read_data(xdf_tr, OUTPUTS_a, cache=cache, shuffle=SHUFFLE_EACH_EPOCH)
    val_ds = read_data(xdf_val, OUTPUTS_a, cache=cache)
    y_val = xdf_val['target'].map(class_to_idx).to_numpy()

    ## Normalization statistics from the training part only
    stats = None
    if NORM_MODE in ('minmax', 'zscore'):
        stats = compute_input_stats(train_ds)

    # best_model = the best saved epoch, loaded once from model_<NICKNAME>.keras for validation and test
    history, best_model = train_func(train_ds, val_ds, y_val, stats)

    ## Metrics of the best saved model on the validation part

    print('\n================ VALIDATION (best saved epoch) ================')
    xdf_dset = xdf_val.copy()
    xdf_dset['results'] = predict_labels(val_ds, best_model)
    val_values = metrics_func(['f1_macro', 'coh', 'acc'], ['avg'])

    ## Test dataset -> results_<NICKNAME>.xlsx

    print('\n================ TEST ================')
    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()
    if len(xdf_dset) == 0:
        raise ValueError('No rows with split == "test" in {}'.format(FILE_NAME))

    test_ds= read_data(xdf_dset, OUTPUTS_a)
    predict_func(test_ds, best_model)

    ## Metrics Function over the result of the test dataset
    list_of_metrics = ['f1_macro', 'coh', 'acc']
    list_of_agg = ['avg']
    test_values = metrics_func(list_of_metrics, list_of_agg)

    print('\nSETTINGS:', settings_text(), '| CACHE={}'.format(cache))
    log_experiment(history, val_values, test_values)
# ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    main()
#------------------------------------------------------------------------------------------------------------------
