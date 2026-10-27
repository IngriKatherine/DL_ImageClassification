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
10/27/2026: wider network (neurons of every hidden layer doubled, ~31M parameters), BatchNorm OFF, Dropout 0.2 after
            every hidden layer. The input normalization (minmax, first layer of the model) is unchanged.
10/26/2026 pm: generalization (book NND2 Ch.13 "Methods for Improving Generalization", Early Stopping, Fig. 13.4)
            - EARLY STOPPING as in the book: the validation error F (the same cross-entropy the network is trained on)
              decides when to stop AND which weights are kept. Point "a" = epoch with the minimum validation error;
              training stops when F_val has not gone below it for EARLY_STOP_PATIENCE epochs (point "b") and the
              weights of point "a" are saved. Before, val macro F1 with patience 10 stopped the run by chance at
              epoch 32 while F_val was still going down (no point "a" reached).
              Every epoch the training error F is also measured on the ORIGINAL training images in inference mode,
              so the two curves of Fig. 13.4 are comparable. early_stopping_<NICKNAME>.png reproduces Fig. 13.4
              (log-log, a and b marked) and early_stopping_<NICKNAME>.csv keeps the numbers of every epoch.
            - LEARNING-RATE SCHEDULE: alpha is halved when F_val stops improving (book: "use a relatively slow
              training method", so the training does not jump past point "a")
            - CLASS WEIGHTS: each training image of class c weighs N/(K*n_c) in the loss (balanced), so the small
              classes count as much as the big ones in the gradient (macro F1)
            - TEST-TIME AUGMENTATION: final prediction = average of the softmax over the original image and its
              flipped versions (left-right, up-down, both). The flips used in training are written into the model
              name (e.g. MLP_Matthew_flip_hv_prep_r8_aa) and test_Matthew.py reads them back
10/26/2026: geometric DATA AUGMENTATION of the training images only (random horizontal/vertical flip, rotation, zoom
            and shift, combined into one affine transform per image, reflect fill, a new transformation every epoch).
            Applied after the RAM cache, so every epoch sees different versions of the same images. The normalization
            statistics, the validation, the test and test_Matthew.py stay on the original images (nothing to change
            at test time). Training metric: categorical cross-entropy instead of macro F1 (targets are one-hot).
            The run log (experiments_Matthew.csv) records the augmentation settings.
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
## 10/27/2026: neurons of every layer doubled -> [256, 256, 192, 192, 128, 128, 128, 64, 64, 64], ~31M parameters
HIDDEN_LAYERS = [256, 256, 192, 192, 128, 128, 128, 64, 64, 64]   # neurons of each hidden Dense layer
ACTIVATION = 'relu'

## ---- Regularization (set to 0 / False to switch off) -----------------------------------------------------------
DROPOUT_RATE = 0.2       # e.g. 0.2 - 0.5 ; used ONLY when USE_BATCHNORM = False (BatchNorm and Dropout are not combined)
USE_BATCHNORM = False    # True -> Dense -> BatchNorm -> activation (-> Dropout).  KEEP True with Adam alpha=0.001 (see Stability)
                         # 10/27/2026: False -> Dense (with bias) -> activation -> Dropout(DROPOUT_RATE)
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

## ---- Early stopping (book NND2 Ch.13 "Early Stopping", Fig. 13.4) --------------------------------------------
## The "train" rows are divided into a training set (used for the gradients) and a validation set (only monitored).
## After every epoch the validation error F_val is computed. Point "a" = the epoch with the smallest F_val.
## When F_val has not gone below that minimum for EARLY_STOP_PATIENCE epochs in a row, training stops (point "b")
## and the weights of point "a" are the final weights (saved to model_<NICKNAME>.keras).
## F = the performance index the network is trained on: cross-entropy, class-weighted when CLASS_WEIGHTS is on.
## F_val goes up and down from epoch to epoch. Book Ch.25 case study: "the validation error curve does not always fall
## at each iteration, and it may rise before falling to a lower value" (they waited 40 iterations) -> long patience.
##  EARLY_STOP_MONITOR: 'val_loss' (book: validation error F) | 'val_f1_macro' | 'val_kappa' | 'val_accuracy'
EARLY_STOP_MONITOR = 'val_loss'
EARLY_STOP_PATIENCE = 25     # epochs without a new minimum before stopping (0 = off: runs all n_epoch epochs)
EARLY_STOP_MIN_DELTA = 0.0   # book: any decrease of F_val is a new minimum

## Which validation value picks the weights saved to model_<NICKNAME>.keras
##  'val_loss' (book: point "a", minimum validation error) | 'val_f1_macro' | 'val_kappa' | 'val_accuracy'
## Middle ground: EARLY_STOP_MONITOR = 'val_loss' (when to stop) + CHECKPOINT_MONITOR = 'val_f1_macro' (which weights)
CHECKPOINT_MONITOR = 'val_loss'

## Training curve of Fig. 13.4: after every epoch F is also computed on the ORIGINAL training images in inference mode
## (no augmentation, BatchNorm with its moving averages, exactly like the validation images). The loss Keras prints is
## measured on the augmented images while the weights are changing (BatchNorm in training mode), so it cannot be
## compared with F_val. Costs one extra pass over the cached training images per epoch. Needs the RAM cache.
TRAIN_CURVE = True

## ---- Learning-rate schedule ----------------------------------------------------------------------------------
## Book: "we should use a relatively slow training method ... If the training method is too fast, it will likely jump
## past the point at which the validation error is minimized." Big steps at the start, smaller steps near point "a":
##  'plateau' : alpha = alpha * LR_FACTOR when the monitored value (EARLY_STOP_MONITOR) has not improved for
##              LR_PATIENCE epochs, never below LR_MIN
##  'none'    : alpha = LEARNING_RATE for the whole run (as before)
## Keep LR_PATIENCE well below EARLY_STOP_PATIENCE, so alpha is lowered a few times before the training is stopped.
LR_SCHEDULE = 'plateau'
LR_FACTOR = 0.5
LR_PATIENCE = 4
LR_MIN = 1e-5

## ---- Class weights -------------------------------------------------------------------------------------------
## The classes are unbalanced (validation: 676 class6 and 648 class3 images vs 166 class4 and 224 class1), so with the
## plain loss the big classes dominate the gradient. With class weights, the loss of every training image of class c
## is multiplied by w_c (computed on the training part only):
##  'balanced' : w_c = N / (K * n_c)   (N images, K classes, n_c images of class c) -> every class has the same total
##               weight; the average weight over the training images is exactly 1
##  'sqrt'     : square root of the balanced weights, rescaled to an average of 1 (softer)
##  'none'     : every image weighs 1 (as before)
## The validation error F_val and the training curve use the same weights, so F is the same function everywhere.
CLASS_WEIGHTS = 'balanced'

## ---- Test-time augmentation (TTA) ------------------------------------------------------------------------------
## The prediction of an image = average of the softmax outputs for the original image and its flipped versions
## (left-right, up-down, both = 180 degree rotation): 4 views. Only the flips the network saw in training (AUG_FLIP_H,
## AUG_FLIP_V with AUGMENT on) are used. They are written into the model name (e.g. MLP_Matthew_flip_hv_prep_r8_aa),
## so test_Matthew.py uses the same views. TTA is used for the final validation/test numbers and results_<NICKNAME>.xlsx
## (both are printed, with and without TTA); the epoch-by-epoch validation (early stopping) uses the original images.
TTA = True

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

## ---- Data augmentation (geometric, TRAINING images only) -------------------------------------------------------
## Every training image gets a random transformation each time it is used, so the network never sees exactly the same
## image twice. An MLP has one weight per pixel and no built-in invariance to position or orientation: augmentation
## teaches it that a leaf flipped, turned, shifted or seen a little closer is still the same class, and it works as a
## regularizer (last run without it: train macro F1 0.80 vs validation 0.46 at epoch 17 -> heavy overfitting).
## The leaves are photographed at any angle, so left/right AND up/down flips keep the label.
## Flips, rotation, zoom and shift are combined into ONE affine transform -> each image is interpolated only once.
## Validation, test and test_Matthew.py use the original images (only the flips come back at test time, see TTA).
AUGMENT = True           # master switch: False = original images only (as before)
AUG_FLIP_H = True        # random left-right flip (probability 0.5)
AUG_FLIP_V = True        # random up-down flip (probability 0.5)
AUG_ROTATION = 20.0      # max rotation in degrees, uniform in [-AUG_ROTATION, +AUG_ROTATION]      (0 = off)
AUG_ZOOM = 0.10          # max zoom, scale uniform in [1-AUG_ZOOM, 1+AUG_ZOOM]; >1 = closer          (0 = off)
AUG_SHIFT = 0.10         # max shift as a fraction of the image side, horizontal and vertical          (0 = off)
## pixels that come from outside the picture (corners after a rotation, borders after a shift or zoom-out):
##  'reflect'  : mirror of the image (leaf/foliage texture, no artificial black corners for the MLP to learn)
##  'nearest'  : repeat the closest edge pixel
##  'constant' : black (0)
AUG_FILL_MODE = 'reflect'
## save augmentation_preview_<NICKNAME>.png before training (rows: 4 training images, columns: original + 5 random
## versions) to check by eye that the transformations keep the leaf and its spots visible
AUG_PREVIEW = True

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
    if EARLY_STOP_MONITOR not in ('val_f1_macro', 'val_kappa', 'val_accuracy', 'val_loss'):
        raise ValueError('EARLY_STOP_MONITOR not recognised: {}'.format(EARLY_STOP_MONITOR))
    if LR_SCHEDULE not in ('plateau', 'none'):
        raise ValueError("LR_SCHEDULE must be 'plateau' or 'none', got {}".format(LR_SCHEDULE))
    if LR_SCHEDULE == 'plateau' and not 0.0 < LR_FACTOR < 1.0:
        raise ValueError('LR_FACTOR must be in (0, 1)')
    if CLASS_WEIGHTS not in ('balanced', 'sqrt', 'none'):
        raise ValueError("CLASS_WEIGHTS must be 'balanced', 'sqrt' or 'none', got {}".format(CLASS_WEIGHTS))
    if EARLY_STOP_PATIENCE > 0 and LR_SCHEDULE == 'plateau' and LR_PATIENCE >= EARLY_STOP_PATIENCE:
        print('WARNING: LR_PATIENCE >= EARLY_STOP_PATIENCE, the training stops before the learning rate is lowered')
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
    if AUG_FILL_MODE not in ('reflect', 'nearest', 'constant'):
        raise ValueError("AUG_FILL_MODE must be 'reflect', 'nearest' or 'constant', got {}".format(AUG_FILL_MODE))
    if not 0.0 <= AUG_ROTATION <= 180.0:
        raise ValueError('AUG_ROTATION must be in [0, 180] degrees')
    if not 0.0 <= AUG_ZOOM < 1.0:
        raise ValueError('AUG_ZOOM must be in [0, 1)')
    if not 0.0 <= AUG_SHIFT <= 0.5:
        raise ValueError('AUG_SHIFT must be in [0, 0.5]')
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

def read_data(df, num_classes, cache=None, shuffle=False, with_unshuffled=False):
    '''
          builds the tf.data pipeline for the rows in df
          only the file names are kept in memory, each image is read from disk when its batch is needed
          shuffle=False keeps the order of the rows in df (needed so predictions line up with the rows)
          shuffle=True  (training only) gives a new random order of the images at every epoch
          with_unshuffled=True also returns the same images in the order of df, reading the SAME RAM cache
          (no second copy, no shuffle buffer). None when there is no RAM cache.
    '''

    ds_inputs = (DATA_DIR + df['id']).tolist()
    ds_targets = get_target(df, num_classes)

    missing = [p for p in ds_inputs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('{} images not found, for example {} . Check the Data folder: {}'.format(
            len(missing), missing[:3], DATA_DIR))

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs, ds_targets)) # creates a tensor from the image paths and targets

    n = len(ds_inputs)
    unshuffled_ds = None

    if cache == 'memory':
        # decode once, keep in RAM, then reshuffle the cached images every epoch
        cached_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).cache()
        final_ds = cached_ds
        if shuffle:
            final_ds = final_ds.shuffle(n, seed=SEED, reshuffle_each_iteration=True)
        unshuffled_ds = cached_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    else:
        # shuffle only the file names (tiny) before the images are read
        if shuffle:
            list_ds = list_ds.shuffle(n, seed=SEED, reshuffle_each_iteration=True)
        final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    final_ds = final_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    if with_unshuffled:
        return final_ds, unshuffled_ds
    return final_ds
#------------------------------------------------------------------------------------------------------------------

def compute_class_weights(y_idx, num_classes):
    '''
        weight w_c of every class, from the class index of every TRAINING image (see CLASS_WEIGHTS)
        balanced: w_c = N / (K * n_c)  ->  sum over the images of w = N  (average weight 1)
    '''
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float64)
    n, k = counts.sum(), num_classes

    if CLASS_WEIGHTS == 'none':
        w = np.ones(k)
    else:
        w = n / (k * np.maximum(counts, 1.0))
        if CLASS_WEIGHTS == 'sqrt':
            w = np.sqrt(w)
            w = w * n / np.sum(counts * w)      # rescaled so the average weight over the images is again 1

    print('Class weights ({}):'.format(CLASS_WEIGHTS))
    for c in range(k):
        print('  {:10s} {:5d} training images   w = {:.3f}'.format(str(class_names[c]), int(counts[c]), w[c]))

    return w.astype(np.float32)
#------------------------------------------------------------------------------------------------------------------

def add_sample_weights(ds, class_w):
    '''
        (x, y) -> (x, y, w): w = weight of the class of every image (y is one-hot, so w = sum(y * class_w))
        Keras multiplies the loss of every image by its w:  F = sum_q w_q * ce_q / Q
    '''
    w_vec = tf.constant(class_w, tf.float32)
    return ds.map(lambda x, y: (x, y, tf.reduce_sum(y * w_vec, axis=-1)), num_parallel_calls=AUTOTUNE)
#------------------------------------------------------------------------------------------------------------------

def weighted_ce(prob, y_idx, class_w):
    '''
        performance index F on a set of images: class-weighted average cross-entropy
            F = sum_q w_q * (-log a_q[t_q]) / sum_q w_q          (w_q = 1 for all images -> plain cross-entropy)
        returns (F, plain cross-entropy)
    '''
    p_true = np.clip(prob[np.arange(len(y_idx)), y_idx], 1e-7, 1.0)
    ce = -np.log(p_true)
    return float(np.average(ce, weights=class_w[y_idx])), float(np.mean(ce))
#------------------------------------------------------------------------------------------------------------------

def trained_flips():
    '''
        flips the network sees in training: '' | 'h' | 'v' | 'hv'   (h = left-right, v = up-down)
    '''
    if not AUGMENT:
        return ''
    return ('h' if AUG_FLIP_H else '') + ('v' if AUG_FLIP_V else '')
#------------------------------------------------------------------------------------------------------------------

def tta_views(flips):
    '''
        views used by test-time augmentation: list of (flip left-right, flip up-down), the original image first
        'hv' -> original, left-right, up-down, both (= 180 degree rotation)
    '''
    views = [(False, False)]
    if 'h' in flips:
        views.append((True, False))
    if 'v' in flips:
        views.append((False, True))
    if 'h' in flips and 'v' in flips:
        views.append((True, True))
    return views
#------------------------------------------------------------------------------------------------------------------

def flip_flat(x, flip_lr, flip_ud):
    '''
        flips a batch of flat images [batch, IMAGE_SIZE*IMAGE_SIZE*CHANNELS] (exact pixel mirror, no interpolation)
    '''
    axes = ([2] if flip_lr else []) + ([1] if flip_ud else [])
    if not axes:
        return x
    img = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
    img = tf.reverse(img, axis=axes)
    return tf.reshape(img, [-1, IMAGE_SIZE * IMAGE_SIZE * CHANNELS])
#------------------------------------------------------------------------------------------------------------------

def predict_proba(final_model, ds, tta=False):
    '''
        softmax output of every image in ds (ds gives (x, y) batches, in the order of the rows)
        tta=True: average of the softmax over the TTA views (same code as test_Matthew.py)
    '''
    views = tta_views(trained_flips()) if tta else [(False, False)]
    if len(views) == 1:
        return final_model.predict(ds, verbose=0)

    out = []
    for xb, _ in ds:
        p = 0.0
        for flip_lr, flip_ud in views:
            p = p + np.asarray(final_model.predict_on_batch(flip_flat(xb, flip_lr, flip_ud)))
        out.append(p / len(views))
    return np.concatenate(out, axis=0)
#------------------------------------------------------------------------------------------------------------------

def augment_batch(x, seed):
    '''
        random geometric transformation of a batch of flat images (training only)
        x    : [batch, IMAGE_SIZE*IMAGE_SIZE*CHANNELS] pixels in [0,1], as produced by process_path
        seed : 2 integers. Stateless random numbers: the same seed always gives the same transformations

        every image gets its own flips, rotation theta, zoom s and shift t. The image content is moved as
              p = c + s * R(theta) * F * (q - c) + t        (q = input pixel, p = output pixel, c = image center)
        ImageProjectiveTransformV3 needs the opposite direction (for every output pixel, where to read it from):
              q = A (p - c - t) + c ,   A = F * R(-theta) / s
        and samples the input there with bilinear interpolation (one interpolation for all the transformations).
    '''
    size = IMAGE_SIZE
    b = tf.shape(x)[0]
    img = tf.reshape(x, [b, size, size, CHANNELS])

    u = tf.random.stateless_uniform([b, 6], seed=seed)      # 6 independent numbers in [0,1) for every image

    ones = tf.ones_like(u[:, 0])
    fx = tf.where(u[:, 0] < 0.5, -1.0, 1.0) if AUG_FLIP_H else ones     # -1 = left-right flip
    fy = tf.where(u[:, 1] < 0.5, -1.0, 1.0) if AUG_FLIP_V else ones     # -1 = up-down flip
    theta = (2.0 * u[:, 2] - 1.0) * (AUG_ROTATION * np.pi / 180.0)      # radians
    s = 1.0 + (2.0 * u[:, 3] - 1.0) * AUG_ZOOM                          # scale, > 1 = zoom in
    tx = (2.0 * u[:, 4] - 1.0) * (AUG_SHIFT * size)                     # shift in pixels (columns)
    ty = (2.0 * u[:, 5] - 1.0) * (AUG_SHIFT * size)                     # shift in pixels (rows)

    c = (size - 1) / 2.0
    cos, sin = tf.cos(theta), tf.sin(theta)
    a00, a01 = fx * cos / s, fx * sin / s
    a10, a11 = -fy * sin / s, fy * cos / s
    b0 = c - a00 * (c + tx) - a01 * (c + ty)
    b1 = c - a10 * (c + tx) - a11 * (c + ty)
    zeros = tf.zeros_like(a00)
    # one row per image: [a0, a1, a2, b0, b1, b2, c0, c1]  (c0 = c1 = 0: affine, no perspective)
    transforms = tf.stack([a00, a01, b0, a10, a11, b1, zeros, zeros], axis=1)

    out = tf.raw_ops.ImageProjectiveTransformV3(images=img, transforms=transforms,
                                                output_shape=tf.constant([size, size], tf.int32),
                                                fill_value=tf.constant(0.0, tf.float32),
                                                interpolation='BILINEAR', fill_mode=AUG_FILL_MODE.upper())

    return tf.reshape(out, [b, size * size * CHANNELS])
#------------------------------------------------------------------------------------------------------------------

def augment_dataset(ds):
    '''
        adds the random augmentation to the (already shuffled and batched) training pipeline from read_data.
        It comes AFTER the RAM cache, so the cache keeps the original images and each epoch draws new transformations.
        ds itself is not changed: it keeps giving the original images (used for the normalization statistics),
        and both share the same cache (the images are decoded only once).

        Random numbers: one seed per batch from tf.data.Dataset.random(SEED) -> repeatable runs, and
        rerandomize_each_iteration=True gives a different sequence at every epoch.
    '''
    try:
        seeds = tf.data.Dataset.random(seed=SEED, rerandomize_each_iteration=True)
    except TypeError:
        # older TensorFlow: same seed sequence every epoch, but the batches are reshuffled, so each image still gets a
        # different transformation every epoch
        seeds = tf.data.Dataset.random(seed=SEED)

    def augment(xy, r):
        x, y = xy
        return augment_batch(x, tf.stack([r, tf.constant(SEED, tf.int64)])), y

    return tf.data.Dataset.zip((ds, seeds)).map(augment, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
#------------------------------------------------------------------------------------------------------------------

def augmentation_text():
    if not AUGMENT:
        return 'off'
    parts = (['flipH'] if AUG_FLIP_H else []) + (['flipV'] if AUG_FLIP_V else [])
    parts += ['rot{:g}'.format(AUG_ROTATION), 'zoom{:g}'.format(AUG_ZOOM), 'shift{:g}'.format(AUG_SHIFT), AUG_FILL_MODE]
    return ' '.join(parts)
#------------------------------------------------------------------------------------------------------------------

def save_augmentation_preview(df, n_images=4, n_versions=5):
    '''
        writes augmentation_preview_<NICKNAME>.png: one row per training image, first column the original,
        then n_versions random augmented versions. Only n_images files are decoded (the cache is not touched).
    '''
    files = (DATA_DIR + df['id'].iloc[:n_images]).tolist()
    clean = tf.stack([process_path(tf.constant(f), 0)[0] for f in files])

    columns = [clean] + [augment_batch(clean, tf.constant([k + 1, SEED], tf.int64)) for k in range(n_versions)]

    def column_image(flat):
        imgs = tf.reshape(flat, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
        imgs = tf.pad(imgs, [[0, 0], [3, 3], [3, 3], [0, 0]], constant_values=1.0)   # white frame
        return tf.concat(tf.unstack(imgs), axis=0)                                   # images stacked vertically

    grid = tf.concat([column_image(col) for col in columns], axis=1)
    png = tf.io.encode_png(tf.cast(tf.round(tf.clip_by_value(grid, 0.0, 1.0) * 255.0), tf.uint8))

    file_name = out_file('augmentation_preview_{}.png'.format(NICKNAME))
    tf.io.write_file(file_name, png)
    print('Augmentation preview saved to', file_name)
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
    # _flip_hv: flips seen in training, test_Matthew.py uses them for test-time augmentation
    # _prep_r8_aa: JPEG decode ratio and antialias, test_Matthew.py prepares the images the same way (kept at the end)
    flips = trained_flips()
    return 'MLP_{}{}_prep_r{}{}'.format(NICKNAME, '_flip_' + flips if flips else '',
                                        JPEG_DECODE_RATIO, '_aa' if ANTIALIAS else '')
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

    # metric shown during training: categorical cross-entropy (one-hot targets, softmax output, 6 classes).
    # Metrics are only reported, the weights are trained on the loss. With WEIGHT_DECAY = 0 it equals the loss;
    # with WEIGHT_DECAY > 0 the loss also includes the L2 penalty and this metric shows the cross-entropy alone.
    # With CLASS_WEIGHTS on, both are class-weighted. With AUGMENT on, they are measured on the augmented images.
    # F on the original training images, F_val, macro F1, kappa and accuracy are computed every epoch by
    # ValidationMetrics and decide the learning rate, the stop and the saved epoch.
    model.compile(optimizer=make_optimizer(), loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.CategoricalCrossentropy(name='categorical_ce')])

    # shown on screen only; summary_<NICKNAME>.txt is written once after training, from the saved model
    model.summary()
    return model
#------------------------------------------------------------------------------------------------------------------

class ValidationMetrics(tf.keras.callbacks.Callback):
    '''
        at the end of every epoch adds to the epoch log (ReduceLROnPlateau / ModelCheckpoint / EarlyStopping, placed
        after this callback, can then monitor any of them):
          lr           : learning rate alpha used during this epoch
          val_loss     : validation error F_val (cross-entropy, class-weighted when CLASS_WEIGHTS is on)
          val_ce       : plain cross-entropy on the validation images
          val_accuracy, val_f1_macro, val_kappa
          train_F, train_accuracy : F and accuracy on the ORIGINAL training images, inference mode (TRAIN_CURVE)
    '''
    def __init__(self, val_ds, y_val, class_w, train_eval_ds=None):
        super().__init__()
        self.val_ds = val_ds
        self.y_val = y_val                  # integer class index of every validation image
        self.class_w = class_w              # weight of every class
        self.train_eval_ds = train_eval_ds  # original training images (x, one-hot y), None = no training curve

    def training_error(self):
        '''
            F and accuracy over the original training images, with the weights at the end of this epoch
        '''
        f_sum = w_sum = correct = n = 0.0
        for xb, yb in self.train_eval_ds:
            prob = np.asarray(self.model.predict_on_batch(xb))
            y_idx = np.argmax(yb.numpy(), axis=1)
            w = self.class_w[y_idx]
            p_true = np.clip(prob[np.arange(len(y_idx)), y_idx], 1e-7, 1.0)
            f_sum += float(np.sum(w * -np.log(p_true)))
            w_sum += float(np.sum(w))
            correct += float(np.sum(np.argmax(prob, axis=1) == y_idx))
            n += len(y_idx)
        return f_sum / w_sum, correct / n

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        logs['lr'] = float(np.asarray(self.model.optimizer.learning_rate))
        prob = self.model.predict(self.val_ds, verbose=0)

        if not np.all(np.isfinite(prob)):
            # the weights exploded: report the worst possible values so this epoch is never saved as the best one
            logs.update({'val_loss': np.inf, 'val_ce': np.inf, 'val_accuracy': -1.0, 'val_f1_macro': -1.0,
                         'val_kappa': -1.0, 'train_F': np.nan, 'train_accuracy': np.nan})
            print('  epoch {:3d} | NaN/inf in the predictions, this epoch is discarded'.format(epoch + 1))
            return

        pred = np.argmax(prob, axis=1)

        logs['val_loss'], logs['val_ce'] = weighted_ce(prob, self.y_val, self.class_w)
        logs['val_accuracy'] = float(accuracy_score(self.y_val, pred))
        logs['val_f1_macro'] = float(f1_score(self.y_val, pred, average='macro'))
        logs['val_kappa'] = float(cohen_kappa_score(self.y_val, pred))

        if self.train_eval_ds is not None:
            logs['train_F'], logs['train_accuracy'] = self.training_error()
        else:
            logs['train_F'], logs['train_accuracy'] = np.nan, np.nan

        print('  epoch {:3d} | lr {:.2e} | train loss (augmented) {:.4f} | train F {:.4f} acc {:.4f} | '
              'val F {:.4f} acc {:.4f} f1_macro {:.4f} kappa {:.4f}'.format(
            epoch + 1, logs['lr'], logs.get('loss', np.nan), logs['train_F'], logs['train_accuracy'],
            logs['val_loss'], logs['val_accuracy'], logs['val_f1_macro'], logs['val_kappa']))
#------------------------------------------------------------------------------------------------------------------

def monitor_mode(monitor):
    # validation error: lower is better; F1 / kappa / accuracy: higher is better
    return 'min' if monitor == 'val_loss' else 'max'
#------------------------------------------------------------------------------------------------------------------

def train_func(train_ds, val_ds, y_val, stats, class_w, train_eval_ds=None):
    '''
        train the model with early stopping (book NND2 Ch.13), keep the weights of the best validation epoch
    '''

    model_file = out_file('model_{}.keras'.format(NICKNAME))

    # the order matters: ValidationMetrics adds lr / val_* / train_* to the epoch log, the callbacks after it read them
    callbacks = [tf.keras.callbacks.TerminateOnNaN(),   # stops at once if the loss explodes to NaN / inf
                 ValidationMetrics(val_ds, y_val, class_w, train_eval_ds)]
    # weights of the best epoch (point "a" with CHECKPOINT_MONITOR = 'val_loss') -> model_<NICKNAME>.keras
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(model_file, monitor=CHECKPOINT_MONITOR,
                                                        mode=monitor_mode(CHECKPOINT_MONITOR),
                                                        save_best_only=True, verbose=1))
    if LR_SCHEDULE == 'plateau':
        # alpha <- alpha * LR_FACTOR after LR_PATIENCE epochs without a new best value of EARLY_STOP_MONITOR
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=EARLY_STOP_MONITOR,
                                                              mode=monitor_mode(EARLY_STOP_MONITOR),
                                                              factor=LR_FACTOR, patience=LR_PATIENCE,
                                                              min_delta=0.0, min_lr=LR_MIN, verbose=1))
    if EARLY_STOP_PATIENCE > 0:
        # stop after EARLY_STOP_PATIENCE epochs without a new minimum of F_val (point "b"), back to point "a"
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=EARLY_STOP_MONITOR,
                                                          mode=monitor_mode(EARLY_STOP_MONITOR),
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

def early_stopping_report(history):
    '''
        book NND2 Fig. 13.4 for this run:
          point "a" = epoch with the minimum validation error F_val   (the weights kept by early stopping)
          point "b" = last epoch trained (where the training was stopped)
        writes early_stopping_<NICKNAME>.csv (one row per epoch) and early_stopping_<NICKNAME>.png
        returns {'a_epoch', 'a_val_F', 'b_epoch', 'saved_epoch'}
    '''
    h = history.history
    val_F = np.asarray(h.get('val_loss', []), dtype=float)
    n = len(val_F)
    if n == 0 or not np.any(np.isfinite(val_F)):
        print('Early stopping report skipped: no valid validation error')
        return {}

    def column(key):
        v = list(h.get(key, []))[:n]
        return np.asarray(v + [np.nan] * (n - len(v)), dtype=float)

    epochs = np.arange(1, n + 1)
    a = int(np.nanargmin(np.where(np.isfinite(val_F), val_F, np.nan))) + 1
    b = n
    monitored = column(CHECKPOINT_MONITOR)
    saved = int(np.nanargmin(monitored) if monitor_mode(CHECKPOINT_MONITOR) == 'min' else np.nanargmax(monitored)) + 1
    info = {'a_epoch': a, 'a_val_F': float(val_F[a - 1]), 'b_epoch': b, 'saved_epoch': saved}

    print('\n================ EARLY STOPPING (book NND2 Fig. 13.4) ================')
    print('point a = epoch {} : minimum validation error F_val = {:.4f} (val macro F1 {:.4f})'.format(
        a, val_F[a - 1], column('val_f1_macro')[a - 1]))
    print('point b = epoch {} : training stopped, F_val = {:.4f}'.format(b, val_F[b - 1]))
    print('weights saved to model_{}.keras: epoch {} (best {})'.format(NICKNAME, saved, CHECKPOINT_MONITOR))

    table = pd.DataFrame({'epoch': epochs, 'lr': column('lr'),
                          'train_loss_keras_augmented': column('loss'),
                          'train_F_original_images': column('train_F'), 'train_accuracy': column('train_accuracy'),
                          'val_F': val_F, 'val_ce': column('val_ce'), 'val_accuracy': column('val_accuracy'),
                          'val_f1_macro': column('val_f1_macro'), 'val_kappa': column('val_kappa')})
    table.to_csv(out_file('early_stopping_{}.csv'.format(NICKNAME)), index=False)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib is not installed: early_stopping_{}.png skipped (the numbers are in the .csv)'.format(NICKNAME))
        return info

    # three panels side by side, one measure each (no second y-axis): F as in Fig. 13.4, validation macro F1, alpha
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={'width_ratios': [1.5, 1, 1]})

    ax = axes[0]
    ax.plot(epochs, val_F, color='#1f3fd1', lw=2, label='Validation')
    if np.any(np.isfinite(table['train_F_original_images'])):
        ax.plot(epochs, table['train_F_original_images'], color='black', lw=2, label='Training (original images)')
    ax.plot(epochs, table['train_loss_keras_augmented'], color='0.55', lw=1.2, ls='--',
            label='Training, as printed by Keras (augmented)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('F (cross-entropy{})'.format(', class-weighted' if CLASS_WEIGHTS != 'none' else ''))
    ax.set_title('Early stopping (NND2 Fig. 13.4): a = epoch {}, b = epoch {}'.format(a, b))
    ax.legend(frameon=False, loc='best')

    ax = axes[1]
    ax.plot(epochs, table['val_f1_macro'], color='#1f3fd1', lw=2)
    ax.set_xscale('log')
    ax.set_ylabel('validation macro F1')
    ax.set_title('Validation macro F1')

    ax = axes[2]
    ax.step(epochs, table['lr'], where='post', color='black', lw=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('learning rate alpha')
    ax.set_title('Learning rate ({})'.format(LR_SCHEDULE))

    for ax in axes:
        ax.set_title(ax.get_title(), pad=24)    # room for the a / b labels above the plot
        ax.set_xlabel('Epoch (iteration)')
        ax.grid(True, which='major', color='0.9', lw=0.8)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        # a = b when F_val was still going down at the last epoch (no overfitting seen yet: try more epochs)
        for x, name in (((a, 'a'), (b, 'b')) if a != b else ((a, 'a = b'),)):
            ax.axvline(x, color='0.35', ls=':', lw=1.2)
            ax.text(x, 1.01, name, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=13)

    fig.tight_layout()
    file_name = out_file('early_stopping_{}.png'.format(NICKNAME))
    fig.savefig(file_name, dpi=130)
    plt.close(fig)
    print('Early stopping plot saved to', file_name)
    return info
#------------------------------------------------------------------------------------------------------------------

def predict_func(pred):
    '''
        predict function: writes results_<NICKNAME>.xlsx (same format as the template)
        pred = predicted class index of every row of xdf_dset
    '''

    if len(pred) != len(xdf_dset):
        raise ValueError('{} predictions for {} rows'.format(len(pred), len(xdf_dset)))

    xdf_dset['results'] = pred
    xdf_dset.to_excel(out_file('results_{}.xlsx'.format(NICKNAME)), index=False)
#------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=None, confusion=True):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_weighted, coh, acc, mat
    list of aggregates : avg, sum
    confusion: also print the confusion matrix
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

    if confusion:
        # rows = true class, columns = predicted class
        print('Confusion matrix (rows = true, columns = predicted), classes:', list(class_names))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        print(cm)
        # recall of every class = correct / images of that class (diagonal / row sum)
        recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        print('Recall per class:', ' '.join('{} {:.2f}'.format(c, r) for c, r in zip(class_names, recall)))

    return values
#------------------------------------------------------------------------------------------------------------------

def dropout_used():
    # Dropout is not combined with BatchNorm
    return DROPOUT_RATE if not USE_BATCHNORM else 0.0

def lr_schedule_text():
    if LR_SCHEDULE == 'none':
        return 'none'
    return 'plateau x{:g} after {} ep, min {:g}'.format(LR_FACTOR, LR_PATIENCE, LR_MIN)

def tta_text():
    return '{} views'.format(len(tta_views(trained_flips()))) if TTA else 'off'

def settings_text():
    return ('NORM_MODE={} | HIDDEN={} | DROPOUT={} | BATCHNORM={} | WEIGHT_DECAY={} | OPTIMIZER={} | LR={} | '
            'LR_SCHEDULE={} | INIT={} | CLIP={} | EPOCHS={} | EARLY_STOP={} {}/{} | SHUFFLE={} | BATCH={} | '
            'IMAGE_SIZE={} | JPEG_RATIO={} | ANTIALIAS={} | AUGMENT={} | CLASS_WEIGHTS={} | TTA={} | VAL_FRACTION={} | '
            'CHECKPOINT={} | SEED={}').format(
        NORM_MODE, HIDDEN_LAYERS, dropout_used(), USE_BATCHNORM, WEIGHT_DECAY, OPTIMIZER, LEARNING_RATE,
        lr_schedule_text(), KERNEL_INIT, CLIP_NORM, n_epoch, EARLY_STOP_MONITOR, EARLY_STOP_PATIENCE,
        EARLY_STOP_MIN_DELTA, SHUFFLE_EACH_EPOCH, BATCH_SIZE, IMAGE_SIZE, JPEG_DECODE_RATIO, ANTIALIAS,
        augmentation_text(), CLASS_WEIGHTS, tta_text(), VAL_FRACTION, CHECKPOINT_MONITOR, SEED)
#------------------------------------------------------------------------------------------------------------------

def log_experiment(history, val_values, test_values, es_info, val_plain, test_plain):
    '''
        appends one line per run to experiments_<NICKNAME>.csv so different combinations can be compared
        (an extra file; the official output files are not changed)
        val_values / test_values: final numbers (with TTA when TTA is on); val_plain / test_plain: without TTA
    '''
    lrs = history.history.get('lr', [])

    row = {'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
           'norm': NORM_MODE, 'hidden': str(HIDDEN_LAYERS), 'dropout': dropout_used(),
           'batchnorm': USE_BATCHNORM, 'weight_decay': WEIGHT_DECAY,
           'optimizer': OPTIMIZER, 'lr': LEARNING_RATE, 'lr_schedule': lr_schedule_text(),
           'final_lr': float(lrs[-1]) if len(lrs) else '',
           'init': KERNEL_INIT, 'clip': CLIP_NORM, 'shuffle': SHUFFLE_EACH_EPOCH,
           'stop_monitor': EARLY_STOP_MONITOR, 'patience': EARLY_STOP_PATIENCE,
           'epochs_run': len(history.history.get('loss', [])), 'best_epoch': es_info.get('saved_epoch', ''),
           'a_epoch': es_info.get('a_epoch', ''), 'a_val_F': es_info.get('a_val_F', ''),
           'batch': BATCH_SIZE, 'image_size': IMAGE_SIZE, 'jpeg_ratio': JPEG_DECODE_RATIO, 'antialias': ANTIALIAS,
           'augment': augmentation_text(), 'class_weights': CLASS_WEIGHTS, 'tta': tta_text(),
           'monitor': CHECKPOINT_MONITOR, 'seed': SEED,
           'val_f1_macro': val_values.get('f1_macro'), 'val_kappa': val_values.get('coh'),
           'val_acc': val_values.get('acc'),
           'test_f1_macro': test_values.get('f1_macro'), 'test_kappa': test_values.get('coh'),
           'test_acc': test_values.get('acc'),
           'val_f1_macro_no_tta': val_plain.get('f1_macro'), 'test_f1_macro_no_tta': test_plain.get('f1_macro')}

    file_name = out_file('experiments_{}.csv'.format(NICKNAME))
    new_file = not os.path.exists(file_name)

    if not new_file:
        # a file written by an older version of this script has other columns: rewrite it with all the columns
        # (old runs get an empty cell in the new ones) so the rows never shift under the wrong header
        old = pd.read_csv(file_name)
        if list(old.columns) != list(row.keys()):
            pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(file_name, index=False)
            print('Run logged in', file_name, '(columns updated)')
            return

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
    # train_clean_ds: original images, reshuffled every epoch; train_eval_ds: the same cached images in a fixed order
    train_clean_ds, train_eval_ds = read_data(xdf_tr, OUTPUTS_a, cache=cache, shuffle=SHUFFLE_EACH_EPOCH,
                                              with_unshuffled=True)
    val_ds = read_data(xdf_val, OUTPUTS_a, cache=cache)
    y_val = xdf_val['target'].map(class_to_idx).to_numpy()

    if TRAIN_CURVE and train_eval_ds is None:
        print('TRAIN_CURVE: no RAM cache, the training error on the original images is skipped '
              '(reading all training images from disk every epoch would be too slow)')
    if not TRAIN_CURVE:
        train_eval_ds = None

    ## Class weights from the training part only
    class_w = compute_class_weights(xdf_tr['target'].map(class_to_idx).to_numpy(), OUTPUTS_a)

    ## Data augmentation: training images only, added after the cache (validation and test stay original)
    if AUGMENT:
        train_ds = augment_dataset(train_clean_ds)
        if AUG_PREVIEW:
            save_augmentation_preview(xdf_tr)
    else:
        train_ds = train_clean_ds

    ## the loss of every training image is multiplied by the weight of its class
    if CLASS_WEIGHTS != 'none':
        train_ds = add_sample_weights(train_ds, class_w)

    ## Normalization statistics from the training part only, on the ORIGINAL images (the test images are not
    ## augmented, so the input layer must be centred and scaled for them). This pass also fills the RAM cache
    ## (read in a fixed order when the cache is on: no shuffle buffer to fill while the JPEGs are decoded).
    stats = None
    if NORM_MODE in ('minmax', 'zscore'):
        stats = compute_input_stats(train_eval_ds if train_eval_ds is not None else train_clean_ds)

    # best_model = the best saved epoch, loaded once from model_<NICKNAME>.keras for validation and test
    history, best_model = train_func(train_ds, val_ds, y_val, stats, class_w, train_eval_ds)

    ## Point "a", point "b", early_stopping_<NICKNAME>.png / .csv
    es_info = early_stopping_report(history)

    use_tta = TTA and len(tta_views(trained_flips())) > 1
    tta_label = 'TTA, {} views: original + flips {}'.format(len(tta_views(trained_flips())), trained_flips())

    ## Metrics of the best saved model on the validation part (without TTA, then with TTA)

    print('\n================ VALIDATION (best saved epoch) ================')
    xdf_dset = xdf_val.copy()
    xdf_dset['results'] = np.argmax(predict_proba(best_model, val_ds), axis=1)
    if use_tta:
        print('--- without TTA ---')
    val_plain = metrics_func(['f1_macro', 'coh', 'acc'], ['avg'], confusion=not use_tta)
    val_values = val_plain
    if use_tta:
        print('--- with {} ---'.format(tta_label))
        xdf_dset['results'] = np.argmax(predict_proba(best_model, val_ds, tta=True), axis=1)
        val_values = metrics_func(['f1_macro', 'coh', 'acc'], ['avg'])

    ## Test dataset -> results_<NICKNAME>.xlsx (with TTA when TTA is on, like test_Matthew.py)

    print('\n================ TEST ================')
    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()
    if len(xdf_dset) == 0:
        raise ValueError('No rows with split == "test" in {}'.format(FILE_NAME))

    # with TTA the test images are predicted twice (without / with TTA): with the RAM cache they are decoded only once
    # (~1.2 GB in RAM during this step only)
    test_ds= read_data(xdf_dset, OUTPUTS_a, cache='memory' if (use_tta and cache == 'memory') else None)

    ## Metrics Function over the result of the test dataset
    list_of_metrics = ['f1_macro', 'coh', 'acc']
    list_of_agg = ['avg']

    if use_tta:
        print('--- without TTA ---')
    predict_func(np.argmax(predict_proba(best_model, test_ds), axis=1))
    test_plain = metrics_func(list_of_metrics, list_of_agg, confusion=not use_tta)
    test_values = test_plain
    if use_tta:
        print('--- with {} ---'.format(tta_label))
        predict_func(np.argmax(predict_proba(best_model, test_ds, tta=True), axis=1))
        test_values = metrics_func(list_of_metrics, list_of_agg)

    print('\nSETTINGS:', settings_text(), '| CACHE={}'.format(cache))
    log_experiment(history, val_values, test_values, es_info, val_plain, test_plain)
# ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    main()
#------------------------------------------------------------------------------------------------------------------
