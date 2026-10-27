#------------------------------------------------------------------------------------------------------------------
import os
import csv
import time
import argparse
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef, confusion_matrix
from tensorflow.keras.utils import to_categorical

#------------------------------------------------------------------------------------------------------------------

'''
Based on train_example.py (LSDR) and train_Matthew.py
10/27/2026 pm: FEATURE MLP (the previous pixel version is in the git history of this file)
            - INPUT: instead of the 120,000 pixels, every image is described by colour and texture histograms of the
              whole leaf (hue x saturation, brightness, colour moments, excess red/green, local binary patterns at
              3 scales, edge strength, the same for the centre of the image). A histogram counts a lesion the same
              wherever it is, which a Dense layer cannot learn from pixels (pixel run: train acc 0.59, val 0.55).
              Tested on 240 training images, same classifier: raw pixels (F1+kappa)/2 = 0.35-0.38, features 0.60.
              Pre-processing only: the model is still Input -> Normalization -> Dense layers -> softmax.
              Ideas: colour histograms (Swain & Ballard, 1991), local binary patterns (Ojala et al., 2002),
              square-root (Hellinger) normalization of histograms (Arandjelovic & Zisserman, 2012).
            - FEATURE CACHE: features of all 14,685 images are computed once (tf.data, all CPU cores in parallel)
              and saved to features_cache/. Every later run starts training in seconds.
            - BIGGER NETWORK: 1,000-2,000 inputs instead of 120,000 -> wide layers are cheap
              ([1024, 512, 256] = 2.4M weights vs 15.4M before).
            - GRID: every combination of the GRID settings is trained on the same train/validation split and
              ranked by the validation (F1 + kappa)/2 = the leaderboard formula (checked: 10262026 model, local test
              0.4201 vs leaderboard 0.4200). Test numbers are printed but never used to choose.
            - ENSEMBLE: the best combination is trained FINAL_FOLDS times (k-fold: every fold is the validation set
              once) -> model_Matthew1..k.keras; test_Matthew.py averages their softmax outputs (Exam1 section 4.8).
              Out-of-fold predictions give a score on all 12,093 "train" images.
            - selection of the best epoch on val (F1 + kappa)/2, stop on the validation error F (book NND2 Ch.13)
10/26/2026 pm: early stopping as in NND2 Fig. 13.4, LR plateau schedule, class weights, TTA (pixel version)
10/26/2026: geometric data augmentation (pixel version)
10/25/2026: Adam, BatchNorm, Kaiming init, gradient clipping, normalization inside the model, RAM cache
10/24/2026: validation split from the "train" rows, best epoch on validation, metrics on validation and test
'''

#==================================================================================================================
#  EXPERIMENT SETTINGS  -  change these and re-run
#==================================================================================================================

NICKNAME = 'Matthew'

## ---- Features (see the FEATURE EXTRACTION block below) ------------------------------------------------------------
##  'hs'  hue x saturation histogram (288)        'v'    brightness histogram (16)      'mom'  colour moments (12)
##  'exr' excess red / excess green (64)          'lbp'  texture, full size (51)        'lbp2' texture, 1/2 size (51)
##  'lbp4' texture, 1/4 size (51)                 'grad' edge strength (16)             'ctr'  hs + v + lbp of the centre (355)
##  'thumb' 16x16 colour thumbnail (768)
## 240-image test: all groups but 'thumb' was the best set (thumbnail = raw pixels again, it lowered the score)
FEATURE_GROUPS = ['hs', 'v', 'mom', 'exr', 'lbp', 'lbp2', 'lbp4', 'grad', 'ctr']
FEATURE_RATIO = 8            # JPEG decoded at 1/8 size (500x334). Ratio 4 (1000x668) was not better and is 3x slower
EXTRACT_PARALLEL = 0         # images processed at the same time: 0 = automatic (all CPU cores), n = n images
EXTRACT_BATCH = 64
CACHE_DIR = 'features_cache' # folder next to this file; delete it to recompute the features
## a feature that (almost) never changes in the training set would be divided by ~0 in the z-score and explode on a new
## image: its standard deviation is floored at MIN_FEATURE_STD (median std of the features is ~0.01)
MIN_FEATURE_STD = 0.01

## ---- Network ---------------------------------------------------------------------------------------------------
HIDDEN_LAYERS = [1024, 512, 256]   # neurons of each hidden Dense layer
ACTIVATION = 'relu'
USE_BATCHNORM = True               # Dense -> BatchNorm -> activation -> Dropout
DROPOUT_RATE = 0.3                 # after every hidden layer (0 = off). Works together with BatchNorm here
INPUT_DROPOUT = 0.0                # dropout on the normalized features themselves (0 = off)
WEIGHT_DECAY = 1e-4                # L2 penalty on every Dense kernel (0 = off)
KERNEL_INIT = 'he_normal'          # Kaiming init for ReLU (book Eq. 4.34)
LABEL_SMOOTHING = 0.0              # e.g. 0.1: targets 0.9 / 0.02 instead of 1 / 0 (0 = off)

## ---- Training --------------------------------------------------------------------------------------------------
n_epoch = 300                # upper limit, early stopping ends the run before
BATCH_SIZE = 64
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'           # 'adam' | 'rmsprop' | 'sgd'
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-8
SGD_MOMENTUM = 0.9
CLIP_NORM = 1.0              # global gradient clipping (0 = off)
CLASS_WEIGHTS = 'balanced'   # 'balanced' | 'sqrt' | 'none'
VAL_FRACTION = 0.20          # validation part of the "train" rows for the GRID runs
SEED = 42

## Early stopping (NND2 Ch.13): stop when the validation error F has not improved for EARLY_STOP_PATIENCE epochs.
## The weights kept are those of the epoch with the best CHECKPOINT_MONITOR.
##  'val_score' = (macro F1 + Cohen kappa) / 2 on validation = leaderboard formula
EARLY_STOP_MONITOR = 'val_loss'
EARLY_STOP_PATIENCE = 30
CHECKPOINT_MONITOR = 'val_score'   # 'val_score' | 'val_loss' | 'val_f1_macro' | 'val_kappa' | 'val_accuracy'
LR_SCHEDULE = 'plateau'            # alpha x LR_FACTOR after LR_PATIENCE epochs without a better val_loss
LR_FACTOR = 0.5
LR_PATIENCE = 8
LR_MIN = 1e-6
PRINT_EVERY = 10                   # print one line every PRINT_EVERY epochs (and the last one)

## ---- Grid of experiments -------------------------------------------------------------------------------------
## Every combination is trained on the same train/validation split (VAL_FRACTION, SEED) and logged to
## grid_<NICKNAME>.csv and experiments_<NICKNAME>.csv. The best mean validation (F1 + kappa)/2 wins.
## Allowed keys: the setting names above (HIDDEN_LAYERS, DROPOUT_RATE, INPUT_DROPOUT, WEIGHT_DECAY, USE_BATCHNORM,
## LEARNING_RATE, BATCH_SIZE, CLASS_WEIGHTS, LABEL_SMOOTHING, FEATURE_GROUPS, ACTIVATION, OPTIMIZER).
## GRID = {} -> one run with the settings above.
## Reading the result (train accuracy is measured on the training part, inference mode):
##   train and validation both low and close   -> too simple / too regularized: wider, deeper, less dropout
##   train much higher than validation (>0.15) -> overfitting: more dropout / weight decay, or smaller
GRID = {
    'HIDDEN_LAYERS': [[512, 256], [1024, 512, 256], [2048, 1024, 512, 256]],
    'DROPOUT_RATE': [0.2, 0.4],
    'WEIGHT_DECAY': [0.0, 1e-4],
}
## each combination is trained once per seed and the validation scores are averaged (differences below ~0.01 between
## two combinations are within the noise of one seed: use 2-3 seeds to separate close ones)
GRID_SEEDS = [42]

## ---- Final model(s) ------------------------------------------------------------------------------------------
## FINAL_FOLDS >= 2: the best grid combination is trained FINAL_FOLDS times (stratified k-fold over all "train" rows)
##                   -> model_<NICKNAME>1.keras ... model_<NICKNAME>k.keras (averaged by test_<NICKNAME>.py)
## FINAL_FOLDS = 0  : the best grid run itself is saved as model_<NICKNAME>.keras
## Older model files in this folder are renamed to *.keras.old first, so test_<NICKNAME>.py never mixes them in.
FINAL_FOLDS = 5

#==================================================================================================================

AUTOTUNE = tf.data.AUTOTUNE

## Layout:   <project>/Data/*.jpg
##           <project>/excel/*.xlsx
##           <project>/Code/train_Matthew.py   <- this file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR
PATH = os.path.dirname(SCRIPT_DIR)
DATA_DIR = PATH + os.path.sep + 'Data' + os.path.sep

GRID_KEYS = ('HIDDEN_LAYERS', 'DROPOUT_RATE', 'INPUT_DROPOUT', 'WEIGHT_DECAY', 'USE_BATCHNORM', 'LEARNING_RATE',
             'BATCH_SIZE', 'CLASS_WEIGHTS', 'LABEL_SMOOTHING', 'FEATURE_GROUPS', 'ACTIVATION', 'OPTIMIZER')
MONITORS = ('val_score', 'val_loss', 'val_f1_macro', 'val_kappa', 'val_accuracy')


def out_file(name):
    return os.path.join(OUT_DIR, name)
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
# FEATURE EXTRACTION - this block is copied unchanged into train_Matthew.py and test_Matthew.py
#------------------------------------------------------------------------------------------------------------------
## Every group is a histogram or a summary of the WHOLE image (or of its centre), so a lesion counts the same wherever
## it is on the leaf. Groups are always computed and concatenated in the order of FEATURE_ORDER.
## Histograms are square-rooted (Hellinger / power normalization: rare lesion colours and textures count more).
## LBP codes are grouped into the 51 classes that stay the same when the leaf is flipped or turned by 90 degrees.
## Change FEATURE_VERSION whenever this block changes: test_Matthew.py refuses a model made with another version.
FEATURE_VERSION = 1
FEATURE_ORDER = ('hs', 'v', 'mom', 'exr', 'lbp', 'lbp2', 'lbp4', 'grad', 'ctr', 'thumb')
FEATURE_SIZE = {'hs': 288, 'v': 16, 'mom': 12, 'exr': 64, 'lbp': 51, 'lbp2': 51, 'lbp4': 51, 'grad': 16,
                'ctr': 355, 'thumb': 768}
SQRT_GROUPS = ('hs', 'v', 'exr', 'lbp', 'lbp2', 'lbp4', 'grad', 'ctr')


def _lbp_orbits():
    '''
        table code (0..255) -> class (0..50). Neighbours are numbered clockwise around the pixel
        (top-left, top, top-right, right, bottom-right, bottom, bottom-left, left): turning the image by 90 degrees
        shifts the 8 bits by 2 places, a left-right flip mirrors them. Codes that one of these 8 moves turns into each
        other belong to the same class.
    '''
    def moved(p):
        return np.array([sum(((c >> i) & 1) << p[i] for i in range(8)) for c in range(256)])
    turn, mirror = moved([(i + 2) % 8 for i in range(8)]), moved([2, 1, 0, 7, 6, 5, 4, 3])
    images = [np.arange(256)]
    for _ in range(3):
        images.append(turn[images[-1]])
    images += [mirror[m] for m in images]
    smallest = np.min(np.stack(images), axis=0)            # representative of the class of every code
    return np.searchsorted(np.unique(smallest), smallest).astype(np.int32)


LBP_CLASS = _lbp_orbits()


def sort_feature_groups(groups):
    '''
        checks the names and puts them in the canonical order of FEATURE_ORDER (train and test must agree)
    '''
    unknown = [g for g in groups if g not in FEATURE_ORDER]
    if unknown:
        raise ValueError('Unknown feature groups {}, valid: {}'.format(unknown, FEATURE_ORDER))
    return [g for g in FEATURE_ORDER if g in groups]


def _counts(idx, n):
    '''
        normalized histogram (sums to 1) of integer codes 0..n-1
    '''
    c = tf.math.bincount(tf.reshape(idx, [-1]), minlength=n, maxlength=n, dtype=tf.float32)
    return c / tf.reduce_sum(c)


def _value_hist(x, n, lo, hi):
    '''
        normalized histogram of real values in [lo, hi] (values outside go to the first / last bin)
    '''
    idx = tf.cast(tf.floor((x - lo) / (hi - lo) * n), tf.int32)
    return _counts(tf.clip_by_value(idx, 0, n - 1), n)


def _hs_v(img):
    '''
        hs : joint hue x saturation histogram, 36 x 8 bins (288)   - colour of the leaf and of the lesions
        v  : brightness histogram, 16 bins                          - light / dark spots, white mildew
    '''
    hsv = tf.image.rgb_to_hsv(img)
    h = tf.minimum(tf.cast(hsv[..., 0] * 36, tf.int32), 35)
    s = tf.minimum(tf.cast(hsv[..., 1] * 8, tf.int32), 7)
    v = tf.minimum(tf.cast(hsv[..., 2] * 16, tf.int32), 15)
    return _counts(h * 8 + s, 288), _counts(v, 16), hsv


def _lbp(gray):
    '''
        local binary pattern: every pixel is compared with its 8 neighbours (1 if neighbour >= centre)
        -> an 8-bit code describing the local texture -> histogram of the 51 flip/turn classes of the codes
    '''
    c = gray[1:-1, 1:-1]
    nb = [gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:], gray[1:-1, 2:],
          gray[2:, 2:], gray[2:, 1:-1], gray[2:, :-2], gray[1:-1, :-2]]
    code = tf.add_n([tf.cast(n >= c, tf.int32) * (2 ** i) for i, n in enumerate(nb)])
    return _counts(tf.gather(tf.constant(LBP_CLASS), code), 51)


def _shrink(gray2d, factor):
    '''
        gray image [H, W] reduced by an integer factor (area average)
    '''
    hw = tf.shape(gray2d)[:2] // factor
    return tf.image.resize(gray2d[..., None], hw, antialias=True)[..., 0]


def decode_image(file_path, ratio):
    '''
        reads a JPEG and decodes it directly at 1/ratio of its size (4000x2672 -> 500x334 with ratio 8)
        returns float32 [H, W, 3] in [0, 1]
    '''
    img = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3, ratio=ratio)
    return tf.cast(img, tf.float32) / 255.0


def image_features(img, groups):
    '''
        img    : [H, W, 3] float32 in [0, 1] (from decode_image)
        groups : names from FEATURE_ORDER, already sorted with sort_feature_groups
        returns a 1-D float32 vector (length = sum of FEATURE_SIZE of the groups)
    '''
    out = {}
    need = set(groups)
    gray = tf.image.rgb_to_grayscale(img)[..., 0]

    if need & {'hs', 'v', 'mom'}:
        hs, v, hsv = _hs_v(img)
        out['hs'], out['v'] = hs, v
        if 'mom' in need:
            # colour moments: mean and std of R, G, B, S, V + circular mean of the hue (cos, sin)  -> 12
            chans = [img[..., 0], img[..., 1], img[..., 2], hsv[..., 1], hsv[..., 2]]
            ang = 2.0 * np.pi * hsv[..., 0]
            out['mom'] = tf.stack([tf.reduce_mean(c) for c in chans] + [tf.math.reduce_std(c) for c in chans]
                                  + [tf.reduce_mean(tf.cos(ang)), tf.reduce_mean(tf.sin(ang))])
    if 'exr' in need:
        # excess red (R - G: brown / orange lesions) and excess green (2G - R - B: healthy tissue), 32 bins each
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        out['exr'] = tf.concat([_value_hist(r - g, 32, -0.4, 0.4), _value_hist(2 * g - r - b, 32, -0.4, 0.8)], 0)
    if 'lbp' in need:
        out['lbp'] = _lbp(gray)                      # fine texture, at the decoded size
    if 'lbp2' in need:
        out['lbp2'] = _lbp(_shrink(gray, 2))         # texture at 1/2 size
    if 'lbp4' in need:
        out['lbp4'] = _lbp(_shrink(gray, 4))         # texture at 1/4 size
    if 'grad' in need:
        # edge strength histogram (borders of spots and lesions), 16 bins
        gx = gray[:-1, 1:] - gray[:-1, :-1]
        gy = gray[1:, :-1] - gray[:-1, :-1]
        out['grad'] = _value_hist(tf.sqrt(gx * gx + gy * gy), 16, 0.0, 0.25)
    if 'ctr' in need:
        # hs + v + lbp of the central part (half the height and half the width): mostly leaf, little background
        h, w = tf.shape(img)[0], tf.shape(img)[1]
        crop = img[h // 4: h - h // 4, w // 4: w - w // 4]
        c_hs, c_v, _ = _hs_v(crop)
        out['ctr'] = tf.concat([c_hs, c_v, _lbp(tf.image.rgb_to_grayscale(crop)[..., 0])], 0)
    if 'thumb' in need:
        # 16 x 16 colour thumbnail (layout of the image), 768 values
        out['thumb'] = tf.reshape(tf.image.resize(img, [16, 16], antialias=True), [-1])

    return tf.concat([tf.sqrt(out[k]) if k in SQRT_GROUPS else out[k] for k in groups], axis=0)
#------------------------------------------------------------------------------------------------------------------

def extract_features(paths, groups, ratio):
    '''
        features of every image in paths (same order), as a float32 array [len(paths), n_features]

        Parallel "data generator": tf.data runs image_features on EXTRACT_PARALLEL images at the same time
        (0 = AUTOTUNE = one per CPU core). The JPEG decoding and the histogram ops run in TensorFlow's C++ thread
        pool, outside the Python GIL, so all the vCPUs of the instance work together (measured: 1.6x faster with
        2 cores than with 1). deterministic=True keeps the output in the order of paths.
    '''
    par = AUTOTUNE if EXTRACT_PARALLEL <= 0 else EXTRACT_PARALLEL
    ds = tf.data.Dataset.from_tensor_slices(list(paths))
    ds = ds.map(lambda f: image_features(decode_image(f, ratio), groups), num_parallel_calls=par, deterministic=True)
    ds = ds.batch(EXTRACT_BATCH).prefetch(AUTOTUNE)

    out, done, t0 = [], 0, time.time()
    for xb in ds:
        out.append(xb.numpy())
        done += xb.shape[0]
        if done % (EXTRACT_BATCH * 16) < EXTRACT_BATCH or done == len(paths):
            rate = done / max(time.time() - t0, 1e-9)
            print('  features: {:6d} / {} images  ({:.0f} images/s, {:.1f} min left)'.format(
                done, len(paths), rate, (len(paths) - done) / rate / 60))
    return np.concatenate(out, axis=0).astype(np.float32)
#------------------------------------------------------------------------------------------------------------------

def load_feature_cache(ids):
    '''
        features of ALL groups (FEATURE_ORDER) for the image ids, from features_cache/ when available.
        Only the images missing from the cache are computed (first run: all of them, ~14,700 images).
        Returns X_all [len(ids), sum of all FEATURE_SIZE]
    '''
    folder = os.path.join(SCRIPT_DIR, CACHE_DIR)
    os.makedirs(folder, exist_ok=True)
    file_name = os.path.join(folder, 'features_r{}_v{}.npz'.format(FEATURE_RATIO, FEATURE_VERSION))
    n_all = sum(FEATURE_SIZE[g] for g in FEATURE_ORDER)

    cached_ids, cached_X = np.array([], dtype=object), np.zeros((0, n_all), np.float32)
    if os.path.exists(file_name):
        z = np.load(file_name, allow_pickle=True)
        if z['X'].shape[1] == n_all:
            cached_ids, cached_X = z['ids'], z['X']
        else:
            print('Feature cache has another layout, it is recomputed')

    position = {k: i for i, k in enumerate(cached_ids)}
    missing = [i for i in dict.fromkeys(ids) if i not in position]     # unique ids, in order
    if missing:
        print('Computing the features of {} images (ratio {}, all groups, cached in {}) ...'.format(
            len(missing), FEATURE_RATIO, file_name))
        X_new = extract_features([DATA_DIR + i for i in missing], list(FEATURE_ORDER), FEATURE_RATIO)
        if not np.all(np.isfinite(X_new)):
            raise ValueError('NaN/inf in the features of some images')
        cached_ids = np.concatenate([cached_ids, np.array(missing, dtype=object)])
        cached_X = np.concatenate([cached_X, X_new], axis=0)
        np.savez(file_name, ids=cached_ids, X=cached_X)
        position = {k: i for i, k in enumerate(cached_ids)}
    else:
        print('Features loaded from', file_name)

    return cached_X[[position[i] for i in ids]]
#------------------------------------------------------------------------------------------------------------------

def feature_columns(groups):
    '''
        column indices of the groups inside the all-groups array (groups sorted with sort_feature_groups)
    '''
    start = dict(zip(FEATURE_ORDER, np.cumsum([0] + [FEATURE_SIZE[g] for g in FEATURE_ORDER])[:-1]))
    return np.concatenate([np.arange(start[g], start[g] + FEATURE_SIZE[g]) for g in groups])
#------------------------------------------------------------------------------------------------------------------

def base_config():
    '''
        the settings at the top as a dict (a grid combination overrides some of them)
    '''
    return {k: globals()[k] for k in GRID_KEYS}
#------------------------------------------------------------------------------------------------------------------

def grid_configs():
    '''
        list of configs, one per combination of the GRID values
    '''
    for k in GRID:
        if k not in GRID_KEYS:
            raise ValueError('GRID key {} not allowed, use one of {}'.format(k, GRID_KEYS))
    keys = list(GRID.keys())
    configs = []
    for values in itertools.product(*[GRID[k] for k in keys]):
        cfg = base_config()
        cfg.update(dict(zip(keys, values)))
        cfg['FEATURE_GROUPS'] = sort_feature_groups(cfg['FEATURE_GROUPS'])
        configs.append(cfg)
    return configs
#------------------------------------------------------------------------------------------------------------------

def config_text(cfg):
    return 'feat={} | hidden={} | dropout={} | in_drop={} | wd={} | bn={} | lr={} | batch={} | cw={} | ls={}'.format(
        '-'.join(cfg['FEATURE_GROUPS']), cfg['HIDDEN_LAYERS'], cfg['DROPOUT_RATE'], cfg['INPUT_DROPOUT'],
        cfg['WEIGHT_DECAY'], cfg['USE_BATCHNORM'], cfg['LEARNING_RATE'], cfg['BATCH_SIZE'], cfg['CLASS_WEIGHTS'],
        cfg['LABEL_SMOOTHING'])
#------------------------------------------------------------------------------------------------------------------

def check_settings():
    '''
        stops early with a clear message if a setting at the top is mistyped
    '''
    sort_feature_groups(FEATURE_GROUPS)
    if FEATURE_RATIO not in (1, 2, 4, 8):
        raise ValueError('FEATURE_RATIO must be 1, 2, 4 or 8')
    if CHECKPOINT_MONITOR not in MONITORS or EARLY_STOP_MONITOR not in MONITORS:
        raise ValueError('CHECKPOINT_MONITOR / EARLY_STOP_MONITOR must be one of {}'.format(MONITORS))
    if LR_SCHEDULE not in ('plateau', 'none'):
        raise ValueError("LR_SCHEDULE must be 'plateau' or 'none'")
    if CLASS_WEIGHTS not in ('balanced', 'sqrt', 'none'):
        raise ValueError("CLASS_WEIGHTS must be 'balanced', 'sqrt' or 'none'")
    if OPTIMIZER not in ('adam', 'rmsprop', 'sgd'):
        raise ValueError("OPTIMIZER must be 'adam', 'rmsprop' or 'sgd'")
    if not 0.0 < VAL_FRACTION < 1.0:
        raise ValueError('VAL_FRACTION must be in (0, 1)')
    if FINAL_FOLDS == 1 or FINAL_FOLDS < 0:
        raise ValueError('FINAL_FOLDS must be 0 (single model) or >= 2 (k-fold ensemble)')
    if EARLY_STOP_PATIENCE > 0 and LR_SCHEDULE == 'plateau' and LR_PATIENCE >= EARLY_STOP_PATIENCE:
        print('WARNING: LR_PATIENCE >= EARLY_STOP_PATIENCE, the training stops before the learning rate is lowered')
    for cfg in grid_configs():
        if not 0.0 <= cfg['DROPOUT_RATE'] < 1.0 or not 0.0 <= cfg['INPUT_DROPOUT'] < 1.0:
            raise ValueError('DROPOUT_RATE and INPUT_DROPOUT must be in [0, 1)')
        if cfg['CLASS_WEIGHTS'] not in ('balanced', 'sqrt', 'none'):
            raise ValueError("CLASS_WEIGHTS must be 'balanced', 'sqrt' or 'none'")
        if len(cfg['HIDDEN_LAYERS']) == 0:
            raise ValueError('HIDDEN_LAYERS needs at least one layer')
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

def compute_class_weights(y_idx, num_classes, mode, verbose=False):
    '''
        weight w_c of every class, from the class index of every TRAINING image
        balanced: w_c = N / (K * n_c)  ->  average weight over the images = 1
        sqrt    : square root of the balanced weights, rescaled to an average of 1 (softer)
    '''
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float64)
    n, k = counts.sum(), num_classes

    if mode == 'none':
        w = np.ones(k)
    else:
        w = n / (k * np.maximum(counts, 1.0))
        if mode == 'sqrt':
            w = np.sqrt(w)
            w = w * n / np.sum(counts * w)

    if verbose:
        print('Class weights ({}):'.format(mode))
        for c in range(k):
            print('  {:10s} {:5d} training images   w = {:.3f}'.format(str(class_names[c]), int(counts[c]), w[c]))

    return w.astype(np.float32)
#------------------------------------------------------------------------------------------------------------------

def weighted_ce(prob, y_idx, class_w):
    '''
        performance index F on a set of images: class-weighted average cross-entropy
            F = sum_q w_q * (-log a_q[t_q]) / sum_q w_q
        returns (F, plain cross-entropy)
    '''
    p_true = np.clip(prob[np.arange(len(y_idx)), y_idx], 1e-7, 1.0)
    ce = -np.log(p_true)
    return float(np.average(ce, weights=class_w[y_idx])), float(np.mean(ce))
#------------------------------------------------------------------------------------------------------------------

def scores(y_true, y_pred):
    '''
        macro F1, Cohen kappa, accuracy and score = (F1 + kappa) / 2 (leaderboard formula)
    '''
    f1 = float(f1_score(y_true, y_pred, average='macro'))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    return {'f1_macro': f1, 'kappa': kappa, 'acc': float(accuracy_score(y_true, y_pred)), 'score': (f1 + kappa) / 2}
#------------------------------------------------------------------------------------------------------------------

def predict_np(model, X, batch=4096):
    '''
        softmax output for a numpy array of features (inference mode: BatchNorm moving averages, no Dropout)
    '''
    return np.concatenate([np.asarray(model(X[i:i + batch], training=False)) for i in range(0, len(X), batch)])
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

def make_optimizer(cfg):
    '''
        returns the optimizer selected with OPTIMIZER (Adam: Kingma and Ba [2014] defaults)
    '''
    clip = {'global_clipnorm': CLIP_NORM} if CLIP_NORM > 0 else {}
    lr = cfg['LEARNING_RATE']
    if cfg['OPTIMIZER'] == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2,
                                        epsilon=ADAM_EPSILON, **clip)
    if cfg['OPTIMIZER'] == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=lr, **clip)
    return tf.keras.optimizers.SGD(learning_rate=lr, momentum=SGD_MOMENTUM, **clip)
#------------------------------------------------------------------------------------------------------------------

def model_name(cfg):
    '''
        the name tells test_Matthew.py how to compute the input of the model:
        MLP_Matthew_feat_r8_g-hs-v-mom-exr-lbp-lbp2-lbp4-grad-ctr_v1
            r8 = JPEG decode ratio, g-... = feature groups in FEATURE_ORDER, v1 = FEATURE_VERSION
    '''
    return 'MLP_{}_feat_r{}_g-{}_v{}'.format(NICKNAME, FEATURE_RATIO, '-'.join(cfg['FEATURE_GROUPS']), FEATURE_VERSION)
#------------------------------------------------------------------------------------------------------------------

def model_definition(cfg, mean, std):
    '''
        Input (features) -> Normalization (z-score with the TRAINING statistics, saved inside the model)
        -> [Dropout] -> hidden layers: Dense -> [BatchNorm] -> activation -> [Dropout] -> Dense softmax
    '''
    model = tf.keras.Sequential(name=model_name(cfg))
    model.add(tf.keras.layers.Input(shape=(len(mean),)))
    model.add(tf.keras.layers.Normalization(axis=-1, mean=mean.astype(np.float32),
                                            variance=(std ** 2).astype(np.float32), name='zscore_norm_features'))
    if cfg['INPUT_DROPOUT'] > 0:
        model.add(tf.keras.layers.Dropout(cfg['INPUT_DROPOUT'], name='input_dropout'))

    wd = cfg['WEIGHT_DECAY']
    reg = tf.keras.regularizers.L2(wd) if wd > 0 else None
    bn = cfg['USE_BATCHNORM']

    for units in cfg['HIDDEN_LAYERS']:
        # with BatchNorm the Dense bias is redundant (BatchNorm subtracts the mean and adds its own shift beta)
        model.add(tf.keras.layers.Dense(units, use_bias=not bn, kernel_initializer=KERNEL_INIT, kernel_regularizer=reg))
        if bn:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(cfg['ACTIVATION']))
        if cfg['DROPOUT_RATE'] > 0:
            model.add(tf.keras.layers.Dropout(cfg['DROPOUT_RATE']))

    model.add(tf.keras.layers.Dense(OUTPUTS_a, activation='softmax', kernel_regularizer=reg))

    model.compile(optimizer=make_optimizer(cfg),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=cfg['LABEL_SMOOTHING']),
                  metrics=[tf.keras.metrics.CategoricalCrossentropy(name='categorical_ce')])
    return model
#------------------------------------------------------------------------------------------------------------------

class ValidationMetrics(tf.keras.callbacks.Callback):
    '''
        at the end of every epoch adds to the epoch log (the callbacks placed after it can monitor these):
          lr, val_loss (F_val, class-weighted cross-entropy), val_ce, val_accuracy, val_f1_macro, val_kappa,
          val_score = (val_f1_macro + val_kappa) / 2, train_F and train_accuracy (training part, inference mode)
    '''
    def __init__(self, X_val, y_val, class_w, X_tr=None, y_tr=None, label=''):
        super().__init__()
        self.X_val, self.y_val, self.class_w = X_val, y_val, class_w
        self.X_tr, self.y_tr, self.label = X_tr, y_tr, label

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        logs['lr'] = float(np.asarray(self.model.optimizer.learning_rate))
        prob = predict_np(self.model, self.X_val)

        if not np.all(np.isfinite(prob)):
            logs.update({'val_loss': np.inf, 'val_ce': np.inf, 'val_accuracy': -1.0, 'val_f1_macro': -1.0,
                         'val_kappa': -1.0, 'val_score': -1.0, 'train_F': np.nan, 'train_accuracy': np.nan})
            print('  {} epoch {:3d} | NaN/inf in the predictions, this epoch is discarded'.format(self.label, epoch + 1))
            return

        pred = np.argmax(prob, axis=1)
        logs['val_loss'], logs['val_ce'] = weighted_ce(prob, self.y_val, self.class_w)
        s = scores(self.y_val, pred)
        logs['val_accuracy'], logs['val_f1_macro'], logs['val_kappa'], logs['val_score'] = \
            s['acc'], s['f1_macro'], s['kappa'], s['score']

        if self.X_tr is not None:
            p_tr = predict_np(self.model, self.X_tr)
            logs['train_F'] = weighted_ce(p_tr, self.y_tr, self.class_w)[0]
            logs['train_accuracy'] = float(np.mean(np.argmax(p_tr, axis=1) == self.y_tr))
        else:
            logs['train_F'], logs['train_accuracy'] = np.nan, np.nan

        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
            print('  {} epoch {:3d} | lr {:.1e} | train F {:.4f} acc {:.4f} | val F {:.4f} acc {:.4f} '
                  'f1 {:.4f} kappa {:.4f} score {:.4f}'.format(
                      self.label, epoch + 1, logs['lr'], logs['train_F'], logs['train_accuracy'], logs['val_loss'],
                      logs['val_accuracy'], logs['val_f1_macro'], logs['val_kappa'], logs['val_score']))
#------------------------------------------------------------------------------------------------------------------

class BestWeights(tf.keras.callbacks.Callback):
    '''
        keeps in memory the weights of the epoch with the best CHECKPOINT_MONITOR (no file written per run)
    '''
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.sign = -1.0 if monitor == 'val_loss' else 1.0      # val_loss: lower is better
        self.best, self.weights, self.epoch, self.logs = -np.inf, None, 0, {}

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get(self.monitor)
        if v is not None and np.isfinite(v) and self.sign * v > self.best:
            self.best, self.weights, self.epoch = self.sign * v, self.model.get_weights(), epoch + 1
            self.logs = dict(logs)
#------------------------------------------------------------------------------------------------------------------

def monitor_mode(monitor):
    return 'min' if monitor == 'val_loss' else 'max'
#------------------------------------------------------------------------------------------------------------------

def train_one(cfg, X_tr, y_tr, X_val, y_val, seed, label=''):
    '''
        trains one model of config cfg on (X_tr, y_tr), early stopping on (X_val, y_val)
        X_* : features of the cfg feature groups,  y_* : class index
        returns model (weights of the best epoch), history, info dict
    '''
    tf.keras.utils.set_random_seed(seed)

    class_w = compute_class_weights(y_tr, OUTPUTS_a, cfg['CLASS_WEIGHTS'])

    # z-score statistics of the TRAINING part only, stored in the first layer of the model
    mean = X_tr.mean(axis=0)
    std = np.maximum(X_tr.std(axis=0), MIN_FEATURE_STD)

    y_onehot = to_categorical(y_tr, num_classes=OUTPUTS_a).astype(np.float32)
    w = class_w[y_tr]
    ds = tf.data.Dataset.from_tensor_slices((X_tr, y_onehot, w))
    ds = ds.shuffle(len(X_tr), seed=seed, reshuffle_each_iteration=True).batch(cfg['BATCH_SIZE']).prefetch(AUTOTUNE)

    best = BestWeights(CHECKPOINT_MONITOR)
    callbacks = [tf.keras.callbacks.TerminateOnNaN(),
                 ValidationMetrics(X_val, y_val, class_w, X_tr, y_tr, label), best]
    if LR_SCHEDULE == 'plateau':
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=EARLY_STOP_MONITOR,
                                                              mode=monitor_mode(EARLY_STOP_MONITOR),
                                                              factor=LR_FACTOR, patience=LR_PATIENCE,
                                                              min_delta=0.0, min_lr=LR_MIN, verbose=0))
    if EARLY_STOP_PATIENCE > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=EARLY_STOP_MONITOR,
                                                          mode=monitor_mode(EARLY_STOP_MONITOR),
                                                          patience=EARLY_STOP_PATIENCE, restore_best_weights=False,
                                                          verbose=0))

    model = model_definition(cfg, mean, std)
    t0 = time.time()
    history = model.fit(ds, epochs=n_epoch, shuffle=False, callbacks=callbacks, verbose=0)

    if best.weights is None:
        raise RuntimeError('No valid epoch (loss NaN/inf from the start). Lower LEARNING_RATE or turn on USE_BATCHNORM.')
    model.set_weights(best.weights)

    n_run = len(history.history.get('loss', []))
    info = {'epochs_run': n_run, 'best_epoch': best.epoch, 'seconds': time.time() - t0,
            'train_accuracy': best.logs.get('train_accuracy'), 'final_lr': history.history.get('lr', [np.nan])[-1],
            'val': {'f1_macro': best.logs['val_f1_macro'], 'kappa': best.logs['val_kappa'],
                    'acc': best.logs['val_accuracy'], 'score': best.logs['val_score']}}
    print('  {} stopped at epoch {}, best epoch {} ({:.0f} s): train acc {:.4f} | val acc {:.4f} f1 {:.4f} '
          'kappa {:.4f} score {:.4f}'.format(label, n_run, best.epoch, info['seconds'], info['train_accuracy'],
                                             info['val']['acc'], info['val']['f1_macro'], info['val']['kappa'],
                                             info['val']['score']))
    return model, history, info
#------------------------------------------------------------------------------------------------------------------

def early_stopping_report(history, title):
    '''
        book NND2 Fig. 13.4 for one run:
          point "a" = epoch with the minimum validation error F_val, point "b" = last epoch trained
        writes early_stopping_<NICKNAME>.csv (one row per epoch) and early_stopping_<NICKNAME>.png
    '''
    h = history.history
    val_F = np.asarray(h.get('val_loss', []), dtype=float)
    n = len(val_F)
    if n == 0 or not np.any(np.isfinite(val_F)):
        print('Early stopping report skipped: no valid validation error')
        return

    def column(key):
        v = list(h.get(key, []))[:n]
        return np.asarray(v + [np.nan] * (n - len(v)), dtype=float)

    epochs = np.arange(1, n + 1)
    a = int(np.nanargmin(np.where(np.isfinite(val_F), val_F, np.nan))) + 1
    b = n
    monitored = column(CHECKPOINT_MONITOR)
    saved = int(np.nanargmin(monitored) if monitor_mode(CHECKPOINT_MONITOR) == 'min' else np.nanargmax(monitored)) + 1

    table = pd.DataFrame({'epoch': epochs, 'lr': column('lr'), 'train_loss_keras': column('loss'),
                          'train_F': column('train_F'), 'train_accuracy': column('train_accuracy'),
                          'val_F': val_F, 'val_ce': column('val_ce'), 'val_accuracy': column('val_accuracy'),
                          'val_f1_macro': column('val_f1_macro'), 'val_kappa': column('val_kappa'),
                          'val_score': column('val_score')})
    table.to_csv(out_file('early_stopping_{}.csv'.format(NICKNAME)), index=False)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib is not installed: early_stopping_{}.png skipped (the numbers are in the .csv)'.format(NICKNAME))
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={'width_ratios': [1.5, 1, 1]})

    ax = axes[0]
    ax.plot(epochs, val_F, color='#1f3fd1', lw=2, label='Validation')
    ax.plot(epochs, table['train_F'], color='black', lw=2, label='Training (inference mode)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('F (cross-entropy{})'.format(', class-weighted' if CLASS_WEIGHTS != 'none' else ''))
    ax.set_title('Early stopping (NND2 Fig. 13.4): a = epoch {}, b = epoch {}'.format(a, b))
    ax.legend(frameon=False, loc='best')

    ax = axes[1]
    ax.plot(epochs, table['val_score'], color='#1f3fd1', lw=2, label='val (F1 + kappa)/2')
    ax.plot(epochs, table['train_accuracy'], color='black', lw=1.5, ls='--', label='train accuracy')
    ax.plot(epochs, table['val_accuracy'], color='#1f3fd1', lw=1.5, ls='--', label='val accuracy')
    ax.set_xscale('log')
    ax.set_title('Score and accuracy (kept: epoch {})'.format(saved))
    ax.legend(frameon=False, loc='best')

    ax = axes[2]
    ax.step(epochs, table['lr'], where='post', color='black', lw=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('learning rate alpha')
    ax.set_title('Learning rate ({})'.format(LR_SCHEDULE))

    for ax in axes:
        ax.set_title(ax.get_title(), pad=24)
        ax.set_xlabel('Epoch (iteration)')
        ax.grid(True, which='major', color='0.9', lw=0.8)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for x, name in (((a, 'a'), (b, 'b')) if a != b else ((a, 'a = b'),)):
            ax.axvline(x, color='0.35', ls=':', lw=1.2)
            ax.text(x, 1.01, name, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=13)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    file_name = out_file('early_stopping_{}.png'.format(NICKNAME))
    fig.savefig(file_name, dpi=130)
    plt.close(fig)
    print('Early stopping plot saved to', file_name)
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

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_true = xdf_dset['target'].map(class_to_idx).to_numpy()
    y_pred = xdf_dset['results'].to_numpy()

    table = {'f1_micro': ('f1_score micro', lambda t, p: f1_score(t, p, average='micro')),
             'f1_macro': ('f1_score macro', lambda t, p: f1_score(t, p, average='macro')),
             'f1_weighted': ('f1_score weighted', lambda t, p: f1_score(t, p, average='weighted')),
             'coh': ('cohen_kappa_score', cohen_kappa_score),
             'acc': ('accuracy_score', accuracy_score),
             'mat': ('mattews_coef', matthews_corrcoef)}

    xcont, xsum, values = 0, 0, {}
    for xm in metrics:
        if xm not in table:
            print('Metric does not exist:', xm)
            continue
        name, fn = table[xm]
        xmet = fn(y_true, y_pred)
        print(name, xmet)
        values[xm] = float(xmet)
        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum)
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum / xcont)

    if confusion:
        print('Confusion matrix (rows = true, columns = predicted), classes:', list(class_names))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        print(cm)
        recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        print('Recall per class:', ' '.join('{} {:.2f}'.format(c, r) for c, r in zip(class_names, recall)))

    return values
#------------------------------------------------------------------------------------------------------------------

def log_experiment(row):
    '''
        appends one line to experiments_<NICKNAME>.csv (rewritten with all columns if the columns changed)
    '''
    row = dict({'date': datetime.now().strftime('%Y-%m-%d %H:%M')}, **row)
    file_name = out_file('experiments_{}.csv'.format(NICKNAME))
    new_file = not os.path.exists(file_name)

    if not new_file:
        old = pd.read_csv(file_name)
        if list(old.columns) != list(row.keys()):
            pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(file_name, index=False)
            return

    with open(file_name, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
#------------------------------------------------------------------------------------------------------------------

def experiment_row(stage, cfg, info, test, seed, extra=None):
    row = {'stage': stage, 'model': 'features', 'feature_groups': '-'.join(cfg['FEATURE_GROUPS']),
           'n_features': int(sum(FEATURE_SIZE[g] for g in cfg['FEATURE_GROUPS'])),
           'hidden': str(cfg['HIDDEN_LAYERS']), 'dropout': cfg['DROPOUT_RATE'], 'input_dropout': cfg['INPUT_DROPOUT'],
           'batchnorm': cfg['USE_BATCHNORM'], 'weight_decay': cfg['WEIGHT_DECAY'], 'optimizer': cfg['OPTIMIZER'],
           'lr': cfg['LEARNING_RATE'], 'label_smoothing': cfg['LABEL_SMOOTHING'], 'class_weights': cfg['CLASS_WEIGHTS'],
           'batch': cfg['BATCH_SIZE'], 'seed': seed, 'epochs_run': info.get('epochs_run'),
           'best_epoch': info.get('best_epoch'), 'final_lr': info.get('final_lr'),
           'train_acc': info.get('train_accuracy'),
           'val_score': info['val']['score'], 'val_f1_macro': info['val']['f1_macro'],
           'val_kappa': info['val']['kappa'], 'val_acc': info['val']['acc'],
           'test_score': test['score'], 'test_f1_macro': test['f1_macro'], 'test_kappa': test['kappa'],
           'test_acc': test['acc']}
    row.update(extra or {})
    return row
#------------------------------------------------------------------------------------------------------------------

def retire_old_models():
    '''
        renames model_<NICKNAME>.keras and model_<NICKNAME><k>.keras in this folder to *.keras.old
        (test_<NICKNAME>.py loads every model_<NICKNAME>*.keras it finds)
    '''
    prefix = 'model_{}'.format(NICKNAME)
    for name in sorted(os.listdir(OUT_DIR)):
        middle = name[len(prefix):-len('.keras')] if name.startswith(prefix) and name.endswith('.keras') else None
        if middle is not None and (middle == '' or middle.isdigit()):
            os.replace(out_file(name), out_file(name + '.old'))
            print('  previous model renamed:', name, '->', name + '.old')
#------------------------------------------------------------------------------------------------------------------

def main():
    global xdf_data, class_names, OUTPUTS_a, xdf_dset, PATH, DATA_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, required=False)  # project folder (optional)
    args = parser.parse_args()
    if args.path:
        PATH = os.path.abspath(args.path)
        DATA_DIR = PATH + os.path.sep + 'Data' + os.path.sep

    check_settings()
    configs = grid_configs()
    print('Grid: {} combinations x {} seed(s) = {} runs, then {}'.format(
        len(configs), len(GRID_SEEDS), len(configs) * len(GRID_SEEDS),
        '{}-fold ensemble'.format(FINAL_FOLDS) if FINAL_FOLDS >= 2 else 'the best run is saved'))

    FILE_NAME = find_excel()
    xdf_data = pd.read_excel(FILE_NAME)
    class_names = process_target(1)  # 1: Multiclass 2: Multilabel 3:Binary
    OUTPUTS_a = len(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    is_train = (xdf_data['split'] == 'train').to_numpy()
    is_test = (xdf_data['split'] == 'test').to_numpy()
    if not is_train.any() or not is_test.any():
        raise ValueError('The excel file needs rows with split "train" and "test"')

    missing = [p for p in (DATA_DIR + xdf_data['id']) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('{} images not found, for example {}. Data folder: {}'.format(
            len(missing), missing[:3], DATA_DIR))

    ## Features of every row (train + test), all groups; each config takes its columns
    X_all = load_feature_cache(xdf_data['id'].tolist())
    y_all = xdf_data['target'].map(class_to_idx).to_numpy()

    idx_train = np.where(is_train)[0]
    idx_test = np.where(is_test)[0]

    ## Fixed split for the grid: stratified, same seed -> every combination sees the same images
    idx_tr, idx_val = train_test_split(idx_train, test_size=VAL_FRACTION, stratify=y_all[idx_train],
                                       random_state=SEED)
    idx_tr, idx_val = np.sort(idx_tr), np.sort(idx_val)
    print('Training images: {}   Validation images: {}   Test images: {}'.format(
        len(idx_tr), len(idx_val), len(idx_test)))
    compute_class_weights(y_all[idx_tr], OUTPUTS_a, CLASS_WEIGHTS, verbose=True)

    #--------------------------------------------------------------------------------------------------------------
    # GRID
    ## only the leading combination is kept in memory (its best seed): model, history, scores
    grid_rows, best = [], None
    for c, cfg in enumerate(configs):
        cols = feature_columns(cfg['FEATURE_GROUPS'])
        Xc = X_all[:, cols]
        print('\n==== Grid {}/{}: {} ({} features) ===='.format(c + 1, len(configs), config_text(cfg), len(cols)))
        runs = []
        for seed in GRID_SEEDS:
            model, history, info = train_one(cfg, Xc[idx_tr], y_all[idx_tr], Xc[idx_val], y_all[idx_val], seed,
                                             label='[{}/{} s{}]'.format(c + 1, len(configs), seed))
            test = scores(y_all[idx_test], np.argmax(predict_np(model, Xc[idx_test]), axis=1))
            print('  test (for information only, not used to choose): f1 {:.4f} kappa {:.4f} score {:.4f}'.format(
                test['f1_macro'], test['kappa'], test['score']))
            log_experiment(experiment_row('grid', cfg, info, test, seed))
            runs.append({'model': model, 'history': history, 'info': info, 'test': test, 'seed': seed})
            grid_rows.append({'combination': c + 1, 'seed': seed, 'config': config_text(cfg),
                              'params': model.count_params(), 'epochs_run': info['epochs_run'],
                              'best_epoch': info['best_epoch'], 'train_acc': info['train_accuracy'],
                              'val_acc': info['val']['acc'], 'val_score': info['val']['score'],
                              'test_score': test['score']})

        mean_val = float(np.mean([r['info']['val']['score'] for r in runs]))
        if best is None or mean_val > best['mean_val']:
            top = max(runs, key=lambda r: r['info']['val']['score'])
            best = dict(top, cfg=cfg, mean_val=mean_val, combination=c + 1)
        del runs
        print('  combination {}: mean val score {:.4f} | leader: combination {} ({:.4f})'.format(
            c + 1, mean_val, best['combination'], best['mean_val']))
        pd.DataFrame(grid_rows).to_csv(out_file('grid_{}.csv'.format(NICKNAME)), index=False)

    ranking = pd.DataFrame(grid_rows).groupby(['combination', 'config'], sort=False)[
        ['params', 'train_acc', 'val_acc', 'val_score', 'test_score']].mean()
    ranking = ranking.sort_values('val_score', ascending=False)
    print('\n================ GRID RANKING (mean over seeds, sorted by validation score) ================')
    print('train_acc much higher than val_acc -> overfitting; both low and close -> underfitting')
    with pd.option_context('display.width', 250, 'display.max_colwidth', 140):
        print(ranking.round(4).to_string())
    best_cfg = best['cfg']
    print('\nBest combination {}: {}'.format(best['combination'], config_text(best_cfg)))
    early_stopping_report(best['history'], 'Best grid run: ' + config_text(best_cfg))

    cols = feature_columns(best_cfg['FEATURE_GROUPS'])
    Xb = X_all[:, cols]

    #--------------------------------------------------------------------------------------------------------------
    # FINAL MODEL(S)
    print('\n================ FINAL MODEL(S) ================')
    retire_old_models()

    if FINAL_FOLDS >= 2:
        skf = StratifiedKFold(n_splits=FINAL_FOLDS, shuffle=True, random_state=SEED)
        oof = np.zeros((len(idx_train), OUTPUTS_a), np.float32)
        test_prob = np.zeros((len(idx_test), OUTPUTS_a), np.float32)
        model_files = []
        for k, (tr, va) in enumerate(skf.split(idx_train, y_all[idx_train])):
            i_tr, i_va = idx_train[tr], idx_train[va]
            model, history, info = train_one(best_cfg, Xb[i_tr], y_all[i_tr], Xb[i_va], y_all[i_va], SEED + k,
                                             label='[fold {}/{}]'.format(k + 1, FINAL_FOLDS))
            oof[va] = predict_np(model, Xb[i_va])
            p_test = predict_np(model, Xb[idx_test])
            test_prob += p_test / FINAL_FOLDS
            test = scores(y_all[idx_test], np.argmax(p_test, axis=1))
            print('  fold {} model alone on test: score {:.4f}'.format(k + 1, test['score']))
            log_experiment(experiment_row('fold', best_cfg, info, test, SEED + k, {'fold': k + 1}))
            file_name = out_file('model_{}{}.keras'.format(NICKNAME, k + 1))
            model.save(file_name)
            model_files.append(file_name)

        oof_s = scores(y_all[idx_train], np.argmax(oof, axis=1))
        print('\nOut-of-fold (every "train" image predicted by the model that did not train on it, {} images): '
              'f1 {:.4f} kappa {:.4f} acc {:.4f} score {:.4f}'.format(len(idx_train), oof_s['f1_macro'],
                                                                     oof_s['kappa'], oof_s['acc'], oof_s['score']))
        final_info = {'epochs_run': '', 'best_epoch': '', 'final_lr': '', 'train_accuracy': '', 'val': oof_s}
    else:
        file_name = out_file('model_{}.keras'.format(NICKNAME))
        best['model'].save(file_name)
        model_files = [file_name]
        test_prob = predict_np(best['model'], Xb[idx_test])
        final_info = best['info']

    sizes = sum(os.path.getsize(f) for f in model_files) / 1e6
    print('Saved {} model file(s), {:.0f} MB in total: {}'.format(len(model_files), sizes,
                                                                  ', '.join(os.path.basename(f) for f in model_files)))

    # summary from the first SAVED model, as test_Matthew.py writes it
    save_model(tf.keras.models.load_model(model_files[0]))

    #--------------------------------------------------------------------------------------------------------------
    # TEST: same averaging as test_Matthew.py -> results_<NICKNAME>.xlsx
    print('\n================ TEST ({}) ================'.format(
        'ensemble of {} models'.format(len(model_files)) if len(model_files) > 1 else 'single model'))
    xdf_dset = xdf_data.iloc[idx_test].copy()
    predict_func(np.argmax(test_prob, axis=1))
    values = metrics_func(['f1_macro', 'coh', 'acc'], ['avg'])
    final_test = scores(y_all[idx_test], np.argmax(test_prob, axis=1))
    print('Leaderboard formula (F1 + kappa)/2 on test: {:.4f}'.format(final_test['score']))
    log_experiment(experiment_row('final', best_cfg, final_info, final_test, SEED,
                                  {'fold': 'ensemble {}'.format(len(model_files)) if len(model_files) > 1 else ''}))
    print('Run logged in', out_file('experiments_{}.csv'.format(NICKNAME)), 'and', out_file('grid_{}.csv'.format(NICKNAME)))
    return values
# ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    main()
#------------------------------------------------------------------------------------------------------------------
