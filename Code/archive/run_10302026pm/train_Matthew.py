#------------------------------------------------------------------------------------------------------------------
import os
import csv
import time
import re
import argparse
import itertools
import zlib
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
10/30/2026 pm: FEATURE AUGMENTATION + TEST-TIME CROPS (suggestion 3) + LESION FEATURES (suggestion 4)
            - Diagnosis of the 10/30 run: train accuracy 0.996 vs validation 0.875, and 0.6M / 1.6M / 4.6M weights
              all gave the same validation score -> the network memorizes the training images; it needs more varied
              data and more informative features, not more capacity.
            - AUGMENTATION: N_AUG_VIEWS random views of every TRAINING image (random crop of 60-100% of the height and
              width, small brightness / contrast / saturation change) are computed once and cached. In every epoch
              each training image is shown as one of its 1 + N_AUG_VIEWS versions, chosen at random. Validation
              images are never augmented.
            - TEST-TIME AUGMENTATION (TTA): the prediction is the mean of the log-probabilities over TTA_VIEWS views
              of the image (whole image + 4 corner crops of 80%), the same views in train and test scripts.
              The model name ends with _t<TTA_VIEWS> so test_Matthew.py knows how many views to use.
            - NEW FEATURE GROUPS (FEATURE_VERSION 2, version 1 groups unchanged):
              'lab'  CIE L*a*b* colour histograms (brown / orange / yellow lesions vs green leaf)
              'tile' tile pooling: histograms of 4x4 tiles, maximum over the tiles for every bin, so a lesion
                     concentrated in one part of the image is not diluted by the rest
              (spatial pyramid / max pooling of local histograms: Lazebnik et al., 2006; Boureau et al., 2010;
               CIE Lab: CIE 15:2004)
            - GRID compares the feature sets with and without augmentation; TUNE_BIAS off (no gain on 10/30).
10/30/2026: K-FOLD ENSEMBLE + CLASS BIAS TUNING
            - GRID (unchanged) only chooses the settings, on the fixed train/validation split. GRID = {} skips it.
            - FINAL MODELS: the "train" rows are split into N_FOLDS stratified parts (k-fold cross-validation).
              Model k is trained on the other N_FOLDS-1 parts and early-stopped on part k, so every training image
              is used by N_FOLDS-1 of the models (before: the 20% validation part was never used for training).
              Saved as model_Matthew1.keras ... model_MatthewN.keras (exam section 4.8: ensembles in one submission).
            - ENSEMBLE: test_Matthew.py averages the log-probabilities of the models (geometric mean of the softmax).
            - CLASS BIAS: every training image has one "out-of-fold" prediction (from the model that did not train on
              it). A bias b_c per class is searched on these predictions so that argmax(log p + b) maximizes
              (F1 + kappa)/2, then added to the bias of the output Dense layer of every model:
              softmax(Wx + c + b). The models stay plain MLPs and need no extra file. With the geometric mean the
              ensemble argmax moves by exactly b, so the bias found out-of-fold is the one applied at test time.
            - Saved models hold the weights only (no Adam state): ~6 MB per file instead of ~19 MB.
            The "test" rows are never used for training or choosing; they are only reported.
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
            - ONE MODEL FILE: the best grid run (best seed of the best combination) is saved as model_Matthew.keras,
              overwriting the previous file (the exam submission takes a single model file).
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
##  'lab' CIE L*a*b* histograms (272)               'tile' max over 4x4 tiles of hs + exr + lbp histograms (403)
## 240-image test: all groups but 'thumb' was the best set (thumbnail = raw pixels again, it lowered the score)
FEATURES_V1 = ['hs', 'v', 'mom', 'exr', 'lbp', 'lbp2', 'lbp4', 'grad', 'ctr']      # the 10/30 set (904 features)
FEATURE_GROUPS = FEATURES_V1 + ['lab', 'tile']
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
DROPOUT_RATE = 0.4                 # after every hidden layer (0 = off). Works together with BatchNorm here
INPUT_DROPOUT = 0.0                # dropout on the normalized features themselves (0 = off)
WEIGHT_DECAY = 0.0                 # L2 penalty on every Dense kernel (0 = off)
## (hidden / dropout / weight decay = best combination of the 10/30 grid)
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

## ---- Feature-level augmentation (suggestion 3) ------------------------------------------------------------------
## N_AUG_VIEWS random views of every training image, computed once and cached in features_cache/ (first run: about
## N_AUG_VIEWS x 4 min). Each view: random crop of AUG_SCALE of the height and of the width at a random position, then
## a small random change of brightness, contrast and saturation (no hue change: colour is the main signal). The same
## image always gets the same views (seeded by its name), so the cache stays valid. 0 = no augmentation.
N_AUG_VIEWS = 4
AUG_SEED = 2026
AUG_SCALE = (0.6, 1.0)
AUG_BRIGHTNESS = 0.08        # random value in [-0.08, 0.08] added to the pixels (0..1)
AUG_CONTRAST = 0.15          # contrast factor in [0.85, 1.15]
AUG_SATURATION = 0.10        # saturation factor in [0.90, 1.10]

## ---- Test-time augmentation -------------------------------------------------------------------------------------
## prediction = mean of the log-probabilities over the first TTA_VIEWS views of TTA_CROPS (whole image + 4 corner crops
## of 80%); 1 = whole image only. Written in the model name (_t5): test_Matthew.py uses the same views.
## The views are computed for the training images too (cached), so the GRID ranks the combinations by their
## validation score WITH test-time augmentation, i.e. the way the submitted models predict.
TTA_VIEWS = 5

## ---- Grid of experiments -------------------------------------------------------------------------------------
## Every combination is trained on the same train/validation split (VAL_FRACTION, SEED) and logged to
## grid_<NICKNAME>.csv and experiments_<NICKNAME>.csv. The best mean validation (F1 + kappa)/2 wins.
## Allowed keys: the setting names above (HIDDEN_LAYERS, DROPOUT_RATE, INPUT_DROPOUT, WEIGHT_DECAY, USE_BATCHNORM,
## LEARNING_RATE, BATCH_SIZE, CLASS_WEIGHTS, LABEL_SMOOTHING, FEATURE_GROUPS, ACTIVATION, OPTIMIZER, N_AUG_VIEWS).
## Ranking: validation (F1 + kappa)/2 with TTA (TTA_VIEWS > 1); the epoch kept is chosen on the whole images.
## The test score is printed with and without TTA (information only).
## GRID = {} -> one run with the settings above.
## Reading the result (train accuracy is measured on the training part, inference mode):
##   train and validation both low and close   -> too simple / too regularized: wider, deeper, less dropout
##   train much higher than validation (>0.15) -> overfitting: more dropout / weight decay, or smaller
GRID = {
    'FEATURE_GROUPS': [FEATURES_V1, FEATURES_V1 + ['lab'], FEATURES_V1 + ['tile'], FEATURES_V1 + ['lab', 'tile']],
    'N_AUG_VIEWS': [0, 4],
}
## each combination is trained once per seed and the validation scores are averaged (differences below ~0.01 between
## two combinations are within the noise of one seed: use 2-3 seeds to separate close ones)
## The best combination is then used for the K-fold final models below (GRID = {} -> the settings above, no grid run)
GRID_SEEDS = [42]

## ---- Final models: K-fold cross-validation ensemble (suggestions 1 + 2) -----------------------------------------
## model_<NICKNAME>1.keras ... model_<NICKNAME><N_FOLDS>.keras; numbered files left over from a run with more folds
## are deleted, so the folder only holds the models of this run (test_Matthew.py loads every numbered file)
N_FOLDS = 5
CV_SEED = 42                 # split of the folds; model k uses seed CV_SEED + k (initial weights, shuffling)

## ---- Class bias tuning on the out-of-fold predictions (suggestion 5) --------------------------------------------
TUNE_BIAS = False            # 10/30 run: 0.8589 without, 0.8581 with -> off
BIAS_RANGE = 1.5             # each class bias is searched in [-BIAS_RANGE, +BIAS_RANGE] (in log-probability units)
BIAS_STEP = 0.05
BIAS_ROUNDS = 3              # passes over the classes (coordinate ascent), stops early when nothing improves

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
             'BATCH_SIZE', 'CLASS_WEIGHTS', 'LABEL_SMOOTHING', 'FEATURE_GROUPS', 'ACTIVATION', 'OPTIMIZER',
             'N_AUG_VIEWS')
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
## Version 2 (10/30/2026): new groups 'lab' and 'tile', test-time crops TTA_CROPS. The version 1 groups are computed
## exactly as in version 1, so models of both versions are accepted (FEATURE_VERSIONS_OK).
## Change FEATURE_VERSION whenever this block changes: test_Matthew.py refuses a model made with another version.
FEATURE_VERSION = 2
FEATURE_VERSIONS_OK = (1, 2)
FEATURE_ORDER = ('hs', 'v', 'mom', 'exr', 'lbp', 'lbp2', 'lbp4', 'grad', 'ctr', 'thumb', 'lab', 'tile')
FEATURE_SIZE = {'hs': 288, 'v': 16, 'mom': 12, 'exr': 64, 'lbp': 51, 'lbp2': 51, 'lbp4': 51, 'grad': 16,
                'ctr': 355, 'thumb': 768, 'lab': 272, 'tile': 403}
SQRT_GROUPS = ('hs', 'v', 'exr', 'lbp', 'lbp2', 'lbp4', 'grad', 'ctr', 'lab', 'tile')
TILE_GRID = 4                # 'tile': the image is cut into TILE_GRID x TILE_GRID tiles
## Test-time augmentation views: (crop scale, vertical position, horizontal position), position 0 = top / left,
## 1 = bottom / right. View 0 is the whole image; the model name says how many views are used (..._t5).
TTA_CROPS = ((1.0, 0.5, 0.5), (0.8, 0.0, 0.0), (0.8, 0.0, 1.0), (0.8, 1.0, 0.0), (0.8, 1.0, 1.0))


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


def _value_codes(x, n, lo, hi):
    '''
        bin index 0..n-1 of real values in [lo, hi] (values outside go to the first / last bin)
    '''
    idx = tf.cast(tf.floor((x - lo) / (hi - lo) * n), tf.int32)
    return tf.clip_by_value(idx, 0, n - 1)


def _value_hist(x, n, lo, hi):
    '''
        normalized histogram of real values in [lo, hi] (values outside go to the first / last bin)
    '''
    return _counts(_value_codes(x, n, lo, hi), n)


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


def _lbp_codes(gray):
    '''
        local binary pattern: every pixel is compared with its 8 neighbours (1 if neighbour >= centre)
        -> an 8-bit code describing the local texture -> its flip/turn class 0..50, for every inner pixel
    '''
    c = gray[1:-1, 1:-1]
    nb = [gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:], gray[1:-1, 2:],
          gray[2:, 2:], gray[2:, 1:-1], gray[2:, :-2], gray[1:-1, :-2]]
    code = tf.add_n([tf.cast(n >= c, tf.int32) * (2 ** i) for i, n in enumerate(nb)])
    return tf.gather(tf.constant(LBP_CLASS), code)


def _lbp(gray):
    '''
        histogram of the 51 flip/turn classes of the LBP codes
    '''
    return _counts(_lbp_codes(gray), 51)


def _shrink(gray2d, factor):
    '''
        gray image [H, W] reduced by an integer factor (area average)
    '''
    hw = tf.shape(gray2d)[:2] // factor
    return tf.image.resize(gray2d[..., None], hw, antialias=True)[..., 0]


def _rgb_to_lab(img):
    '''
        sRGB in [0, 1] -> CIE L*a*b* (D65 white): L* 0..100 lightness, a* green (-) / red (+), b* blue (-) / yellow (+)
    '''
    lin = tf.where(img <= 0.04045, img / 12.92, tf.pow((img + 0.055) / 1.055, 2.4))
    m = tf.constant([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], tf.float32)
    xyz = tf.tensordot(lin, tf.transpose(m), axes=1) / tf.constant([0.95047, 1.0, 1.08883], tf.float32)
    f = tf.where(xyz > 0.008856, tf.pow(tf.maximum(xyz, 1e-12), 1.0 / 3.0), 7.787 * xyz + 16.0 / 116.0)
    return 116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])


def _tile_max(idx, n):
    '''
        idx [H, W] integer codes 0..n-1. The map is cut into TILE_GRID x TILE_GRID tiles, the normalized histogram of
        every tile is computed, and for every bin the MAXIMUM over the tiles is kept (n values)
    '''
    h, w = tf.shape(idx)[0], tf.shape(idx)[1]
    tile = (tf.range(h) * TILE_GRID // h)[:, None] * TILE_GRID + (tf.range(w) * TILE_GRID // w)[None, :]
    k = TILE_GRID * TILE_GRID
    c = tf.math.bincount(tf.reshape(tile * n + idx, [-1]), minlength=k * n, maxlength=k * n, dtype=tf.float32)
    c = tf.reshape(c, [k, n])
    c = c / tf.maximum(tf.reduce_sum(c, axis=1, keepdims=True), 1.0)
    return tf.reduce_max(c, axis=0)


def decode_image(file_path, ratio):
    '''
        reads a JPEG and decodes it directly at 1/ratio of its size (4000x2672 -> 500x334 with ratio 8)
        returns float32 [H, W, 3] in [0, 1]
    '''
    img = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3, ratio=ratio)
    return tf.cast(img, tf.float32) / 255.0


def tta_crop(img, view):
    '''
        test-time view number `view` (python int) of TTA_CROPS: the whole image or a fixed crop of it
    '''
    scale, py, px = TTA_CROPS[view]
    if scale >= 1.0:
        return img
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    ch = tf.cast(tf.round(scale * tf.cast(h, tf.float32)), tf.int32)
    cw = tf.cast(tf.round(scale * tf.cast(w, tf.float32)), tf.int32)
    y0 = tf.cast(tf.round(py * tf.cast(h - ch, tf.float32)), tf.int32)
    x0 = tf.cast(tf.round(px * tf.cast(w - cw, tf.float32)), tf.int32)
    return img[y0:y0 + ch, x0:x0 + cw]


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
    if 'lab' in need:
        # CIE L*a*b*: brown, orange and yellow lesions are further from healthy green on a* / b* than in HSV.
        # 16 lightness bins + joint 16 x 16 histogram of a* (-48..32) and b* (-16..80)   -> 272
        lab_l, lab_a, lab_b = _rgb_to_lab(img)
        ab = _value_codes(lab_a, 16, -48.0, 32.0) * 16 + _value_codes(lab_b, 16, -16.0, 80.0)
        out['lab'] = tf.concat([_value_hist(lab_l, 16, 0.0, 100.0), _counts(ab, 256)], 0)
    if 'tile' in need:
        # TILE POOLING: hue x saturation (288), excess red (32), excess green (32) and LBP (51) histograms of every
        # tile; for every bin the maximum over the tiles. A lesion concentrated in one part of the image stands out
        # instead of being diluted by the rest of the image, and it still counts the same wherever it is   -> 403
        hsv_t = tf.image.rgb_to_hsv(img)
        h_c = tf.minimum(tf.cast(hsv_t[..., 0] * 36, tf.int32), 35)
        s_c = tf.minimum(tf.cast(hsv_t[..., 1] * 8, tf.int32), 7)
        r_t, g_t, b_t = img[..., 0], img[..., 1], img[..., 2]
        out['tile'] = tf.concat([_tile_max(h_c * 8 + s_c, 288),
                                 _tile_max(_value_codes(r_t - g_t, 32, -0.4, 0.4), 32),
                                 _tile_max(_value_codes(2 * g_t - r_t - b_t, 32, -0.4, 0.8), 32),
                                 _tile_max(_lbp_codes(gray), 51)], 0)

    return tf.concat([tf.sqrt(out[k]) if k in SQRT_GROUPS else out[k] for k in groups], axis=0)
#------------------------------------------------------------------------------------------------------------------

def augment_image(img, seed):
    '''
        one random training view of img [H, W, 3] in [0, 1]: random crop (AUG_SCALE of the height and of the width,
        random position), then random brightness, contrast and saturation. seed: int64 [2] (stateless: the same seed
        always gives the same view)
    '''
    s = tf.cast(seed, tf.int64)
    sub = lambda k: s + tf.constant([0, 1], tf.int64) * k            # a different seed for every random choice
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    scale = tf.random.stateless_uniform([], sub(1), AUG_SCALE[0], AUG_SCALE[1])
    ch = tf.maximum(tf.cast(tf.round(scale * tf.cast(h, tf.float32)), tf.int32), 8)
    cw = tf.maximum(tf.cast(tf.round(scale * tf.cast(w, tf.float32)), tf.int32), 8)
    out = tf.image.stateless_random_crop(img, tf.stack([tf.minimum(ch, h), tf.minimum(cw, w), 3]), sub(2))
    if AUG_BRIGHTNESS > 0:
        out = tf.image.stateless_random_brightness(out, AUG_BRIGHTNESS, sub(3))
    if AUG_CONTRAST > 0:
        out = tf.image.stateless_random_contrast(out, 1.0 - AUG_CONTRAST, 1.0 + AUG_CONTRAST, sub(4))
    if AUG_SATURATION > 0:
        out = tf.image.stateless_random_saturation(tf.clip_by_value(out, 0.0, 1.0), 1.0 - AUG_SATURATION,
                                                   1.0 + AUG_SATURATION, sub(5))
    return tf.clip_by_value(out, 0.0, 1.0)
#------------------------------------------------------------------------------------------------------------------

def extract_features(paths, seeds, groups, ratio, view):
    '''
        features of every image in paths (same order), as a float32 array [len(paths), n_features]
        view: 'orig' (whole image), 'aug<j>' (random training view j, seeds [len(paths), 2]),
              'tta<j>' (test-time crop j of TTA_CROPS)

        Parallel "data generator": tf.data runs image_features on EXTRACT_PARALLEL images at the same time
        (0 = AUTOTUNE = one per CPU core). The JPEG decoding and the histogram ops run in TensorFlow's C++ thread
        pool, outside the Python GIL, so all the vCPUs of the instance work together (measured: 1.6x faster with
        2 cores than with 1). deterministic=True keeps the output in the order of paths.
    '''
    def one(f, seed):
        img = decode_image(f, ratio)
        if view.startswith('aug'):
            img = augment_image(img, seed)
        elif view.startswith('tta'):
            img = tta_crop(img, int(view[3:]))
        return image_features(img, groups)

    par = AUTOTUNE if EXTRACT_PARALLEL <= 0 else EXTRACT_PARALLEL
    ds = tf.data.Dataset.from_tensor_slices((list(paths), np.asarray(seeds, np.int64)))
    ds = ds.map(one, num_parallel_calls=par, deterministic=True)
    ds = ds.batch(EXTRACT_BATCH).prefetch(AUTOTUNE)

    out, done, t0 = [], 0, time.time()
    for xb in ds:
        out.append(xb.numpy())
        done += xb.shape[0]
        if done % (EXTRACT_BATCH * 16) < EXTRACT_BATCH or done == len(paths):
            rate = done / max(time.time() - t0, 1e-9)
            print('  features ({}): {:6d} / {} images  ({:.0f} images/s, {:.1f} min left)'.format(
                view, done, len(paths), rate, (len(paths) - done) / rate / 60))
    return np.concatenate(out, axis=0).astype(np.float32)
#------------------------------------------------------------------------------------------------------------------

def load_feature_cache(ids, view='orig'):
    '''
        features of ALL groups (FEATURE_ORDER) of one view of the image ids, from features_cache/ when available.
        Only the images missing from the cache are computed.
          'orig'   whole image (all rows; first run ~14,700 images)
          'aug<j>' random training view j (training rows). The file name holds a code of the AUG_* settings, so
                   changing them makes new views instead of reusing old ones
          'tta<j>' test-time crop j (all rows: validation with TTA in the grid / folds, test rows)
        Returns [len(ids), sum of all FEATURE_SIZE]
    '''
    folder = os.path.join(SCRIPT_DIR, CACHE_DIR)
    os.makedirs(folder, exist_ok=True)
    tag = view
    if view.startswith('aug'):
        tag += '_{:08x}'.format(zlib.crc32(repr((AUG_SEED, tuple(AUG_SCALE), AUG_BRIGHTNESS, AUG_CONTRAST,
                                                 AUG_SATURATION)).encode()))
    file_name = os.path.join(folder, 'features_r{}_v{}_{}.npz'.format(FEATURE_RATIO, FEATURE_VERSION, tag))
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
        print('Computing the features of {} images, view {} (ratio {}, all groups, cached in {}) ...'.format(
            len(missing), view, FEATURE_RATIO, file_name))
        j = int(view[3:]) if view[3:].isdigit() else 0
        seeds = [[AUG_SEED * 1000 + j, zlib.crc32(i.encode())] for i in missing]   # same image -> same view
        X_new = extract_features([DATA_DIR + i for i in missing], seeds, list(FEATURE_ORDER), FEATURE_RATIO, view)
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
    return ('feat={} | aug={} | hidden={} | dropout={} | in_drop={} | wd={} | bn={} | lr={} | batch={} | cw={} '
            '| ls={}').format(
        '-'.join(cfg['FEATURE_GROUPS']), cfg['N_AUG_VIEWS'], cfg['HIDDEN_LAYERS'], cfg['DROPOUT_RATE'],
        cfg['INPUT_DROPOUT'], cfg['WEIGHT_DECAY'], cfg['USE_BATCHNORM'], cfg['LEARNING_RATE'], cfg['BATCH_SIZE'],
        cfg['CLASS_WEIGHTS'], cfg['LABEL_SMOOTHING'])
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
    if int(N_FOLDS) < 2:
        raise ValueError('N_FOLDS must be 2 or more')
    if not 1 <= int(TTA_VIEWS) <= len(TTA_CROPS):
        raise ValueError('TTA_VIEWS must be between 1 and {}'.format(len(TTA_CROPS)))
    if not 0.0 < AUG_SCALE[0] <= AUG_SCALE[1] <= 1.0:
        raise ValueError('AUG_SCALE must be (low, high) with 0 < low <= high <= 1')
    if TUNE_BIAS and (BIAS_RANGE <= 0 or BIAS_STEP <= 0 or BIAS_ROUNDS < 1):
        raise ValueError('BIAS_RANGE, BIAS_STEP must be > 0 and BIAS_ROUNDS >= 1')
    if EARLY_STOP_PATIENCE > 0 and LR_SCHEDULE == 'plateau' and LR_PATIENCE >= EARLY_STOP_PATIENCE:
        print('WARNING: LR_PATIENCE >= EARLY_STOP_PATIENCE, the training stops before the learning rate is lowered')
    for cfg in grid_configs():
        if not 0.0 <= cfg['DROPOUT_RATE'] < 1.0 or not 0.0 <= cfg['INPUT_DROPOUT'] < 1.0:
            raise ValueError('DROPOUT_RATE and INPUT_DROPOUT must be in [0, 1)')
        if cfg['CLASS_WEIGHTS'] not in ('balanced', 'sqrt', 'none'):
            raise ValueError("CLASS_WEIGHTS must be 'balanced', 'sqrt' or 'none'")
        if len(cfg['HIDDEN_LAYERS']) == 0:
            raise ValueError('HIDDEN_LAYERS needs at least one layer')
        if int(cfg['N_AUG_VIEWS']) < 0:
            raise ValueError('N_AUG_VIEWS must be 0 or more')
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

def save_model(models):
    '''
         receives the models [(file name, model), ...] and prints their summaries into ONE .txt file
         (same text as test_Matthew.py writes)
    '''
    os.environ['COLUMNS'] = '80'   # same summary width as an automated run (no terminal)
    with open(out_file('summary_{}.txt'.format(NICKNAME)), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('Ensemble of {} model(s), prediction = geometric mean of their softmax outputs\n\n'.format(len(models)))
        for name, model in models:
            fh.write('==== {} ====\n'.format(name))
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x, **kwargs: fh.write(x + '\n'))
            fh.write('\n')
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
        MLP_Matthew_feat_r8_g-hs-v-mom-exr-lbp-lbp2-lbp4-grad-ctr-lab-tile_v2_t5
            r8 = JPEG decode ratio, g-... = feature groups in FEATURE_ORDER, v2 = FEATURE_VERSION,
            t5 = number of test-time views (TTA_VIEWS)
    '''
    return 'MLP_{}_feat_r{}_g-{}_v{}_t{}'.format(NICKNAME, FEATURE_RATIO, '-'.join(cfg['FEATURE_GROUPS']),
                                                 FEATURE_VERSION, TTA_VIEWS)
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

    model.add(tf.keras.layers.Dense(OUTPUTS_a, activation='softmax', kernel_regularizer=reg, name='output_softmax'))

    compile_model(model, cfg)
    return model
#------------------------------------------------------------------------------------------------------------------

def compile_model(model, cfg):
    model.compile(optimizer=make_optimizer(cfg),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=cfg['LABEL_SMOOTHING']),
                  metrics=[tf.keras.metrics.CategoricalCrossentropy(name='categorical_ce')])
#------------------------------------------------------------------------------------------------------------------

def weights_only_copy(model):
    '''
        same network (same name, layers and weights) rebuilt from its config, without the optimizer state:
        the .keras file is ~3x smaller (~6 MB instead of ~19 MB, no Adam moments) and loads without warnings.
        Predictions are identical.
    '''
    copy = model.__class__.from_config(model.get_config())
    copy.set_weights(model.get_weights())
    return copy
#------------------------------------------------------------------------------------------------------------------

def predict_views_logp(model, X_views):
    '''
        X_views [T, n, d]: the same n images seen in T views -> mean over the views of the log-probabilities [n, K]
        (test-time augmentation; T = 1 -> the plain log-probabilities)
    '''
    return np.mean([log_softmax_np(predict_np(model, Xv)) for Xv in X_views], axis=0)
#------------------------------------------------------------------------------------------------------------------

def log_softmax_np(prob):
    '''
        log of softmax outputs (floored, so a probability of exactly 0 does not give -inf)
    '''
    return np.log(np.clip(prob, 1e-12, 1.0)).astype(np.float64)
#------------------------------------------------------------------------------------------------------------------

def fast_score(y_true, y_pred, k):
    '''
        (macro F1 + Cohen kappa) / 2 from the confusion matrix (same values as sklearn, ~100x faster: the bias
        search calls it a few thousand times). Like sklearn, macro F1 averages over the classes that appear in
        y_true or y_pred.
    '''
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k).astype(np.float64)
    n, tp = cm.sum(), np.diag(cm)
    rows, cols = cm.sum(axis=1), cm.sum(axis=0)
    present = (rows + cols) > 0
    f1 = np.mean(2.0 * tp[present] / (rows + cols)[present])
    po, pe = tp.sum() / n, np.sum(rows * cols) / (n * n)
    kappa = (po - pe) / (1.0 - pe) if pe < 1.0 else 0.0
    return float((f1 + kappa) / 2.0)
#------------------------------------------------------------------------------------------------------------------

def tune_class_bias(log_prob, y_true, k):
    '''
        coordinate ascent on a bias b_c per class: prediction = argmax(log p + b), objective = (F1 + kappa)/2.
        b_0 stays 0 (adding the same number to every class changes nothing). Each pass tries every value of the grid
        for each class in turn and keeps the best; stops when a pass improves nothing.
        returns b, score before, score after
    '''
    b = np.zeros(k)
    start = best = fast_score(y_true, np.argmax(log_prob, axis=1), k)
    values = np.round(np.arange(-BIAS_RANGE, BIAS_RANGE + 1e-9, BIAS_STEP), 6)
    for r in range(BIAS_ROUNDS):
        improved = False
        for c in range(1, k):
            for v in values:
                trial = b.copy()
                trial[c] = v
                s = fast_score(y_true, np.argmax(log_prob + trial, axis=1), k)
                if s > best + 1e-9:
                    best, b, improved = s, trial, True
        print('  bias pass {}: score {:.4f}  b = [{}]'.format(r + 1, best, ' '.join('{:+.2f}'.format(x) for x in b)))
        if not improved:
            break
    return b, start, best
#------------------------------------------------------------------------------------------------------------------

def add_output_bias(model, b):
    '''
        adds b to the bias of the output Dense layer: softmax(Wx + c + b), i.e. log p + b up to a constant per image
    '''
    last = model.layers[-1]
    if not isinstance(last, tf.keras.layers.Dense) or len(last.get_weights()) != 2:
        raise ValueError('The last layer must be a Dense layer with a bias, found {}'.format(last.name))
    W, c = last.get_weights()
    last.set_weights([W, (c + b).astype(c.dtype)])
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
        X_tr  : [V, n, d] the n training images in V versions (version 0 = whole image, 1.. = augmented views)
        X_val : [m, d] whole validation images (never augmented)
        y_*   : class index
        In every epoch each training image is shown in ONE of its V versions, chosen at random.
        returns model (weights of the best epoch), history, info dict
    '''
    tf.keras.utils.set_random_seed(seed)
    n_views, n, d = X_tr.shape

    class_w = compute_class_weights(y_tr, OUTPUTS_a, cfg['CLASS_WEIGHTS'])

    # z-score statistics of the TRAINING part only (all its versions), stored in the first layer of the model
    flat = X_tr.reshape(-1, d)
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), MIN_FEATURE_STD)
    del flat

    y_onehot = to_categorical(y_tr, num_classes=OUTPUTS_a).astype(np.float32)
    w = class_w[y_tr]
    if n_views == 1:
        ds = tf.data.Dataset.from_tensor_slices((X_tr[0], y_onehot, w))
        ds = ds.shuffle(n, seed=seed, reshuffle_each_iteration=True).batch(cfg['BATCH_SIZE'])
    else:
        # [n, V, d]: one row per image with all its versions; after batching, one version per image is picked
        ds = tf.data.Dataset.from_tensor_slices((np.ascontiguousarray(np.transpose(X_tr, (1, 0, 2))), y_onehot, w))
        ds = ds.shuffle(n, seed=seed, reshuffle_each_iteration=True).batch(cfg['BATCH_SIZE'])
        ds = ds.map(lambda x, y, wt: (tf.gather(x, tf.random.uniform([tf.shape(x)[0]], 0, n_views, tf.int32),
                                                batch_dims=1), y, wt), num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)

    best = BestWeights(CHECKPOINT_MONITOR)
    callbacks = [tf.keras.callbacks.TerminateOnNaN(),
                 ValidationMetrics(X_val, y_val, class_w, X_tr[0], y_tr, label), best]
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
           'n_aug_views': cfg['N_AUG_VIEWS'], 'tta_views': TTA_VIEWS,
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
    run_grid = len(GRID) > 0
    if run_grid:
        print('Grid: {} combinations x {} seed(s) = {} runs, then {} final models ({}-fold) with the best one'.format(
            len(configs), len(GRID_SEEDS), len(configs) * len(GRID_SEEDS), N_FOLDS, N_FOLDS))
    else:
        print('GRID = {{}}: no grid, {} final models ({}-fold) with the settings at the top'.format(N_FOLDS, N_FOLDS))

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
    ids = xdf_data['id'].tolist()
    X_all = load_feature_cache(ids, 'orig')
    y_all = xdf_data['target'].map(class_to_idx).to_numpy()

    idx_train = np.where(is_train)[0]
    idx_test = np.where(is_test)[0]
    y_train, y_test = y_all[idx_train], y_all[idx_test]

    ## training rows: whole image + the augmented views (as many as the largest N_AUG_VIEWS used)   [V, n_train, D]
    n_aug = max(int(c['N_AUG_VIEWS']) for c in configs)
    train_ids = [ids[i] for i in idx_train]
    X_train_views = np.stack([X_all[idx_train]] + [load_feature_cache(train_ids, 'aug{}'.format(j))
                                                    for j in range(1, n_aug + 1)])
    ## all rows: whole image + the TTA crops; split into training rows (validation with TTA) and test rows
    tta_all = [X_all] + [load_feature_cache(ids, 'tta{}'.format(j)) for j in range(1, TTA_VIEWS)]
    X_train_tta = np.stack([t[idx_train] for t in tta_all])      # [TTA_VIEWS, n_train, D]
    X_test_views = np.stack([t[idx_test] for t in tta_all])      # [TTA_VIEWS, n_test, D]
    del X_all, tta_all
    print('Training images: {} ({} version(s) each for training, {} TTA view(s) for validation)   '
          'Test images: {} ({} TTA view(s))'.format(len(idx_train), X_train_views.shape[0], X_train_tta.shape[0],
                                                    len(idx_test), X_test_views.shape[0]))
    rank_key = 'val_score_tta' if TTA_VIEWS > 1 else 'val_score'

    def test_scores(model, Xt):
        ''' (score whole image only, score with TTA) of one model on the test rows (information only) '''
        lp1 = log_softmax_np(predict_np(model, Xt[0]))
        lpt = predict_views_logp(model, Xt) if len(Xt) > 1 else lp1
        return lp1, lpt, scores(y_test, np.argmax(lp1, axis=1)), scores(y_test, np.argmax(lpt, axis=1))

    #--------------------------------------------------------------------------------------------------------------
    # GRID (chooses the settings only; its models are not saved). Validation: whole images only.
    if run_grid:
        ## Fixed split for the grid: stratified, same seed -> every combination sees the same images
        ## (positions inside the training rows; same split as before)
        pos_tr, pos_val = train_test_split(np.arange(len(idx_train)), test_size=VAL_FRACTION, stratify=y_train,
                                           random_state=SEED)
        pos_tr, pos_val = np.sort(pos_tr), np.sort(pos_val)
        print('Grid split - training images: {}   validation images: {}   test images: {}'.format(
            len(pos_tr), len(pos_val), len(idx_test)))
        compute_class_weights(y_train[pos_tr], OUTPUTS_a, CLASS_WEIGHTS, verbose=True)

        grid_rows, best = [], None
        for c, cfg in enumerate(configs):
            cols = feature_columns(cfg['FEATURE_GROUPS'])
            Xc = X_train_views[:1 + int(cfg['N_AUG_VIEWS'])][:, :, cols]
            Xv = X_train_tta[:, pos_val][:, :, cols]                    # validation images, TTA views
            Xt = X_test_views[:, :, cols]
            print('\n==== Grid {}/{}: {} ({} features) ===='.format(c + 1, len(configs), config_text(cfg), len(cols)))
            runs = []
            for seed in GRID_SEEDS:
                model, history, info = train_one(cfg, Xc[:, pos_tr], y_train[pos_tr], Xc[0, pos_val],
                                                 y_train[pos_val], seed,
                                                 label='[{}/{} s{}]'.format(c + 1, len(configs), seed))
                val_tta = scores(y_train[pos_val], np.argmax(predict_views_logp(model, Xv), axis=1))
                _, _, test, test_tta = test_scores(model, Xt)
                print('  validation with {} TTA views: score {:.4f}'.format(len(Xv), val_tta['score']))
                print('  test (for information only, not used to choose): score {:.4f} | with {} TTA views {:.4f}'.format(
                    test['score'], len(Xt), test_tta['score']))
                log_experiment(experiment_row('grid', cfg, info, test, seed, extra={
                    'val_score_tta': val_tta['score'], 'test_score_tta': test_tta['score']}))
                runs.append(val_tta['score'] if rank_key == 'val_score_tta' else info['val']['score'])
                grid_rows.append({'combination': c + 1, 'seed': seed, 'config': config_text(cfg),
                                  'params': model.count_params(), 'epochs_run': info['epochs_run'],
                                  'best_epoch': info['best_epoch'], 'train_acc': info['train_accuracy'],
                                  'val_acc': info['val']['acc'], 'val_score': info['val']['score'],
                                  'val_score_tta': val_tta['score'], 'test_score': test['score'],
                                  'test_score_tta': test_tta['score']})
                del model, history

            mean_val = float(np.mean(runs))
            if best is None or mean_val > best['mean_val']:
                best = {'cfg': cfg, 'mean_val': mean_val, 'combination': c + 1}
            print('  combination {}: mean {} {:.4f} | leader: combination {} ({:.4f})'.format(
                c + 1, rank_key, mean_val, best['combination'], best['mean_val']))
            pd.DataFrame(grid_rows).to_csv(out_file('grid_{}.csv'.format(NICKNAME)), index=False)
            del Xc, Xv

        ranking = pd.DataFrame(grid_rows).groupby(['combination', 'config'], sort=False)[
            ['params', 'train_acc', 'val_acc', 'val_score', 'val_score_tta', 'test_score', 'test_score_tta']].mean()
        ranking = ranking.sort_values(rank_key, ascending=False)
        print('\n================ GRID RANKING (mean over seeds, sorted by {}) ================'.format(rank_key))
        print('train_acc much higher than val_acc -> overfitting; both low and close -> underfitting')
        with pd.option_context('display.width', 250, 'display.max_colwidth', 160):
            print(ranking.round(4).to_string())
        best_cfg = best['cfg']
        print('\nBest combination {}: {}'.format(best['combination'], config_text(best_cfg)))
    else:
        best_cfg = configs[0]

    cols = feature_columns(best_cfg['FEATURE_GROUPS'])
    Xb = X_train_views[:1 + int(best_cfg['N_AUG_VIEWS'])][:, :, cols]      # [V, n_train, d]
    Xbv = X_train_tta[:, :, cols]                                          # [TTA_VIEWS, n_train, d]
    Xbt = X_test_views[:, :, cols]                                         # [TTA_VIEWS, n_test, d]
    del X_train_views, X_train_tta

    #--------------------------------------------------------------------------------------------------------------
    # FINAL MODELS: K-fold cross-validation. Model k: trained on the other folds, early-stopped on fold k.
    print('\n================ FINAL MODELS: {}-fold cross-validation ================'.format(N_FOLDS))
    print('Settings: {}'.format(config_text(best_cfg)))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    oof_logp = np.zeros((len(idx_train), OUTPUTS_a), np.float64)     # out-of-fold log-probabilities, whole images
    oof_logp_tta = np.zeros((len(idx_train), OUTPUTS_a), np.float64)  # out-of-fold, mean over the TTA views
    test_logp, test_logp_tta, models, fold_infos = [], [], [], []
    for k, (pos_tr, pos_val) in enumerate(skf.split(np.zeros(len(idx_train)), y_train), start=1):
        print('\n---- Fold {}/{}: {} training images, {} validation images ----'.format(
            k, N_FOLDS, len(pos_tr), len(pos_val)))
        model, history, info = train_one(best_cfg, Xb[:, pos_tr], y_train[pos_tr], Xb[0, pos_val], y_train[pos_val],
                                         CV_SEED + k, label='[fold {}/{}]'.format(k, N_FOLDS))
        if k == 1:
            early_stopping_report(history, 'Fold 1 of {}: {}'.format(N_FOLDS, config_text(best_cfg)))
        oof_logp[pos_val] = log_softmax_np(predict_np(model, Xb[0, pos_val]))
        oof_logp_tta[pos_val] = predict_views_logp(model, Xbv[:, pos_val])
        print('  validation with {} TTA views: score {:.4f}'.format(
            len(Xbv), scores(y_train[pos_val], np.argmax(oof_logp_tta[pos_val], axis=1))['score']))
        lp1, lpt, test_k, test_k_tta = test_scores(model, Xbt)
        test_logp.append(lp1)
        test_logp_tta.append(lpt)
        print('  test of this model alone (information only): score {:.4f} | with {} TTA views {:.4f}'.format(
            test_k['score'], len(Xbt), test_k_tta['score']))
        log_experiment(experiment_row('cv_fold{}'.format(k), best_cfg, info, test_k, CV_SEED + k,
                                      extra={'test_score_tta': test_k_tta['score']}))
        models.append(model)
        fold_infos.append(info)
        del history

    #--------------------------------------------------------------------------------------------------------------
    # CLASS BIAS on the out-of-fold predictions (every training image predicted by the model that did not train on it)
    ## with TTA, as the submitted models predict
    oof_single = scores(y_train, np.argmax(oof_logp, axis=1))
    oof_before = scores(y_train, np.argmax(oof_logp_tta, axis=1))
    print('\n================ OUT-OF-FOLD + CLASS BIAS ================')
    print('Out-of-fold ({} training images): whole image score {:.4f} | with {} TTA views: f1 {:.4f} kappa {:.4f} '
          'acc {:.4f} score {:.4f}'.format(len(idx_train), oof_single['score'], TTA_VIEWS, oof_before['f1_macro'],
                                           oof_before['kappa'], oof_before['acc'], oof_before['score']))
    if TUNE_BIAS:
        bias, s0, s1 = tune_class_bias(oof_logp_tta, y_train, OUTPUTS_a)
        print('Bias per class: {}'.format('  '.join('{} {:+.2f}'.format(c, x) for c, x in zip(class_names, bias))))
        print('Out-of-fold score {:.4f} -> {:.4f} (measured on the images it was tuned on: the real gain is smaller)'.format(
            s0, s1))
    else:
        bias = np.zeros(OUTPUTS_a)
        print('TUNE_BIAS = False: no bias')
    oof_after = scores(y_train, np.argmax(oof_logp_tta + bias, axis=1))

    ens_logp = np.mean(test_logp, axis=0)
    ens_logp_tta = np.mean(test_logp_tta, axis=0)
    t_plain = scores(y_test, np.argmax(ens_logp, axis=1))
    t_tta = scores(y_test, np.argmax(ens_logp_tta, axis=1))
    t_final = scores(y_test, np.argmax(ens_logp_tta + bias, axis=1))
    print('Test (information only, not used to choose): ensemble {:.4f} | ensemble + {} TTA views {:.4f}{}'.format(
        t_plain['score'], TTA_VIEWS, t_tta['score'],
        ' | + bias {:.4f}'.format(t_final['score']) if TUNE_BIAS else ''))

    #--------------------------------------------------------------------------------------------------------------
    # SAVE: bias added to the output layer, weights only (small files), model_<NICKNAME>1..K.keras
    print('\n================ SAVING ================')
    old = [f for f in os.listdir(OUT_DIR) if re.fullmatch(r'model_{}(\d+)\.keras'.format(re.escape(NICKNAME)), f)
           and int(re.fullmatch(r'model_{}(\d+)\.keras'.format(re.escape(NICKNAME)), f).group(1)) > N_FOLDS]
    for f in old:
        os.remove(out_file(f))
        print('Deleted {} (left over from a run with more folds)'.format(f))
    if os.path.exists(out_file('model_{}.keras'.format(NICKNAME))):
        print('NOTE: model_{0}.keras (single model of an older run) is still in the folder. test_{0}.py uses the '
              'numbered files when they exist, so do not submit the old file with them.'.format(NICKNAME))

    model_files = []
    for k, model in enumerate(models, start=1):
        add_output_bias(model, bias)
        f = out_file('model_{}{}.keras'.format(NICKNAME, k))
        weights_only_copy(model).save(f)
        model_files.append(f)
        print('Saved {} ({:.1f} MB): best epoch {}, fold validation score {:.4f}'.format(
            os.path.basename(f), os.path.getsize(f) / 1e6, fold_infos[k - 1]['best_epoch'],
            fold_infos[k - 1]['val']['score']))

    #--------------------------------------------------------------------------------------------------------------
    # TEST: the SAVED models, combined as test_Matthew.py does (mean of log-probabilities) -> results_<NICKNAME>.xlsx
    print('\n================ TEST (saved models, as test_{}.py) ================'.format(NICKNAME))
    saved = [tf.keras.models.load_model(f) for f in model_files]
    save_model([(os.path.basename(f), m) for f, m in zip(model_files, saved)])
    saved_logp = np.mean([predict_views_logp(m, Xbt) for m in saved], axis=0)     # models x TTA views
    pred = np.argmax(saved_logp, axis=1)
    same = float(np.mean(pred == np.argmax(ens_logp_tta + bias, axis=1)))
    print('Check: saved models give the same predictions as the ensemble in memory for {:.2%} of the test images'.format(
        same))

    xdf_dset = xdf_data.iloc[idx_test].copy()
    predict_func(pred)
    values = metrics_func(['f1_macro', 'coh', 'acc'], ['avg'])
    final_test = scores(y_test, pred)
    print('Leaderboard formula (F1 + kappa)/2 on test: {:.4f}'.format(final_test['score']))

    final_info = {'epochs_run': float(np.mean([i['epochs_run'] for i in fold_infos])),
                  'best_epoch': float(np.mean([i['best_epoch'] for i in fold_infos])), 'final_lr': np.nan,
                  'train_accuracy': float(np.mean([i['train_accuracy'] for i in fold_infos])),
                  'val': {'f1_macro': oof_after['f1_macro'], 'kappa': oof_after['kappa'], 'acc': oof_after['acc'],
                          'score': oof_after['score']}}
    log_experiment(experiment_row('final_ensemble', best_cfg, final_info, final_test, CV_SEED, extra={
        'n_models': N_FOLDS, 'oof_score_no_bias': oof_before['score'], 'oof_score_no_tta': oof_single['score'],
        'test_score_no_tta': t_plain['score'],
        'test_score_tta': t_tta['score'],
        'bias': ' '.join('{:+.2f}'.format(x) for x in bias)}))
    print('Run logged in', out_file('experiments_{}.csv'.format(NICKNAME)),
          ('and ' + out_file('grid_{}.csv'.format(NICKNAME))) if run_grid else '')
    print('Submit: test_{0}.py, {1}, summary_{0}.txt, results_{0}.xlsx'.format(
        NICKNAME, ', '.join(os.path.basename(f) for f in model_files)))
    return values
# ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    main()
#------------------------------------------------------------------------------------------------------------------
