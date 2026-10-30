#------------------------------------------------------------------------------------------------------------------
import os
import sys
import argparse
import importlib
import site
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
#------------------------------------------------------------------------------------------------------------------
## Additional package: openpyxl (used by pandas to read and write .xlsx files).
## Installed only if it is missing, with the same python that runs this script.
## The second command is for Ubuntu 23.04+ images that refuse plain pip installs ("externally-managed-environment").
## sudo -n never waits for a password (an automated run would hang); the last command needs no sudo at all.
os.system('"{0}" -c "import openpyxl" 2>/dev/null || sudo -n "{0}" -m pip install -q openpyxl || '
          'sudo -n "{0}" -m pip install -q --break-system-packages openpyxl || '
          '"{0}" -m pip install -q --user openpyxl || '
          '"{0}" -m pip install -q --user --break-system-packages openpyxl'.format(sys.executable))
def test():
    '''
        Usage (Exam1 section 5):
            python3 test_Matthew.py --path <exam folder> --split test
            python3 test_Matthew.py --path <exam folder> --split validate

        <exam folder>/excel/*.xlsx   excel file with id, target, split
        <exam folder>/Data/          the images
        model_Matthew1.keras, model_Matthew2.keras, ...
                                     ENSEMBLE (exam section 4.8): every numbered model file in the local directory
                                     (folder of this file / folder it is run from) is loaded, and the prediction is
                                     the argmax of the MEAN of their log-probabilities (geometric mean of the softmax
                                     outputs). The class bias tuned in train_Matthew.py is already inside the output
                                     layer of each model. If there is no numbered file, the single model_Matthew.keras
                                     (older submissions) is used.
        summary_Matthew.txt          model.summary() of every model of the ensemble, one after the other
        results_Matthew.xlsx         output: the rows of the chosen split + column "results"
                                     (predicted class index, 0 = class1 ... 5 = class6), same format as train_example.py

        FEATURE models (name like MLP_Matthew_feat_r8_g-hs-v-mom-exr-lbp-lbp2-lbp4-grad-ctr-lab-tile_v2_t5,
        train_Matthew.py of 10/27/2026 pm and later): every image is described by colour and texture histograms
        computed with image_features() (the FEATURE EXTRACTION block above, identical to train_Matthew.py). The decode
        ratio (r8), the feature groups (g-...), the feature version (v2) and the number of test-time views (t5) are
        read from the model name; a feature version not in FEATURE_VERSIONS_OK is refused. The z-score of the
        features is the first layer INSIDE the model. The features are computed in parallel on all CPU cores.
        Test-time augmentation of feature models: the model predicts the whole image and the crops of TTA_CROPS
        (t5 = whole image + 4 corner crops of 80%), and its log-probabilities are averaged over the views. Names
        without _t (older models) use the whole image only.
        The models of an ensemble can use different feature groups and decode ratios: for every ratio the features of
        all the groups its models need are computed ONCE per view, and every model takes its own columns.

        PIXEL models (older, name like MLP_Matthew_flip_hv_prep_r8_aa):
        Image preparation is identical to train_Matthew.py, and is read from each model, so it can never mismatch:
            - IMAGE_SIZE from the model input size (IMAGE_SIZE*IMAGE_SIZE*3)
            - JPEG decode ratio and antialias from the model name (e.g. MLP_Matthew_prep_r8_aa);
              models without that tag (older ones) use full decode + resize without antialias, as they were trained
            decode -> resize -> /255 -> flatten
        The input normalization (minmax / zscore, with the training pmin/pmax/pmean/pstd) is the first layer INSIDE
        the model, so it is applied automatically by final_model.predict().

        Test-time augmentation (TTA): the prediction of every image is the average of the softmax outputs for the
        original image and its flipped versions. Only the flips the model saw in training are used, read from the
        model name (e.g. MLP_Matthew_flip_hv_prep_r8_aa -> original, left-right, up-down, both = 4 views).
        Models without the _flip_ tag (older ones) are predicted on the original image only.
    '''
    #--------------------------------------------------------------------------------------------------------------
    # makes a --user install of openpyxl (os.system above) importable in this same run
    try:
        site.addsitedir(site.getusersitepackages())
    except Exception:
        pass
    importlib.invalidate_caches()

    #------------------------------------------------------------------------------------------------------------------
    # FEATURE EXTRACTION - same code as in train_Matthew.py (nested in test(): exam rule, section 5)
    #------------------------------------------------------------------------------------------------------------------
    ## Every group is a histogram or a summary of the WHOLE image (or of its centre), so a lesion counts the same wherever
    ## it is on the leaf. Groups are always computed and concatenated in the order of FEATURE_ORDER.
    ## Histograms are square-rooted (Hellinger / power normalization: rare lesion colours and textures count more).
    ## LBP codes are grouped into the 51 classes that stay the same when the leaf is flipped or turned by 90 degrees.
    ## Version 2 (10/30/2026): new groups 'lab' and 'tile', test-time crops TTA_CROPS.
    ## Version 3 (10/30/2026 pm): new group 'spread'.
    ## Version 4 (10/30/2026 evening): new groups 'blob' (spot detector) and 'meta' (image file size / shape, read from the
    ## JPEG header, the same for every view of an image: see file_meta). The groups of older versions are computed exactly
    ## as before, so models of every version in FEATURE_VERSIONS_OK are accepted.
    ## Change FEATURE_VERSION whenever this block changes: test_Matthew.py refuses a model made with another version.
    FEATURE_VERSION = 4
    FEATURE_VERSIONS_OK = (1, 2, 3, 4)
    FEATURE_ORDER = ('hs', 'v', 'mom', 'exr', 'lbp', 'lbp2', 'lbp4', 'grad', 'ctr', 'thumb', 'lab', 'tile', 'spread',
                     'blob', 'meta')
    FEATURE_SIZE = {'hs': 288, 'v': 16, 'mom': 12, 'exr': 64, 'lbp': 51, 'lbp2': 51, 'lbp4': 51, 'grad': 16,
                    'ctr': 355, 'thumb': 768, 'lab': 272, 'tile': 403, 'spread': 98, 'blob': 90, 'meta': 5}
    SQRT_GROUPS = ('hs', 'v', 'exr', 'lbp', 'lbp2', 'lbp4', 'grad', 'ctr', 'lab', 'tile')
    TILE_GRID = 4                # 'tile': the image is cut into TILE_GRID x TILE_GRID tiles
    SPREAD_GRID = 6              # 'spread': 6 x 6 tiles
    ## 'blob': spot sizes (blur factors) and, per Lab channel, the 3 contrast thresholds (L* units / a*, b* units)
    BLOB_SCALES = (2, 4, 8)
    BLOB_THRESHOLDS = {'L': (4.0, 8.0, 16.0), 'a': (3.0, 6.0, 12.0), 'b': (3.0, 6.0, 12.0)}
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


    def _lab_words(img):
        '''
            48 coarse colour words per pixel: L* (dark < 40 < mid < 65 < light) x a* (< -20 < -10 < 0 <) x
            b* (< 15 < 30 < 45 <). E.g. deep green leaf, yellowing, brown / orange lesion (a* > 0), dark spot, white powder
        '''
        lab_l, lab_a, lab_b = _rgb_to_lab(img)
        lc = tf.add_n([tf.cast(lab_l > e, tf.int32) for e in (40.0, 65.0)])
        ac = tf.add_n([tf.cast(lab_a > e, tf.int32) for e in (-20.0, -10.0, 0.0)])
        bc = tf.add_n([tf.cast(lab_b > e, tf.int32) for e in (15.0, 30.0, 45.0)])
        return lc * 16 + ac * 4 + bc


    def _blur_pyramid(x, factors):
        '''
            x [H, W] smoothed at every scale f of factors: average over f x f blocks, then bilinear back to [H, W]
            (H, W are first cut to a multiple of the largest factor so every scale has the same size)
        '''
        m = max(factors)
        h, w = (tf.shape(x)[0] // m) * m, (tf.shape(x)[1] // m) * m
        x = x[:h, :w][None, ..., None]
        return x[0, ..., 0], {f: tf.image.resize(tf.nn.avg_pool2d(x, f, f, 'VALID'), [h, w], method='bilinear')[0, ..., 0]
                              for f in factors}


    def _blob_features(img):
        '''
            SPOT DETECTOR (difference of Gaussians, Lindeberg 1998) on the Lab channels L* (light / dark), a* (red-brown /
            green) and b* (yellow-orange / blue): response = blur(scale s) - blur(scale 2s) for s in BLOB_SCALES, i.e. a
            spot about s pixels wide that is lighter (+) or darker (-) / redder / yellower than its surroundings.
            For each channel x scale x sign (18 maps): the fraction of pixels above the 3 thresholds of the channel and
            the fraction of (2s x 2s) blocks holding a response above the 2 largest thresholds (spot density) -> 90
            A leaf with frog-eye spots + scab blotches gets non-zero values in several channels / scales at once.
        '''
        lab_l, lab_a, lab_b = _rgb_to_lab(img)
        out = []
        for name, ch in (('L', lab_l), ('a', lab_a), ('b', lab_b)):
            t1, t2, t3 = BLOB_THRESHOLDS[name]
            ch, blur = _blur_pyramid(ch, sorted(set(BLOB_SCALES) | {2 * s for s in BLOB_SCALES}))
            for s in BLOB_SCALES:
                dog = blur[s] - blur[2 * s]
                for r in (dog, -dog):
                    out += [tf.reduce_mean(tf.cast(r > t, tf.float32)) for t in (t1, t2, t3)]
                    # spot density: fraction of the (2s x 2s) blocks that contain a response above the threshold
                    hb, wb = tf.shape(r)[0] // (2 * s), tf.shape(r)[1] // (2 * s)
                    blocks = tf.reshape(r, [hb, 2 * s, wb, 2 * s])
                    peak = tf.reduce_max(blocks, axis=[1, 3])                  # strongest response of every block
                    out += [tf.reduce_mean(tf.cast(peak > t, tf.float32)) for t in (t2, t3)]
        return tf.sqrt(tf.stack(out))


    def file_meta(file_path):
        '''
            'meta' (5): long side / 4000, short side / 4000, aspect ratio, JPEG bytes per pixel, 1 if 4000 x 2672.
            Read from the file header without decoding (the photos come from several cameras, and the camera is not
            independent of the class). Identical for every view (crop) of an image.
        '''
        raw = tf.io.read_file(file_path)
        shape = tf.cast(tf.io.extract_jpeg_shape(raw), tf.float32)
        big, small = tf.maximum(shape[0], shape[1]), tf.minimum(shape[0], shape[1])
        n_bytes = tf.cast(tf.strings.length(raw), tf.float32)
        usual = tf.cast(tf.logical_and(tf.equal(big, 4000.0), tf.equal(small, 2672.0)), tf.float32)
        return tf.stack([big / 4000.0, small / 4000.0, big / small, n_bytes / (big * small), usual])


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
            groups : names from FEATURE_ORDER, already sorted with sort_feature_groups ('meta' is skipped: file_meta)
            returns a 1-D float32 vector (length = sum of FEATURE_SIZE of the groups except 'meta')
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

        if 'spread' in need:
            # HOW WIDELY each colour is spread and HOW MANY different colours there are (several lesion types on one
            # leaf, e.g. a "mixed" class). For each of the 48 colour words: the fraction of the 6 x 6 tiles where it
            # covers at least 1% of the tile (48) and the square root of its fraction of the image (48); then the
            # entropy of the word histogram and the fraction of words covering at least 0.5% of the image (2)   -> 98
            words = _lab_words(img)
            hh, ww = tf.shape(words)[0], tf.shape(words)[1]
            k = SPREAD_GRID * SPREAD_GRID
            tile = (tf.range(hh) * SPREAD_GRID // hh)[:, None] * SPREAD_GRID + (tf.range(ww) * SPREAD_GRID // ww)[None, :]
            cnt = tf.math.bincount(tf.reshape(tile * 48 + words, [-1]), minlength=k * 48, maxlength=k * 48,
                                   dtype=tf.float32)
            cnt = tf.reshape(cnt, [k, 48])
            cnt = cnt / tf.maximum(tf.reduce_sum(cnt, axis=1, keepdims=True), 1.0)
            present = tf.reduce_mean(tf.cast(cnt >= 0.01, tf.float32), axis=0)
            p = _counts(words, 48)
            entropy = -tf.reduce_sum(p * tf.math.log(tf.maximum(p, 1e-12))) / np.log(48.0)
            n_words = tf.reduce_mean(tf.cast(p >= 0.005, tf.float32))
            out['spread'] = tf.concat([present, tf.sqrt(p), tf.stack([entropy, n_words])], 0)

        if 'blob' in need:
            out['blob'] = _blob_features(img)

        groups = [k for k in groups if k != 'meta']          # 'meta' comes from the file (file_meta), not the image
        return tf.concat([tf.sqrt(out[k]) if k in SQRT_GROUPS else out[k] for k in groups], axis=0)
    #------------------------------------------------------------------------------------------------------------------


    #--------------------------------------------------------------------------------------------------------------
    # Settings
    CHANNELS = 3
    NICKNAME = 'Matthew'
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.AUTOTUNE
    TTA = True    # False = original image only

    #--------------------------------------------------------------------------------------------------------------
    # Arguments  --path  --split
    # (the PDF prints "–path" / "-–path" with long dashes; they are accepted too, so a copy-pasted command does not fail)
    argv = [re.sub(r'^[-–—]+(?=(path|split)\b)', '--', a) for a in sys.argv[1:]]

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, required=True)   # path of the exam folder
    parser.add_argument("--split", default=None, type=str, required=True)  # test, validate
    args = parser.parse_args(argv)

    PATH = os.path.abspath(args.path)
    SPLIT = args.split.strip()
    DATA_DIR = PATH + os.path.sep + 'Data' + os.path.sep

    # "local directory": folder of this file and folder the script was started from
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    local_dirs = [script_dir] if script_dir == os.getcwd() else [script_dir, os.getcwd()]

    #--------------------------------------------------------------------------------------------------------------
    # Reading the excel file from  path + os.path.sep + "excel"
    excel_dir = PATH + os.path.sep + "excel"
    if not os.path.isdir(excel_dir):
        raise FileNotFoundError('Folder not found: {}'.format(excel_dir))

    excel_files = sorted(f for f in os.listdir(excel_dir) if f.endswith('.xlsx') and not f.startswith('~$'))
    if len(excel_files) == 0:
        raise FileNotFoundError('No .xlsx file found in {}'.format(excel_dir))
    FILE_NAME = excel_dir + os.path.sep + excel_files[0]

    xdf_data = pd.read_excel(FILE_NAME)

    for col in ('id', 'split'):
        if col not in xdf_data.columns:
            raise ValueError('The excel file has no "{}" column, columns found: {}'.format(col, list(xdf_data.columns)))

    #--------------------------------------------------------------------------------------------------------------
    # Multiclass target, same as process_target(1) in train_example.py:
    # classes sorted alphabetically -> index 0..n-1, one-hot written as text "0,0,1,0,0,0" in target_class
    has_target = 'target' in xdf_data.columns and xdf_data['target'].notna().any()
    class_names = np.sort(xdf_data['target'].dropna().unique()) if has_target else np.array([])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    if has_target:
        idx = xdf_data['target'].map(class_to_idx)
        xdf_data['target_class'] = [
            ",".join('1' if j == int(i) else '0' for j in range(len(class_names))) if pd.notna(i) else ''
            for i in idx]

    # Filtering the rows of the requested split (extra spaces / upper case ignored)
    mask = xdf_data['split'].astype(str).str.strip().str.lower() == SPLIT.lower()
    xdf_dset = xdf_data[mask].copy()
    if len(xdf_dset) == 0:
        raise ValueError('No rows with split == "{}". Splits found: {}'.format(
            SPLIT, sorted(xdf_data['split'].astype(str).unique())))

    ds_inputs = (DATA_DIR + xdf_dset['id'].astype(str).str.strip()).tolist()
    missing = [p for p in ds_inputs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('{} images not found, for example {} . Folder used: {}'.format(
            len(missing), missing[:3], DATA_DIR))

    #--------------------------------------------------------------------------------------------------------------
    # The models: model_Matthew1.keras, model_Matthew2.keras, ... in the local directory (first folder where they are
    # found), in number order. Without numbered files: the single model_Matthew.keras (older submissions).
    def models_in(folder):
        pattern = re.compile(r'^model_{}(\d+)\.keras$'.format(re.escape(NICKNAME)))
        numbered = sorted((int(m.group(1)), f) for f in os.listdir(folder) for m in [pattern.match(f)] if m)
        if numbered:
            return [os.path.join(folder, f) for _, f in numbered]
        single = os.path.join(folder, 'model_{}.keras'.format(NICKNAME))
        return [single] if os.path.exists(single) else []

    model_files = []
    for d in local_dirs:
        model_files = models_in(d)
        if model_files:
            break
    if not model_files:
        raise FileNotFoundError('No model_{0}1.keras, model_{0}2.keras, ... or model_{0}.keras found in {1}'.format(
            NICKNAME, local_dirs))
    print('Models of the ensemble: {}'.format(', '.join(os.path.basename(f) for f in model_files)))

    #--------------------------------------------------------------------------------------------------------------
    def features_of(model):
        '''
            feature models: (decode ratio, sorted feature groups, number of TTA views) read from the name,
            None for pixel models
        '''
        m = re.search(r'_feat_r(\d+)_g-([a-z0-9-]+)_v(\d+)(?:_t(\d+))?$', model.name)
        if m is None:
            return None
        if int(m.group(3)) not in FEATURE_VERSIONS_OK:
            raise ValueError('{} was trained with feature version {}, this script computes versions {}'.format(
                model.name, m.group(3), FEATURE_VERSIONS_OK))
        groups = m.group(2).split('-')
        if sort_feature_groups(groups) != groups:
            raise ValueError('Feature groups of {} are not in the expected order'.format(model.name))
        views = int(m.group(4)) if (m.group(4) is not None and TTA) else 1
        if not 1 <= views <= len(TTA_CROPS):
            raise ValueError('{} asks for {} test-time views, this script has {}'.format(model.name, views,
                                                                                     len(TTA_CROPS)))
        return int(m.group(1)), groups, views

    feature_cache = {}

    def feature_matrix(ratio, groups, view=0):
        '''
            features of every image of the split (in the order of the rows) in test-time view `view` of TTA_CROPS
            (0 = whole image), computed once per (ratio, groups, view) and shared by the models.
            tf.data runs image_features on all CPU cores at the same time; deterministic=True keeps the row order.
        '''
        key = (ratio, tuple(groups), view)
        if key not in feature_cache:
            ## 'meta' (last group of FEATURE_ORDER) comes from the file header, the other groups from the image
            image_groups = [g for g in groups if g != 'meta']

            def one(f):
                parts = [image_features(tta_crop(decode_image(f, ratio), view), image_groups)] if image_groups else []
                if 'meta' in groups:
                    parts.append(file_meta(f))
                return tf.concat(parts, axis=0)

            ds = tf.data.Dataset.from_tensor_slices(ds_inputs)
            ds = ds.map(one, num_parallel_calls=AUTOTUNE, deterministic=True)
            ds = ds.batch(64).prefetch(AUTOTUNE)
            feature_cache[key] = np.concatenate([xb.numpy() for xb in ds], axis=0)
            print('Features computed: {} images x {} values (ratio {}, groups {}, view {})'.format(
                feature_cache[key].shape[0], feature_cache[key].shape[1], ratio, '-'.join(groups), view))
        return feature_cache[key]

    def preparation_of(model):
        '''
            reads from the model name how its training images were prepared: (jpeg decode ratio, antialias)
        '''
        m = re.search(r'_prep_r(\d+)(_aa)?$', model.name)
        if m is None:
            return 1, False          # older models: full decode, resize without antialias
        return int(m.group(1)), m.group(2) is not None

    def tta_views_of(model):
        '''
            views for test-time augmentation: list of (flip left-right, flip up-down), the original image first.
            Only the flips the model was trained with (from the model name, e.g. ..._flip_hv_prep_r8_aa)
        '''
        m = re.search(r'_flip_([hv]+)_prep_r\d+', model.name)
        flips = m.group(1) if (m is not None and TTA) else ''
        views = [(False, False)]
        if 'h' in flips:
            views.append((True, False))
        if 'v' in flips:
            views.append((False, True))
        if 'h' in flips and 'v' in flips:
            views.append((True, True))
        return views

    def predict_proba(model, ds, image_size, views):
        '''
            softmax output of every image (in the order of the rows), averaged over the TTA views
            the flips are exact pixel mirrors of the prepared image (same code as train_Matthew.py)
        '''
        if len(views) == 1:
            return model.predict(ds, verbose=0)

        n_inputs = image_size * image_size * CHANNELS
        out = []
        for xb in ds:
            img = tf.reshape(xb, [-1, image_size, image_size, CHANNELS])
            p = 0.0
            for flip_lr, flip_ud in views:
                axes = ([2] if flip_lr else []) + ([1] if flip_ud else [])
                view = tf.reverse(img, axis=axes) if axes else img
                p = p + np.asarray(model.predict_on_batch(tf.reshape(view, [-1, n_inputs])))
            out.append(p / len(views))
        return np.concatenate(out, axis=0)

    def process_path(file_path, image_size, ratio, antialias):
        '''
            reads one image from disk and prepares it exactly as in training:
            decode -> resize -> pixels 0-1 -> flat vector of image_size*image_size*CHANNELS values
            (the TTA flips are applied later, in predict_proba)
        '''
        img = tf.io.read_file(file_path)
        if ratio > 1:
            img = tf.io.decode_jpeg(img, channels=CHANNELS, ratio=ratio)
        else:
            img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)
        img = tf.image.resize(img, [image_size, image_size], antialias=antialias)
        img = img / 255.0
        img = tf.reshape(img, [image_size * image_size * CHANNELS])
        return img

    def read_data(image_size, ratio, antialias):
        '''
            tf.data pipeline over the image paths. NOT shuffled, so predictions stay aligned with the excel rows
        '''
        list_ds = tf.data.Dataset.from_tensor_slices(ds_inputs)
        final_ds = list_ds.map(lambda f: process_path(f, image_size, ratio, antialias), num_parallel_calls=AUTOTUNE)
        return final_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    #--------------------------------------------------------------------------------------------------------------
    feature_union = {}     # decode ratio -> sorted feature groups needed by the models of that ratio

    def group_columns(groups, union):
        '''
            positions of the columns of `groups` inside a feature vector of the groups `union` (both sorted)
        '''
        start, pos = 0, {}
        for g in union:
            pos[g] = np.arange(start, start + FEATURE_SIZE[g])
            start += FEATURE_SIZE[g]
        return np.concatenate([pos[g] for g in groups])

    def model_proba(final_model, mf):
        '''
            softmax output of one model for every image of the split (in the order of the rows)
        '''
        n_inputs = int(final_model.input_shape[-1])
        feat = features_of(final_model)

        if feat is not None:
            # feature model: colour + texture histograms -> model (features computed once, shared by the models)
            # with test-time augmentation: mean of the log-probabilities over the views, normalized again
            ratio, groups, views = feat
            union = feature_union[ratio]
            cols = group_columns(groups, union)
            logp = 0.0
            for v in range(views):
                X = feature_matrix(ratio, union, v)[:, cols]
                if X.shape[1] != n_inputs:
                    raise ValueError('{} expects {} inputs, the features have {}'.format(mf, n_inputs, X.shape[1]))
                p = np.asarray(final_model.predict(X, batch_size=1024, verbose=0))
                logp = logp + np.log(np.clip(p, 1e-12, 1.0)).astype(np.float64)
            logp = logp / views
            prob = np.exp(logp - logp.max(axis=1, keepdims=True))
            prob = prob / prob.sum(axis=1, keepdims=True)
            print('Model used: {} ({} features, jpeg ratio {}, groups {}, {} test-time view(s))'.format(
                os.path.basename(mf), n_inputs, ratio, '-'.join(groups), views))
        else:
            # pixel model (older versions of train_Matthew.py)
            image_size = int(round((n_inputs / CHANNELS) ** 0.5))
            if image_size * image_size * CHANNELS != n_inputs:
                raise ValueError('{} expects {} inputs, which is not IMAGE_SIZE*IMAGE_SIZE*{}'.format(
                    mf, n_inputs, CHANNELS))
            ratio, antialias = preparation_of(final_model)
            views = tta_views_of(final_model)
            prob = predict_proba(final_model, read_data(image_size, ratio, antialias), image_size, views)
            print('Model used: {} (image size {}, jpeg ratio {}, antialias {}, TTA views {}, first layer: {})'.format(
                os.path.basename(mf), image_size, ratio, antialias, len(views), final_model.layers[0].name))

        if len(prob) != len(xdf_dset):
            raise ValueError('{}: {} predictions for {} rows'.format(mf, len(prob), len(xdf_dset)))
        return prob

    #--------------------------------------------------------------------------------------------------------------
    # Predict: every model is loaded exactly as the grader loads it (default compile=True), so the summaries below
    # match the submitted models. Ensemble = mean of the log-probabilities (geometric mean of the softmax outputs).
    os.environ['COLUMNS'] = '80'   # same summary width as an automated run (no terminal)
    summary_lines = ['Ensemble of {} model(s), prediction = geometric mean of their softmax outputs'.format(
        len(model_files)), '']
    loaded = [(mf, tf.keras.models.load_model(mf)) for mf in model_files]
    for mf, final_model in loaded:
        feat = features_of(final_model)
        if feat is not None:
            feature_union[feat[0]] = sort_feature_groups(sorted(set(feature_union.get(feat[0], [])) | set(feat[1])))

    sum_logp = None
    for mf, final_model in loaded:
        summary_lines.append('==== {} ===='.format(os.path.basename(mf)))
        final_model.summary(print_fn=lambda x, **kwargs: summary_lines.append(x))
        summary_lines.append('')

        logp = np.log(np.clip(model_proba(final_model, mf), 1e-12, 1.0)).astype(np.float64)
        if sum_logp is not None and logp.shape != sum_logp.shape:
            raise ValueError('{} has {} outputs, the other models {}'.format(mf, logp.shape[1], sum_logp.shape[1]))
        sum_logp = logp if sum_logp is None else sum_logp + logp
    del loaded

    # summary_<nickname>.txt written from the loaded models, as save_model() in test_to_do.py
    for d in local_dirs:
        with open(os.path.join(d, 'summary_{}.txt'.format(NICKNAME)), 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(summary_lines) + '\n')

    mean_logp = sum_logp / len(model_files)
    prob = np.exp(mean_logp - mean_logp.max(axis=1, keepdims=True))
    prob = prob / prob.sum(axis=1, keepdims=True)     # geometric mean, normalized (argmax = argmax of mean_logp)

    # the answer is the label (index of the most probable class), not the probability
    xdf_dset['results'] = np.argmax(prob, axis=1)

    #--------------------------------------------------------------------------------------------------------------
    # write the results to results_<nickname>.xlsx in the local directory
    for d in local_dirs:
        xdf_dset.to_excel(os.path.join(d, 'results_{}.xlsx'.format(NICKNAME)), index=False)
    print('Predicted {} images of split "{}". Results saved to {}'.format(
        len(xdf_dset), SPLIT, os.path.join(local_dirs[0], 'results_{}.xlsx'.format(NICKNAME))))

    #--------------------------------------------------------------------------------------------------------------
    def metrics_func(metrics, aggregates=None):
        '''
        multiple functiosn of metrics to call each function
        f1, cohen, accuracy, mattews correlation
        list of metrics: f1_micro, f1_macro, f1_weighted, coh, acc, mat
        list of aggregates : avg, sum
        '''
        if aggregates is None:
            aggregates = []

        y_true = xdf_dset['target'].map(class_to_idx).to_numpy()
        y_pred = xdf_dset['results'].to_numpy()

        table = {
            'f1_micro': ('f1_score micro', lambda t, p: f1_score(t, p, average='micro')),
            'f1_macro': ('f1_score macro', lambda t, p: f1_score(t, p, average='macro')),
            'f1_weighted': ('f1_score weighted', lambda t, p: f1_score(t, p, average='weighted')),
            'coh': ('cohen_kappa_score', cohen_kappa_score),
            'acc': ('accuracy_score', accuracy_score),
            'mat': ('mattews_coef', matthews_corrcoef),
        }

        xcont = 0   # number of metrics actually computed
        xsum = 0
        for xm in metrics:
            if xm not in table:
                print('Metric does not exist:', xm)
                continue
            name, fn = table[xm]
            xmet = fn(y_true, y_pred)
            print(name, xmet)
            xsum = xsum + xmet
            xcont = xcont + 1

        if 'sum' in aggregates:
            print('Sum of Metrics : ', xsum)
        if 'avg' in aggregates and xcont > 0:
            print('Average of Metrics : ', xsum / xcont)

    # Metrics only when the split has labels; this part can never stop the script (results are already saved)
    try:
        if not has_target or xdf_dset['target'].isna().any():
            print('Metrics skipped: this split has no labels')
        elif len(class_names) != prob.shape[1]:
            print('Metrics skipped: the excel file has {} classes, the model {}'.format(len(class_names), prob.shape[1]))
        else:
            metrics_func(['f1_macro', 'coh', 'acc'], ['avg'])
    except Exception as e:
        print('Metrics skipped:', e)
#------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    test()
#------------------------------------------------------------------------------------------------------------------
