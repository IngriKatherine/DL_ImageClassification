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
        model_Matthew.keras          THE model (one file only) in the local directory (folder of this file / folder
                                     it is run from). No other model file is ever loaded.
        results_Matthew.xlsx         output: the rows of the chosen split + column "results"
                                     (predicted class index, 0 = class1 ... 5 = class6), same format as train_example.py

        FEATURE models (name like MLP_Matthew_feat_r8_g-hs-v-mom-exr-lbp-lbp2-lbp4-grad-ctr_v1, train_Matthew.py of
        10/27/2026 pm): every image is described by colour and texture histograms computed with image_features()
        (the FEATURE EXTRACTION block above, identical to train_Matthew.py). The decode ratio (r8), the feature
        groups (g-...) and the feature version (v1) are read from the model name; a model made with another
        FEATURE_VERSION is refused. The z-score of the features is the first layer INSIDE the model.
        The features are computed in parallel on all CPU cores (tf.data).

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
    # The model: model_Matthew.keras in the local directory (first folder where it is found). One file only.
    candidates = [os.path.join(d, 'model_{}.keras'.format(NICKNAME)) for d in local_dirs]
    found = [f for f in candidates if os.path.exists(f)]
    if not found:
        raise FileNotFoundError('model_{}.keras not found in {}'.format(NICKNAME, local_dirs))
    mf = found[0]

    #--------------------------------------------------------------------------------------------------------------
    def features_of(model):
        '''
            feature models: (decode ratio, sorted feature groups) read from the name, None for pixel models
        '''
        m = re.search(r'_feat_r(\d+)_g-([a-z0-9-]+)_v(\d+)$', model.name)
        if m is None:
            return None
        if int(m.group(3)) != FEATURE_VERSION:
            raise ValueError('{} was trained with feature version {}, this script computes version {}'.format(
                model.name, m.group(3), FEATURE_VERSION))
        groups = m.group(2).split('-')
        if sort_feature_groups(groups) != groups:
            raise ValueError('Feature groups of {} are not in the expected order'.format(model.name))
        return int(m.group(1)), groups

    feature_cache = {}

    def feature_matrix(ratio, groups):
        '''
            features of every image of the split (in the order of the rows), computed once per (ratio, groups).
            tf.data runs image_features on all CPU cores at the same time; deterministic=True keeps the row order.
        '''
        key = (ratio, tuple(groups))
        if key not in feature_cache:
            ds = tf.data.Dataset.from_tensor_slices(ds_inputs)
            ds = ds.map(lambda f: image_features(decode_image(f, ratio), groups), num_parallel_calls=AUTOTUNE,
                        deterministic=True)
            ds = ds.batch(64).prefetch(AUTOTUNE)
            feature_cache[key] = np.concatenate([xb.numpy() for xb in ds], axis=0)
            print('Features computed: {} images x {} values (ratio {}, groups {})'.format(
                feature_cache[key].shape[0], feature_cache[key].shape[1], ratio, '-'.join(groups)))
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
    # Predict: softmax probabilities of the model
    # loaded exactly as the grader loads it (default compile=True, optimizer state included),
    # so the summary below is an exact match of the summary generated from the submitted model
    final_model = tf.keras.models.load_model(mf)

    os.environ['COLUMNS'] = '80'   # same summary width as an automated run (no terminal)
    # summary_<nickname>.txt written from the loaded model, as save_model() in test_to_do.py
    for d in local_dirs:
        with open(os.path.join(d, 'summary_{}.txt'.format(NICKNAME)), 'w', encoding='utf-8', newline='\n') as fh:
            final_model.summary(print_fn=lambda x, **kwargs: fh.write(x + '\n'))

    n_inputs = int(final_model.input_shape[-1])
    feat = features_of(final_model)

    if feat is not None:
        # feature model: colour + texture histograms -> model
        ratio, groups = feat
        X = feature_matrix(ratio, groups)
        if X.shape[1] != n_inputs:
            raise ValueError('{} expects {} inputs, the features have {}'.format(mf, n_inputs, X.shape[1]))
        prob = np.asarray(final_model.predict(X, batch_size=1024, verbose=0))
        print('Model used: {} ({} features, jpeg ratio {}, groups {})'.format(mf, n_inputs, ratio, '-'.join(groups)))
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
            mf, image_size, ratio, antialias, len(views), final_model.layers[0].name))

    if len(prob) != len(xdf_dset):
        raise ValueError('{} predictions for {} rows'.format(len(prob), len(xdf_dset)))

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
