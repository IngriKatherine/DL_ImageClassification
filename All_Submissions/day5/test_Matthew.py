#------------------------------------------------------------------------------------------------------------------
import os
import sys
import argparse
import importlib
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
try:
    import site
    site.addsitedir(site.getusersitepackages())   # makes a --user install importable in this same run
except Exception:
    pass
importlib.invalidate_caches()
#------------------------------------------------------------------------------------------------------------------

def test():
    '''
        Usage (Exam1 section 5):
            python3 test_Matthew.py --path <exam folder> --split test
            python3 test_Matthew.py --path <exam folder> --split validate

        <exam folder>/excel/*.xlsx   excel file with id, target, split
        <exam folder>/Data/          the images
        model_Matthew.keras          model in the local directory (folder of this file / folder it is run from)
                                     ensemble (section 4.8): model_Matthew1.keras, model_Matthew2.keras, ... are
                                     also loaded if present, and their probabilities are averaged
        results_Matthew.xlsx         output: the rows of the chosen split + column "results"
                                     (predicted class index, 0 = class1 ... 5 = class6), same format as train_example.py

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
    # Settings
    CHANNELS = 3
    NICKNAME = 'Matthew'
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.AUTOTUNE
    TTA = True    # False = original image only

    #--------------------------------------------------------------------------------------------------------------
    # Arguments  --path  --split
    # (the PDF prints "–path" with a long dash; it is accepted too, so a copy-pasted command does not fail)
    argv = [('--' + a.lstrip('–—-')) if a.startswith(('–', '—')) else a for a in sys.argv[1:]]

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
    # Models in the local directory: model_Matthew.keras and/or model_Matthew1.keras, model_Matthew2.keras, ...
    model_files = []
    for d in local_dirs:
        names = sorted(os.listdir(d))
        single = 'model_{}.keras'.format(NICKNAME)
        numbered = sorted((n for n in names if n.startswith('model_{}'.format(NICKNAME)) and n.endswith('.keras')
                           and n[len('model_{}'.format(NICKNAME)):-len('.keras')].isdigit()),
                          key=lambda n: int(n[len('model_{}'.format(NICKNAME)):-len('.keras')]))
        found = ([single] if single in names else []) + numbered
        if found:
            model_files = [os.path.join(d, n) for n in found]
            break
    if not model_files:
        raise FileNotFoundError('model_{}.keras not found in {}'.format(NICKNAME, local_dirs))

    #--------------------------------------------------------------------------------------------------------------
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
    # Predict: softmax probabilities of every model, averaged (one model = no averaging)
    prob_sum = None
    for k, mf in enumerate(model_files):
        # loaded exactly as the grader loads it (default compile=True, optimizer state included),
        # so the summary below is an exact match of the summary generated from the submitted model
        final_model = tf.keras.models.load_model(mf)

        if k == 0:
            os.environ['COLUMNS'] = '80'   # same summary width as an automated run (no terminal)
            # summary_<nickname>.txt written from the loaded model, as save_model() in test_to_do.py
            for d in local_dirs:
                with open(os.path.join(d, 'summary_{}.txt'.format(NICKNAME)), 'w', encoding='utf-8', newline='\n') as fh:
                    final_model.summary(print_fn=lambda x, **kwargs: fh.write(x + '\n'))

        n_inputs = int(final_model.input_shape[-1])
        image_size = int(round((n_inputs / CHANNELS) ** 0.5))
        if image_size * image_size * CHANNELS != n_inputs:
            raise ValueError('{} expects {} inputs, which is not IMAGE_SIZE*IMAGE_SIZE*{}'.format(mf, n_inputs, CHANNELS))

        ratio, antialias = preparation_of(final_model)
        views = tta_views_of(final_model)
        res = predict_proba(final_model, read_data(image_size, ratio, antialias), image_size, views)
        if len(res) != len(xdf_dset):
            raise ValueError('{} predictions for {} rows'.format(len(res), len(xdf_dset)))

        prob_sum = res if prob_sum is None else prob_sum + res
        print('Model used: {} (image size {}, jpeg ratio {}, antialias {}, TTA views {}, first layer: {})'.format(
            mf, image_size, ratio, antialias, len(views), final_model.layers[0].name))

    # the answer is the label (index of the most probable class), not the probability
    xdf_dset['results'] = np.argmax(prob_sum, axis=1)

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
        elif len(class_names) != prob_sum.shape[1]:
            print('Metrics skipped: the excel file has {} classes, the model {}'.format(len(class_names), prob_sum.shape[1]))
        else:
            metrics_func(['f1_macro', 'coh', 'acc'], ['avg'])
    except Exception as e:
        print('Metrics skipped:', e)
#------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    test()
#------------------------------------------------------------------------------------------------------------------
