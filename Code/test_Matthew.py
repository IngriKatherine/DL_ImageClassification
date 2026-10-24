import os
import site
import importlib

# Each line below does nothing when the libraries are already installed. Otherwise it installs them:
# 1) installs pip itself if the machine has none (a bare Ubuntu image often does not)
os.system("python3 -c \"import importlib.util as u,sys; sys.exit(any(u.find_spec(m) is None for m in ['tensorflow','pandas','numpy','openpyxl','sklearn']))\" 2>/dev/null || which pip > /dev/null 2>&1 || (sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip)")
# 2) the normal installation
os.system("python3 -c \"import importlib.util as u,sys; sys.exit(any(u.find_spec(m) is None for m in ['tensorflow','pandas','numpy','openpyxl','sklearn']))\" 2>/dev/null || sudo pip install -q tensorflow pandas numpy openpyxl scikit-learn")
# 3) Ubuntu 23.04 and later refuse step 2 ("externally-managed-environment"), this flag allows it
os.system("python3 -c \"import importlib.util as u,sys; sys.exit(any(u.find_spec(m) is None for m in ['tensorflow','pandas','numpy','openpyxl','sklearn']))\" 2>/dev/null || sudo pip install -q --break-system-packages tensorflow pandas numpy openpyxl scikit-learn")
# 4) machines without sudo
os.system("python3 -c \"import importlib.util as u,sys; sys.exit(any(u.find_spec(m) is None for m in ['tensorflow','pandas','numpy','openpyxl','sklearn']))\" 2>/dev/null || pip install -q --user --break-system-packages tensorflow pandas numpy openpyxl scikit-learn || pip install -q --user tensorflow pandas numpy openpyxl scikit-learn")


def test():
    '''
        python3 test_Matthew.py --path <exam folder> --split test
        python3 test_Matthew.py --path <exam folder> --split validate

        <exam folder>/excel/*.xlsx   excel file with id, target and split
        <exam folder>/Data/          the images
        model_Matthew.keras          the model, in the same folder as this file
        results_Matthew.xlsx         the output: the rows of the chosen split with a new column "results"
                                     (index of the predicted class, 0 = first class in alphabetical order)
    '''

    import argparse

    # make sure libraries that were installed a moment ago by the lines above can be imported
    importlib.invalidate_caches()
    try:
        site.addsitedir(site.getusersitepackages())
    except Exception:
        pass

    import numpy as np
    import pandas as pd
    import tensorflow as tf

    #--------------------------------------------------------------------------------------------------------------
    # Settings (the model decides the real image size, see below)
    CHANNELS = 3
    NICKNAME = 'Matthew'
    BATCH_SIZE = 32          # images are read from disk batch by batch, never all at once
    AUTOTUNE = tf.data.AUTOTUNE

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    # "local directory": the folder of this file and the folder the script was started from
    local_dirs = []
    for d in (script_dir, os.getcwd()):
        if d not in local_dirs:
            local_dirs.append(d)

    #--------------------------------------------------------------------------------------------------------------
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, required=True)   # exam folder
    parser.add_argument("--split", default=None, type=str, required=True)  # test, validate
    args = parser.parse_args()

    path = os.path.abspath(args.path)
    split = args.split.strip()

    #--------------------------------------------------------------------------------------------------------------
    # Excel file: path + os.path.sep + "excel"
    excel_dir = path + os.path.sep + "excel"
    if not os.path.isdir(excel_dir):
        raise FileNotFoundError('Folder not found: {}'.format(excel_dir))

    excel_files = sorted(f for f in os.listdir(excel_dir) if f.endswith('.xlsx') and not f.startswith('~$'))
    if len(excel_files) == 0:
        raise FileNotFoundError('No .xlsx file found in {}'.format(excel_dir))

    xdf_data = pd.read_excel(excel_dir + os.path.sep + excel_files[0])

    for col in ('id', 'split'):
        if col not in xdf_data.columns:
            raise ValueError('The excel file has no "{}" column, columns found: {}'.format(col, list(xdf_data.columns)))

    # rows of the requested split (extra spaces and upper case letters are ignored)
    mask = xdf_data['split'].astype(str).str.strip().str.lower() == split.lower()
    xdf_dset = xdf_data[mask].copy()
    if len(xdf_dset) == 0:
        raise ValueError('No rows with split == "{}". Splits found: {}'.format(split, sorted(xdf_data['split'].astype(str).unique())))

    #--------------------------------------------------------------------------------------------------------------
    # Images: <path>/Data/ (if that folder does not exist, the images are looked for directly in <path>)
    data_dir = path + os.path.sep + 'Data' + os.path.sep
    if not os.path.isdir(data_dir):
        data_dir = path + os.path.sep

    ds_inputs = (data_dir + xdf_dset['id'].astype(str).str.strip()).tolist()

    missing = [p for p in ds_inputs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('{} images not found, for example {} . Folder used: {}'.format(len(missing), missing[:3], data_dir))

    #--------------------------------------------------------------------------------------------------------------
    # Model: model_<nickname>.keras in the local directory
    model_file = None
    for d in local_dirs:
        candidate = os.path.join(d, 'model_{}.keras'.format(NICKNAME))
        if os.path.exists(candidate):
            model_file = candidate
            break
    if model_file is None:
        raise FileNotFoundError('model_{}.keras not found in {}'.format(NICKNAME, local_dirs))

    final_model = tf.keras.models.load_model(model_file, compile=False)

    # the model is a stack of Dense layers fed with the flattened image: IMAGE_SIZE * IMAGE_SIZE * CHANNELS values
    n_inputs = int(final_model.input_shape[-1])
    image_size = int(round((n_inputs / CHANNELS) ** 0.5))
    if image_size * image_size * CHANNELS != n_inputs:
        raise ValueError('The model expects {} inputs, which is not IMAGE_SIZE * IMAGE_SIZE * {}'.format(n_inputs, CHANNELS))

    #--------------------------------------------------------------------------------------------------------------
    def process_path(file_path):
        '''
            reads one image and prepares it exactly as it was prepared in training
        '''
        img = tf.io.read_file(file_path)
        img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)
        img = tf.image.resize(img, [image_size, image_size])
        img = img / 255.0                                   # pixels from 0-255 to 0-1
        img = tf.reshape(img, [image_size * image_size * CHANNELS])
        return img

    # Only the file names are kept in memory. NOT shuffled: predictions line up with the rows of the excel file
    test_ds = tf.data.Dataset.from_tensor_slices(ds_inputs)
    test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    #--------------------------------------------------------------------------------------------------------------
    # Predict. The model returns one probability per class (softmax), the answer is the index of the most probable one
    res = final_model.predict(test_ds, verbose=0)

    if len(res) != len(xdf_dset):
        raise ValueError('{} predictions for {} rows'.format(len(res), len(xdf_dset)))

    xdf_dset['results'] = np.argmax(res, axis=1)

    # Results file: results_<nickname>.xlsx
    for d in local_dirs:
        xdf_dset.to_excel(os.path.join(d, 'results_{}.xlsx'.format(NICKNAME)), index=False)
    print('Predicted {} images of split "{}". Results saved to {}'.format(len(xdf_dset), split, os.path.join(local_dirs[0], 'results_{}.xlsx'.format(NICKNAME))))

    #--------------------------------------------------------------------------------------------------------------
    # Metrics, only when the split has labels. This part can never stop the script, the results file is already saved
    def metrics_func(metrics, aggregates=None):
        '''
        list of metrics: f1_micro, f1_macro, f1_weighted, coh, acc, mat
        list of aggregates : avg, sum
        '''
        from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef

        if aggregates is None:
            aggregates = []

        class_names = np.sort(xdf_data['target'].dropna().unique())
        class_to_idx = {c: i for i, c in enumerate(class_names)}
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

        xcont = 0   # number of metrics that were actually computed
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

    try:
        if 'target' in xdf_dset.columns and xdf_dset['target'].notna().all():
            if len(np.unique(xdf_data['target'].dropna())) == res.shape[1]:
                metrics_func(['f1_macro', 'acc', 'coh', 'mat'], ['avg', 'sum'])
            else:
                print('Metrics skipped: the excel file does not contain the {} classes of the model'.format(res.shape[1]))
        else:
            print('Metrics skipped: this split has no labels')
    except Exception as e:
        print('Metrics skipped:', e)


if __name__ == "__main__":
    test()