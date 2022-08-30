import os
import sys
if os.path.exists('/work/braindecode'):
    sys.path.insert(0, '/work/braindecode')
    sys.path.insert(0, '/work/mne-python')
    print('adding local code resources')
import math
import json
import glob
import pickle
import logging
import warnings
import argparse
from datetime import datetime
from functools import partial
from collections import OrderedDict
from io import StringIO as StringBuffer

import mne
mne.set_log_level('ERROR')
#mne.set_config("MNE_LOGGING_LEVEL", "ERROR")
import torch
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_color_codes('deep')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.set_loglevel('ERROR')
from sklearn.metrics import mean_absolute_error, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, Checkpoint, TrainEndCheckpoint, ProgressBar
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from braindecode.datasets.tuh import TUHAbnormal
from braindecode.preprocessing import Preprocessor, preprocess
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, to_dense_prediction_model, Deep4Net, TCN
from braindecode.models.modules import Expression
from braindecode.regressor import EEGRegressor
from braindecode.classifier import EEGClassifier
from braindecode.training import CroppedLoss, CroppedTrialEpochScoring
from braindecode.augmentation import *


formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
screen_handler = logging.StreamHandler(stream=sys.stdout)
screen_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(screen_handler)


# TODO: use base64 encoded json as input?
def _decode_tueg(params):
    config = json.loads(params)
    decode_tueg(**params, config=config)
    
# TODO: replace color codes in the output log txt file?
# TODO: physician reports are missing in TUH_PRE. add back
# TODO: look at the (negative) outliers reports. why are they outliers?
# TODO: try under-/oversampling to cope with the target distribution?
def decode_tueg(
    batch_size,
    config,
    data_path,
    debug,
    final_eval,
    intuitive_training_scores,
    model_name,
    n_epochs,
    n_jobs,
    n_restarts,
    n_train_recordings,
    out_dir,
    preload,
    seed,
    shuffle_data_before_split,
    split_kind,
    squash_outs,
    standardize_data,
    standardize_targets,
    subset,
    target_name,
    tmax,
    tmin,
    valid_set_i,
    window_size_samples,
    augment,
    fast_mode,
    loss,
):
    """
    TODO: add docstring
    """
    now = datetime.now()
    now = now.strftime("%y%m%d%H%M%S%f")
    out_dir = os.path.join(out_dir, now)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        raise RuntimeError(f'Directory already exists {out_dir}')
    add_file_logger(
        logger=logger,
        out_dir=out_dir,
    )

    warnings.filterwarnings("ignore", message="'pathological' not in description.")
    warnings.filterwarnings("ignore", message="torch.backends.cudnn.benchmark was set to True which")
    warnings.filterwarnings("ignore", message="You are using an callback that overrides on_batch_begin or on_batc")
    warnings.filterwarnings("ignore", message="This function was designed to predict trials from cropped datasets")
    #warnings.filterwarnings("ignore", message="UserWarning: y_pred contains classes not in y_true")
    
    check_input_args(
        batch_size, config, data_path, debug, final_eval, intuitive_training_scores,
        model_name, n_epochs, n_jobs, n_restarts, n_train_recordings, out_dir,
        preload, seed, shuffle_data_before_split, split_kind, squash_outs, standardize_data,
        standardize_targets, subset, target_name, tmax, tmin, valid_set_i,
        window_size_samples, augment, loss, logger,
    )

    #log_capture_string = get_log_capturer(logger, debug)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    logger.info(f'\n{config.sort_index()}')

    # check if GPU is available, if True chooses to use it
    cuda = torch.cuda.is_available()
    if not cuda:
        raise RuntimeError('no gpu found')
    torch.backends.cudnn.benchmark = True
    logger.debug(f"cuda: {cuda}")
    cropped = True
    logger.debug(f"cropped: {cropped}")

    tuabn_train, tuabn_valid, mapping  = get_datasets(
        data_path, 
        target_name,
        subset,
        n_train_recordings,
        tmin,
        tmax,
        n_jobs, 
        final_eval,
        valid_set_i,
        seed,
    )
    title = create_title(
        final_eval,
        len(tuabn_train.datasets),
        len(tuabn_valid.datasets),
        subset,
        tmin,
        tmax,
        target_name,
        model_name,
        n_epochs, 
        n_restarts,
        out_dir,
        augment,
        loss,
    )
    save_input(
        config,
        out_dir,
        tuabn_train.description,
        tuabn_valid.description,
        test_name(final_eval),
    )
    ch_names = tuabn_train.datasets[0].raw.ch_names
    sfreq = tuabn_train.datasets[0].raw.info['sfreq']
    n_channels = len(ch_names)
    model, lr, weight_decay = get_model(
        n_channels,
        seed,
        cuda,
        target_name,
        model_name,
        cropped,
        window_size_samples,
        squash_outs,
    )
    n_preds_per_input = model(torch.ones(1, n_channels, window_size_samples, 1)).size()[2]
    logger.debug(f"model {model_name} produces {n_preds_per_input} preds for every input of size {window_size_samples}")
    tuabn_train, tuabn_valid = create_windows(
        mapping, 
        tuabn_train,
        tuabn_valid,
        window_size_samples,
        n_channels,
        n_jobs,
        preload,
        n_preds_per_input,
        test_name(final_eval),
    )
    tuabn_train, tuabn_valid, target_transform, data_transform = standardize(
        standardize_data, 
        standardize_targets,
        tuabn_train,
        tuabn_valid,
        target_name,
    )
    callbacks = get_callbacks(
        n_epochs, 
        n_restarts,
        target_name,
        target_transform,
        intuitive_training_scores,
        fast_mode,
        test_name(final_eval),
        out_dir,
    )
    estimator = get_estimator(
        cuda, 
        model,
        target_name,
        cropped,
        lr,
        weight_decay,
        n_jobs, 
        n_epochs,
        tuabn_valid,
        batch_size,
        callbacks,
        loss,
    )
    estimator, augmenter = set_augmentation(
        augment,
        ch_names,
        seed,
        estimator,
        sfreq,
        len(tuabn_train),
        batch_size,
    )
    logger.info(title)
    logger.info(f'starting training')
    estimator.fit(tuabn_train, y=None)
    logger.info(f'finished training')
    # generate simple output
    df = pd.DataFrame(estimator.history)
    df.to_csv(os.path.join(out_dir, 'history.csv'))
    with open(os.path.join(out_dir, 'data_scaler.pkl'), 'wb') as f:
        pickle.dump(data_transform, f)
    with open(os.path.join(out_dir, 'target_scaler.pkl'), 'wb') as f:
        pickle.dump(target_transform, f)
    scores = create_final_scores(
        estimator,
        tuabn_train,
        tuabn_valid,
        test_name(final_eval),
        target_name,
        target_transform,
    )
    pd.DataFrame(scores).to_csv(os.path.join(out_dir, 'train_end_scores.csv'))
    logger.info(f'made final predictions')
    fig, ax = plt.subplots(1, 1, figsize=(15,3))
    ax = plot_learning(
        df=df,
        loss_key='loss',
        loss_name=f'{loss} loss',
        test_name=test_name(final_eval),
        ax=ax,
    )
    ax.set_title(title)
    save_fig(fig, out_dir, 'curves')
    # post-hoc analysis
    # TODO: plot learning
#     generate_outputs(
#         target_name,
#         intuitive_training_scores,
#         tuabn_valid, 
#         data_transform,
#         target_transform,
#         tuabn_train,
#         fast_mode,
#         title,
#         out_dir,
#         estimator,
#         test_name(final_eval),
#         standardize_targets,
#         augmenter,
#     )
#     logger.info(f'generated outputs')
#     predict_longitudinal_datasets(
#         data_transform,
#         target_transform,
#         out_dir, 
#         tmin,
#         tmax,
#         n_jobs,
#         window_size_samples,
#         n_channels,
#         n_preds_per_input,
#         preload,
#     )
    logger.info('done.')


def add_file_logger(
    logger, 
    out_dir,
):
    handler = logging.FileHandler(os.path.join(out_dir, 'log.txt'), mode='w')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def check_input_args(
    batch_size, 
    config,
    data_path,
    debug,
    final_eval,
    intuitive_training_scores,
    model_name, 
    n_epochs,
    n_jobs,
    n_restarts,
    n_train_recordings,
    out_dir,
    preload,
    seed,
    shuffle_data_before_split,
    split_kind,
    squash_outs,
    standardize_data,
    standardize_targets, 
    subset,
    target_name,
    tmax,
    tmin,
    valid_set_i,
    window_size_samples,
    augment,
    loss,
    logger,
):
    # warn about impossible choices
    if intuitive_training_scores and target_name != 'age':
        logger.warning(f"'intuitive_training_scores' without effect with this target ({target_name})")
    if subset != 'mixed' and target_name == 'pathological':
        raise ValueError(f"cannot decode '{target_name}' with just one class ({subset})")
    if final_eval == 1:
        logger.warning(f"'valid_set_i' without effect when final_eval is True")
        logger.warning(f"'split_kind' without effect when final_eval is True")
    if target_name != 'age' and standardize_targets:
        logger.warning(f"'standardize_targets' without effect with this target ({target_name})")
    if shuffle_data_before_split and split_kind in ['age_distribution', 'chronological']:
        logger.warning(f"'shuffle_data_before_split' without effect with this split_kind ({split_kind})")
    if target_name not in ['age', 'gender', 'pathological', 'age_clf']:
        raise ValueError(f"unknown target name {target_name}")
    if subset not in ['normal', 'abnormal', 'mixed']:
        raise ValueError(f'unkown subset ({subset})')
    if model_name not in ['shallow', 'deep', 'tcn']:
        raise ValueError(f'Model {model_name} unknown.')
    if n_restarts > 0:
        assert n_epochs % (n_restarts+1) == 0, f'{n_restarts} does not split {n_epochs} into {n_restarts+1} equal parts'
    splits = data_path.split(os.sep)
    if 'train' in splits or 'eval' in splits:
        raise ValueError(f"Please set 'data_path' to parent of train and eval.")
    if final_eval == 1 and n_train_recordings != -1:
        logger.warning(f"'n_train_recordings' without effect if final eval")
    if final_eval not in [0, 1]:
        raise ValueError
    if preload not in [0, 1]:
        raise ValueError
    if split_kind not in ['chronological', 'random', 'age_distribution']:
        raise ValueError(f"unkown split type {split_kind}")
    if augment not in [
        'dropout', 'shuffle', 'fliplr', 'random', 
        'reverse', 'sign', 'noise', 'mask', 'flipfb',
        'identity',
    ]:
        raise ValueError(f"Unkown augmentation {augment}.")
    if loss not in ['mse', 'mae', 'log_cosh', 'huber']:
        raise ValueError(f'Unkown loss {loss}')
    if target_name in ['pathological', 'gender', 'age_clf']:
        logger.warning(f"'loss' without effect with this target ({target_name})")
    if squash_outs not in [0, 1]:
        raise ValueError
    if squash_outs == 1 and target_name != 'age':
        logger.warning(f"'squash_outs' without effect with this target ({target_name})")
    if final_eval == 0:
        assert valid_set_i in [0, 1, 2, 3, 4]


def test_name(final_eval):
    return 'valid' if final_eval == 0 else 'eval'
        

def get_preprocessors(
    tmin,
    tmax,
    ch_names=None,
    lfreq=None,
    hfreq=None,
):
    tmin = None if tmin == -1 else tmin
    tmax = None if tmax == -1 else tmax
    ch_names = None if ch_names == -1 else ch_names
    lfreq = None if lfreq == -1 else lfreq
    hfreq = None if hfreq == -1 else hfreq
    preprocessors = []
    if tmin is not None or tmax is not None:
        logger.debug(f"adding cropper {tmin} – {tmax}")
        preprocessors.append(Preprocessor('crop', tmin=tmin, tmax=tmax, include_tmax=False))
    if ch_names is not None:
        logger.debug(f"adding channel picker {ch_names}")
        preprocessors.append(
            Preprocessor('pick_channels', ch_names=['EEG ' + ch for ch in ch_names], ordered=True))
    if lfreq is not None or hfreq is not None:
        logger.debug(f"adding filter {lfreq} – {hfreq}")
        preprocessors.append(Preprocessor('filter', l_freq=lfreq, h_freq=hfreq))
    return preprocessors


def _preprocess(tuabn_train, tmin, tmax, n_jobs):
    preprocessors = get_preprocessors(tmin, tmax)
    return preprocess(tuabn_train, preprocessors=preprocessors, n_jobs=n_jobs) 


def get_train_eval_datasets(tuabn_train, target_name, seed):
    if target_name in ['age', 'age_clf']:
        logger.debug(f"into train (0.9) and eval (0.1).")
        train_ids, eval_ids = train_eval_split(tuabn_train.description, seed)
        intersection = np.intersect1d(train_ids, eval_ids)
        if intersection.any():
            raise RuntimeError(f"leakage train / eval detected {intersection}")
        tuabn_eval = tuabn_train.split(eval_ids)['0']
        tuabn_train = tuabn_train.split(train_ids)['0']
    else:
        logger.debug(f"using predefined train and eval set as provided by TUH")
        tuabn_eval = tuabn_train.split('train')['False']
        tuabn_train = tuabn_train.split('train')['True']
    return tuabn_train, tuabn_eval


def get_train_valid_datasets(tuabn_train, target_name, valid_set_i, seed):
    logger.info(f"validation run, removing eval from dataset with {len(tuabn_train.description)} recordings")
    tuabn_train, _ = get_train_eval_datasets(tuabn_train, target_name, seed)
    logger.debug(f"splitting dataset with {len(tuabn_train.description)} recordings")
    logger.debug(f"into train (.8) and valid (.2).")
    # for pathology decoding, turn off shuffling and make time series split chronologically
    shuffle = True if target_name in ['age', 'age_clf'] else False
    train_ids, valid_ids = train_valid_split(tuabn_train.description, valid_set_i, seed, shuffle)
    intersection = np.intersect1d(train_ids, valid_ids)
    if intersection.any():
        raise RuntimeError(f"leakage train / valid detected {intersection}")
    tuabn_valid = tuabn_train.split(valid_ids)['0']
    tuabn_train = tuabn_train.split(train_ids)['0']
    return tuabn_train, tuabn_valid


def get_datasets(
    data_path,
    target_name,
    subset,
    n_train_recordings,
    tmin,
    tmax,
    n_jobs,
    final_eval,
    valid_set_i,
    seed,
):
    logger.debug("indexing files")
    tuabn_train = TUHAbnormal(
        path=data_path,
        preload=False,
        add_physician_reports=True,
        n_jobs=n_jobs,
        target_name = 'age' if target_name in ['age', 'age_clf'] else target_name,
    )
    if final_eval == 1:
        logger.debug(f"splitting dataset with {len(tuabn_train.description)} recordings")
        tuabn_train, tuabn_valid = get_train_eval_datasets(tuabn_train, target_name)
    else:
        tuabn_train, tuabn_valid = get_train_valid_datasets(tuabn_train, target_name, valid_set_i, seed)

    # select normal/abnormal only
    logger.debug(f"from train ({len(tuabn_train.datasets)}) and {test_name(final_eval)}"
                 f" ({len(tuabn_valid.datasets)}) selecting {subset}")
    tuabn_train = subselect(tuabn_train, subset)
    tuabn_valid = subselect(tuabn_valid, subset)
    logger.debug(f"selected train ({len(tuabn_train.datasets)}) and {test_name(final_eval)}"
                 f" ({len(tuabn_valid.datasets)})")
    
    # reduce number of train recordings
    if n_train_recordings != -1:
        tuabn_train = tuabn_train.split([list(range(n_train_recordings))])['0']
        logger.debug(f"selected {n_train_recordings} train recordings")

    some_durations = [ds.raw.n_times/ds.raw.info['sfreq'] for ds in tuabn_train.datasets][:3]
    logger.debug(f'some raw durations {some_durations}')
    logger.debug("preprocessing")
    [tuabn_train, tuabn_valid] = [_preprocess(ds, tmin=tmin, tmax=tmax, n_jobs=n_jobs) for ds in [tuabn_train, tuabn_valid]]
    some_durations = [ds.raw.n_times/ds.raw.info['sfreq'] for ds in tuabn_train.datasets][:3]
    logger.debug(f"some preprocessed durations {some_durations}")
    logger.debug(f'train datasets {len(tuabn_train.datasets)}')
    logger.debug(f'{test_name(final_eval)} datasets {len(tuabn_valid.datasets)}')
    
    # map potentially incompatible targets to appropiate types
    if target_name == 'pathological':
        mapping = {True: 1, False: 0}
    elif target_name == 'gender':
        mapping = {'M': 0, 'F': 1}
    else:
        mapping = None
    return tuabn_train, tuabn_valid, mapping


def subselect(
    dataset,
    subset,
):
    # select normal / abnormal only
    if subset != 'mixed':
        k = 'pathological'
        v = 'False' if subset == 'normal' else 'True'
        not_v = 'True' if subset == 'normal' else 'False'
        splits = dataset.split(k)
        dataset = splits[v]
        rest_dataset = splits[not_v]
        #return dataset, rest_dataset
    return dataset


def get_model(
    n_channels,
    seed,
    cuda,
    target_name,
    model_name,
    cropped,
    window_size_samples,
    squash_outs,
):
    logger.debug("creating model")
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    # TODO: add hyperparams
    if model_name == 'shallow':
        lr = 0.0625 * 0.01
        weight_decay = 0
        n_start_channels = 40  # default
        dropout = .5
        if target_name == 'age':
            n_classes = 1
            final_conv_length = 35
        elif target_name == 'gender':
            n_classes = 2
            final_conv_length = 25
        elif target_name == 'pathological':
            n_classes = 2
            final_conv_length = 25
        elif target_name == 'age_clf':
            raise NotImplementedError
        model = ShallowFBCSPNet(
            in_chans=n_channels,
            n_classes=n_classes,
            n_filters_time=n_start_channels,
            n_filters_spat=n_start_channels,
            input_window_samples=window_size_samples,
            final_conv_length=final_conv_length,
            drop_prob=dropout,
        )
    elif model_name == 'deep':
        lr = 0.01
        weight_decay = 0.5 * 0.001
        n_start_channels = 25  # default
        final_conv_length = 1
        dropout = .5
        if target_name == 'age':
            n_classes = 1
        elif target_name == 'gender':
            n_classes = 2
        elif target_name == 'pathological':
            n_classes = 2
        elif target_name == 'age_clf':
            raise NotImplementedError
        model = Deep4Net(
            in_chans=n_channels,
            n_classes=n_classes,
            input_window_samples=window_size_samples,
            final_conv_length=final_conv_length,
            n_filters_time=n_start_channels,
            n_filters_spat=n_start_channels,
            drop_prob=dropout,
            stride_before_pool=True,
        )
    elif model_name == 'tcn':
        # values taken from patryk
        if target_name == 'age':
            n_outputs = 1
            add_log_softmax = False
            lr = 0.0004732953501425473
            weight_decay = 1.0025958447703478e-07
            model = TCN(
                n_in_chans=n_channels,
                n_outputs=n_outputs,
                n_filters=53,
                n_blocks=5,
                kernel_size=9,
                drop_prob=0.0195875974361336,
                add_log_softmax=add_log_softmax,
            )
        elif target_name == 'gender':
            raise NotImplementedError
        elif target_name == 'pathological':
            n_outputs = 2
            lr = 0.0011261049710243193
            weight_decay = 5.83730537673086e-07
            l2_decay = 1.7491630095065614e-08
            dropout = 0.05270154233150525 
            raise NotImplementedError
        elif target_name == 'age_clf':
            # TODO: how many classes for age classification?
            n_outputs = 100
            add_log_softmax = True
            lr = 0.0004732953501425473
            weight_decay = 1.0025958447703478e-07
            model = TCN(
                n_in_chans=n_channels,
                n_outputs=n_outputs,
                n_filters=53,
                n_blocks=5,
                kernel_size=9,
                drop_prob=0.0195875974361336,
                add_log_softmax=add_log_softmax,
            )

    # make deep and shallow dense convert to a regression model with 1 output class
    # we remove the softmax from tcn in constructor, it also does not have to be made dense
    if model_name in ['shallow', 'deep']:
        if cropped:
            to_dense_prediction_model(model)

        if target_name == 'age':
            # remove the softmax layer from models
            new_model = torch.nn.Sequential()
            for name, module_ in model.named_children():
                if "softmax" in name:
                    continue
                new_model.add_module(name, module_)
            model = new_model
    # add a sigmoid to the end of model
    if target_name == 'age' and squash_outs:
        new_model = torch.nn.Sequential()
        new_model.add_module(model_name, model)
        new_model.add_module('sigmoid', torch.nn.Sigmoid())
        model = new_model
    logger.info(model)
    return model, lr, weight_decay


def _create_windows(
    mapping,
    tuabn,
    window_size_samples,
    n_channels,
    n_jobs, 
    preload,
    n_preds_per_input,
):
    return create_fixed_length_windows(
        tuabn,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples-n_preds_per_input,
        drop_last_window=False,
        n_jobs=min(n_jobs, 4),
        preload=bool(preload),
        mapping=mapping,
        drop_bad_windows=True,
        reject=None,
        flat=None,
    )


def create_windows(
    mapping,
    tuabn_train,
    tuabn_valid,
    window_size_samples,
    n_channels,
    n_jobs, 
    preload,
    n_preds_per_input,
    test_name,
):
    logger.debug("windowing")
    [tuabn_train, tuabn_valid] = [
        _create_windows(
            mapping=mapping,
            tuabn=ds,
            window_size_samples=window_size_samples,
            n_channels=n_channels,
            n_jobs=n_jobs,
            preload=preload,
            n_preds_per_input=n_preds_per_input,
        )  for ds in [tuabn_train, tuabn_valid]
    ]
    logger.debug(f'train windows {len(tuabn_train)}')
    logger.debug(f'{test_name} windows {len(tuabn_valid)}')
    return tuabn_train, tuabn_valid


def train_eval_split(df, seed):
    train, eval_ = train_test_split(df, test_size=1/10, random_state=seed)
    return sorted(train.index.to_list()), sorted(eval_.index.to_list())


def train_valid_split(df, valid_set_i, seed, shuffle):
    train, valid1 = train_test_split(df, test_size=1/5, random_state=seed, shuffle=shuffle)
    train, valid2 = train_test_split(train, test_size=1/4, random_state=seed, shuffle=shuffle)
    train, valid3 = train_test_split(train, test_size=1/3, random_state=seed, shuffle=shuffle)
    valid4, valid5 = train_test_split(train, test_size=1/2, random_state=seed, shuffle=shuffle)
    valid_sets = {
        0: valid1.index.to_list(),
        1: valid2.index.to_list(),
        2: valid3.index.to_list(),
        3: valid4.index.to_list(),
        4: valid5.index.to_list(),
    }
    valid_is = sorted(valid_sets.pop(valid_set_i))
    train_is = sorted([l for k, v in valid_sets.items() for l in v])
    return train_is, valid_is


def get_avg_ch_mean_n_std(
    tuabn,
    ch_names,
):
    # read mean and std of all the recordings
    # used to scale the data to roughly zero mean and unit variance
    # note: mean and std are in microvolts already
    mean_df = pd.concat(
        objs=[
            pd.read_csv(f.replace('.edf', '_stats.csv'), index_col=0)['mean'] 
            for f in tuabn.description['path']
        ], 
        axis=1,
    )
    std_df = pd.concat(
        objs=[
            pd.read_csv(f.replace('.edf', '_stats.csv'), index_col=0)['std'] 
            for f in tuabn.description['path']
        ], 
        axis=1,
    )
    avg_ch_mean = mean_df.mean(axis=1)
    avg_ch_std = std_df.mean(axis=1)
    if ch_names is not None:
        avg_ch_mean = avg_ch_mean[ch_names]
        avg_ch_std = avg_ch_std[ch_names]
    avg_ch_mean = avg_ch_mean.to_numpy().reshape(-1, 1)
    avg_ch_std = avg_ch_std.to_numpy().reshape(-1, 1)
    return avg_ch_mean, avg_ch_std


class DataScaler():
    def __init__(self, avg_ch_mean=0, avg_ch_std=1):
        self.factor = 1e6  # volts to microvolts
        self.avg_ch_mean = avg_ch_mean
        self.avg_ch_std = avg_ch_std

    def __call__(self, x):
        x = ((x * self.factor - self.avg_ch_mean) / self.avg_ch_std)
        return x.astype('float32')

    def invert(self, x):
        return (x * self.avg_ch_std) + self.avg_ch_mean


def sine_scale(y, miny, maxy):
    y = y - miny
    y = y / (maxy - miny)
    y = y * np.pi
    y = y + np.pi
    y = np.cos(y)
    y = y + 1 
    y = y / 2
    return y


def inv_sine_scale(y, miny, maxy):
    y = y * 2
    y = y - 1
    # y could be outside of [-1,1]
    # in those cases, arccos fails
    # how to handle?
    y = np.arccos(y)
    y = y - np.pi
    y = y / np.pi
    y = y * (maxy + miny)
    y = y + miny
    return -y

    
class TargetScaler():
    def __init__(self, add_element=0, mult_element=1, kind='standard'):
        self.add_element = add_element
        self.mult_element = mult_element
        self.kind = kind

    def __call__(self, y):
        if self.kind == 'standard':
            return (y - self.add_element) / self.mult_element
            raise NotImplementedError
        elif self.kind == 'exponential':
            return np.power(((y - self.add_element) / self.mult_element), self.exp)
            raise NotImplementedError
        elif self.kind == 'sine':
            return sine_scale(y, self.add_element, self.mult_element)
        elif self.kind == 'percentage':
            raise NotImplementedError
        elif self.kind == 'sigmoid':
            torch.special.expit((y - self.add_element) / self.mult_element)
            raise NotImplementedError
        if self.kind == 'minmax':
            scale = (self.add_element / ((self.mult_element - self.add_element)))
            return (y / (self.mult_element - self.add_element)) - scale

    def invert(self, y):
        if self.kind == 'standard':
            return y * self.mult_element + self.add_element
        elif self.kind == 'exponential':
            raise NotImplementedError
            return (np.power(y, 1/self.exp) * self.mult_element) + self.add_element
        elif self.kind == 'sine':
            raise NotImplementedError
            return inv_sine_scale(y, self.add_element, self.mult_element) 
        elif self.kind == 'percentage':
            raise NotImplementedError
        elif self.kind == 'sigmoid':
            torch.special.logit(y * self.mult_element + self.add_element)
            raise NotImplementedError
        if self.kind == 'minmax':
            scale = (self.add_element / ((self.mult_element - self.add_element)))
            return (y + scale) * (self.mult_element - self.add_element)
        # TODO: check inversion of minmax scale! seemed ok


def standardize(
    standardize_data,
    standardize_targets,
    tuabn_train,
    tuabn_valid,
    target_name,
):
    data_transform = DataScaler()
    if standardize_data:
        # get avg ch mean and std of train data
        avg_ch_mean, avg_ch_std = get_avg_ch_mean_n_std(tuabn_train, ch_names=None)
        data_transform.avg_ch_mean = avg_ch_mean
        data_transform.avg_ch_std = avg_ch_std

    # add a data transform to train and valid that scales the data
    # to microvolts and zero mean unit variance accoring to the train data
    logger.debug(f'prior to data scaling {tuabn_train[0][0][0][0]}')
    tuabn_train.transform = data_transform
    tuabn_valid.transform = data_transform
    logger.debug(f'post data scaling {tuabn_train[0][0][0][0]}')

    # TODO: manually set min / max age?
    # min_ = 0
    # max_ = 150
    kind = 'minmax'
    target_transform = TargetScaler(kind=kind)
    if standardize_targets and target_name == 'age':
        train_targets = tuabn_train.get_metadata()['target']
        add_element = train_targets.min() if kind in ['sine', 'minmax'] else train_targets.mean()
        mult_element = train_targets.max() if kind in ['sine', 'minmax'] else train_targets.std()
        target_transform = TargetScaler(
            add_element=add_element,
            mult_element=mult_element,
            kind=kind,
        )
        logger.debug(f'mean/min train age: {target_transform.add_element:.2f}')
        logger.debug(f'std/max train age: {target_transform.mult_element:.2f}')

    logger.debug(f'prior to {kind} target scaling {tuabn_train[0][1]}')
    tuabn_train.target_transform = target_transform
    tuabn_valid.target_transform = target_transform
    logger.debug(f'post {kind} target scaling {tuabn_train[0][1]}')
    return tuabn_train, tuabn_valid, target_transform, data_transform


def create_title(
    final_eval,
    n_train,
    n_valid,
    subset,
    tmin,
    tmax,
    target_name,
    model_name,
    n_epochs,
    n_restarts,
    out_dir,
    augment,
    loss,
):
    # create an output subdir
    cv = 'valid' if not final_eval else 'eval'
    title = f'TUAB, {n_train}–{n_valid} {subset}, {tmin}s–{tmax}s, {target_name}, {model_name}, {n_epochs}–{n_restarts}, {loss}, {augment}, {cv}'
    return title


def save_input(
    config,
    out_dir,
    train_description,
    valid_description,
    test_name,
):
    config.to_csv(os.path.join(out_dir,'config.csv'))
    train_description.to_csv(os.path.join(out_dir, 'train_description.csv'))
    valid_description.to_csv(os.path.join(out_dir, f'{test_name}_description.csv'))


def trial_age_mae(
    model,
    X,
    y,
    target_scaler,
    return_y_yhat,
):
    return age_mae(
        model=model,
        X=X,
        y=y,
        target_scaler=target_scaler,
        trialwise=True,
        return_y_yhat=return_y_yhat,
    )


def window_age_mae(
    model,
    X,
    y,
    target_scaler,
    return_y_yhat,
):
    return age_mae(
        model=model,
        X=X,
        y=y,
        target_scaler=target_scaler,
        trialwise=False,
        return_y_yhat=return_y_yhat,
    )


def age_mae(
    model,
    X,
    y,
    target_scaler,
    trialwise,
    return_y_yhat,
    
):
    """Custom scoring that inverts the target scaling, such that it gives intuitively 
    understandable age mae scores."""
    if trialwise:
        y_pred, y_true = model.predict_trials(X)
        y_pred = np.array([np.mean(y_pred_, axis=1) for y_pred_ in y_pred])
    else:
        y_pred = model.predict(X)
    y_true = target_scaler.invert(y_true)
    y_pred = target_scaler.invert(y_pred)
    mae = float(mean_absolute_error(y_true=y_true, y_pred=y_pred))
    if return_y_yhat:
        return {'score_name': 'mae', 'score': mae, 'y_true': y_true, 'y_pred': y_pred}
    return {'score_name': 'mae', 'score': mae}


def acc(
    y_true,
    y_pred,
    return_y_yhat,
):
    score = 1 - balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    if return_y_yhat:
        return {'score_name': 'acc', 'score': score, 'y_true': y_true, 'y_pred': y_pred}
    return {'score_name': 'acc', 'score': mae}


def window_acc(
    model,
    X,
    y,
    return_y_yhat,
):
    y_pred = model.predict(X)
    return acc(y, y_pred, return_y_yhat)


def trial_acc(
    model,
    X,
    y,
    return_y_yhat,
):
    y_pred, y_true = model.predict_trials(X)
    y_pred = np.array([np.mean(y_pred_, axis=1) for y_pred_ in y_pred])
    y_pred = y_pred.argmax(axis=1)
    return acc(y_true, y_pred, return_y_yhat)


# TODO: implement a binned acc / mae on classifiction?
    
    
def is_before_restart(
    net,
    epoch_i,
):
    # add checkpoints before restart for potential ensemble building
    return net.history[-1, 'epoch'] == epoch_i


def get_callbacks(
    n_epochs,
    n_restarts,
    target_name,
    target_scaler,
    intuitive_training_scores,
    fast_mode,
    test_name,
    out_dir,
):
    # add callbacks for cosine annealing with warm restarts and a checkpointer
    n_epochs_per_restart = int(n_epochs/(n_restarts+1))
    callbacks = [
        ("lr_scheduler", LRScheduler(
            CosineAnnealingWarmRestarts, T_0=n_epochs_per_restart, T_mult=1)),
        # ("progress_bar", ProgressBar()),  # unfortunately not working in jupyter?
    ]
    # compute the mean and std on train ages
    # scale both train and valid set ages accordingly
    # set up scoring functions that invert the scaling for both
    # such that intuitive performance tracking during training can take place
    if target_name == 'age':
        if intuitive_training_scores and fast_mode == 0:
            callbacks.extend([
                (f"MAE_{test_name}", CroppedTrialEpochScoring(
                    partial(trial_age_mae, target_scaler=target_scaler),
                    name=f'{test_name}_age_mae', on_train=False, 
                    lower_is_better=True)),#, avg_axis=2)),
                ("MAE_train", CroppedTrialEpochScoring(
                    partial(trial_age_mae, target_scaler=target_scaler),
                    name='train_age_mae', on_train=True, 
                    lower_is_better=True)),#, avg_axis=2)),
                ])
    elif target_name in ['pathological', 'gender', 'age_clf']:
        if fast_mode == 0:
            callbacks.extend([
                (f"ACC_{test_name}", CroppedTrialEpochScoring(
                    trial_acc, name=f'{test_name}_misclass', on_train=False, 
                    lower_is_better=True)),#, avg_axis=2)),
                ("ACC_train", CroppedTrialEpochScoring(
                    trial_acc, name='train_misclass', on_train=True, 
                    lower_is_better=True)),#, avg_axis=2)),
                ])
    if n_restarts > 0:
        # one checkpoint for every restart? because of fn_prefix
        callbacks.extend([
            (f'checkpoint_{i}', Checkpoint(
                dirname=os.path.join(out_dir, 'checkpoint'), fn_prefix=f'restart_{i}_',
                monitor=partial(is_before_restart, epoch_i=i*n_epochs_per_restart),
                f_pickle='model.pkl'))
            for i in range(1, n_restarts+1)
        ])
    # order of callbacks matters. 'valid_age_mae_best' / 'valid_acc_best' has to be written
    # to history before checkpoint tries to access it
    if intuitive_training_scores and fast_mode == 0:
        monitor = f'{test_name}_age_mae_best' if target_name == 'age' else f'{test_name}_misclass_best'
    else:
        monitor = f'{test_name}_loss_best'
    callbacks.extend([
        (f"best_{test_name}", Checkpoint(
            dirname=os.path.join(out_dir, 'checkpoint'), monitor=monitor, fn_prefix=f'{test_name}_best_',
            f_pickle='model.pkl')),  # load_best=True?
        ("after_train", TrainEndCheckpoint(
            dirname=os.path.join(out_dir, 'checkpoint'), f_pickle='model.pkl')),
    ])
    return callbacks


def mean_percentage_error(input, target):
    # does not seem to work well
    e = target-input
    return (target/e).mean()


def mean_squared_percentage_error(input, target):
    # does not seem to work well
    e = target-input
    return (target/e*target/e).mean()


def log_cosh_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(input - target))


def get_estimator(
    cuda,
    model,
    target_name,
    cropped,
    lr,
    weight_decay,
    n_jobs,
    n_epochs,
    tuabn_valid,
    batch_size,
    callbacks,
    loss,
):
    # Send model to GPU
    if cuda:
        logger.debug('sending to gpu')
        model.cuda()

    # in case of age try l1_loss?
    if target_name != 'age':
        loss_function = torch.nn.functional.nll_loss
    else:
        if loss == 'mse':
            loss_function = torch.nn.functional.mse_loss
        elif loss == 'mae':
            loss_function = torch.nn.functional.l1_loss
        elif loss == 'log_cosh':
            loss_function = log_cosh_loss
        elif loss == 'huber':
            loss_function = torch.nn.functional.huber_loss
    Estimator = EEGRegressor if target_name == 'age' else EEGClassifier
    estimator = Estimator(
        model,
        cropped=cropped,
        criterion=CroppedLoss,
        criterion__loss_function=loss_function,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        iterator_train__num_workers=n_jobs,
        iterator_valid__num_workers=n_jobs,
        max_epochs=n_epochs,
        train_split=predefined_split(tuabn_valid),
        batch_size=batch_size,
        callbacks=callbacks,
        # make the PrintLog callback use the logger defined above
        callbacks__print_log__sink=logger.info,
        device='cuda' if cuda else 'cpu',
    )
    # use TransformedTargetRegressor?
    #estimator = TransformedTargetRegressor(
    #    regressor=estimator,
    #    transformer=,
    #)
    return estimator


class ChannelsSymmetryFB(ChannelsSymmetry):
    def __init__(self, probability, ordered_ch_names, random_state):
        super().__init__(
            probability=probability, 
            ordered_ch_names=ordered_ch_names, 
            random_state=random_state,
        )
        hardcoded_ch_map = {
            'FP1': 'O1',
            'O1': 'FP1',
            'FP2': 'O2',
            'O2': 'FP2',
            'F7': 'T5',
            'T5': 'F7',
            'F3': 'P3',
            'P3': 'F3',
            'FZ': 'PZ',
            'PZ': 'FZ',
            'F4': 'P4',
            'P4': 'F4',
            'F8': 'T6',
            'T6': 'F8',
            # self mapping
            'A1': 'A1',
            'T3': 'T3',
            'C3': 'C3',
            'CZ': 'CZ',
            'C4': 'C4',
            'T4': 'T4',
            'A2': 'A2',
        }
        prefix = 'EEG '
        ordered_ch_names = [ch.strip(prefix) for ch in ordered_ch_names]
        self.permutation = [
            ordered_ch_names.index(hardcoded_ch_map[ch])
            for ch in ordered_ch_names
        ]


class Augmenter(Transform):
    # there will be one augmenter object per n_job and per epoch...
    # so generating a history might be complicated
    def __init__(self, probability, random_state, n_examples, batch_size):
        super().__init__(probability=probability, random_state=random_state)
        self.choices = []
        self.choice_history = []
        self.calls_per_epoch = n_examples // batch_size
        if n_examples % batch_size != 0:
            self.calls_per_epoch += 1
        logger.debug(f"With {n_examples} examples and batch size of {batch_size}")
        logger.debug(f"expecting {self.calls_per_epoch} batches/calls per epoch")
        self.n_times_called = 0

    def forward(self, X, y):
        if self.n_times_called % self.calls_per_epoch == 0:
            self.choice_history.append([])
        self.n_times_called += 1
        # for every batch, select one of the augmentation in choices
        choice = np.random.choice(len(self.choices))
        # keep track of the choices
        self.choice_history[-1].append(choice)
        transform = self.choices[choice]
        return transform.forward(X, y)


def set_augmentation(
    augment,
    ch_names,
    seed,
    estimator,
    sfreq,
    n_examples,
    batch_size,
):
    probability = 1
    augmentations = {
        'dropout': ChannelsDropout(probability=probability, p_drop=.2, random_state=seed),
        'flipfb': ChannelsSymmetryFB(probability=probability, ordered_ch_names=ch_names, random_state=seed),
        'fliplr': ChannelsSymmetry(probability=probability, ordered_ch_names=ch_names, random_state=seed),
        'mask': SmoothTimeMask(probability=probability, mask_len_samples=int(sfreq), random_state=seed),
        'noise': GaussianNoise(probability=probability, std=.1, random_state=seed),
        'reverse': TimeReverse(probability=probability, random_state=seed),
        'shuffle': ChannelsShuffle(probability=probability, p_shuffle=.2, random_state=seed),
        'sign': SignFlip(probability=probability, random_state=seed),
        'identity': IdentityTransform(),
        'identity': IdentityTransform(),
    }
    """More options:
        FTSurrogate,
        BandstopFilter,
        FrequencyShift,
        SensorsRotation,
        SensorsZRotation,
        SensorsYRotation,
        SensorsXRotation,
        Mixup,
    """
    logger.debug(f"Adding augmentation transform '{augment}'")
    # chooses one of its augmentation choices per batch
    augmenter = Augmenter(
        probability=probability,
        random_state=seed,
        n_examples=n_examples,
        batch_size=batch_size,
    )
    if augment == 'random':
        augmenter.choices = [v for k, v in augmentations.items()]
        logger.debug(f"Batch-wise choices are {augmenter.choices}")
    else:
        # limit to only one choice if not random
        augmenter.choices = [augmentations[augment]]
    estimator.set_params(**{
        'iterator_train__transforms': [augmenter],
        'iterator_train': AugmentedDataLoader,
    })
    return estimator, augmenter


def save_log(
    log_capture_string,
    out_dir,
    close,
):
    # get log from buffer and save to file
    log_contents = log_capture_string.getvalue()
    if close:
        log_capture_string.close()
    with open(os.path.join(out_dir, 'log.txt'), 'w') as f:
        f.writelines(log_contents)

        
def create_final_scores(
    estimator,
    tuabn_train,
    tuabn_valid,
    test_name,
    target_name,
    target_scaler,
):
    scores = []
    for ds_name, ds in [('train', tuabn_train), (test_name, tuabn_valid)]:
        score = _create_final_scores(estimator, ds, ds_name, target_name, target_scaler)
        score['split'] = ds_name
        scores.append(score)
        logger.info(f"on {score['ds_name']} reached {score['score']:.2f} {score['score_name']}")
    return scores


def _create_final_scores(
    estimator,
    ds,
    ds_name,
    target_name,
    target_scaler,
    return_y_yhat,
):
    if target_name == 'age':
        score = trial_age_mae(estimator, ds, None, target_scaler, return_y_yhat)
    else:
        score = trial_acc(estimator, ds, None, return_y_yhat)
    return score


def generate_outputs(
    target_name,
    intuitive_training_scores,
    tuabn_valid,
    data_transform,
    target_transform,
    tuabn_train,
    fast_mode,
    title,
    out_dir,
    estimator,
    test_name,
    standardize_targets,
    augmenter,
):
    # TODO: always make interpretable predictions after fast mode training
    # TODO: make predictions when decoding pathology
    logger.debug("generating outputs")
    df = pd.DataFrame(estimator.history)
    df.to_csv(os.path.join(out_dir, 'history.csv'))
    with open(os.path.join(out_dir, 'data_scaler.pkl'), 'wb') as f:
        pickle.dump(data_transform, f)
    with open(os.path.join(out_dir, 'target_scaler.pkl'), 'wb') as f:
        pickle.dump(target_transform, f)
    mean_train_age = tuabn_train.description['age'].mean()
    std_train_age = tuabn_train.description['age'].std()
    name = 'age' if target_name == 'age_clf' else target_name
    # matplotlib creates a massive amount of output at level debug, therefore overwrite it
    if target_name == 'age':
        loss_key = 'loss' if not intuitive_training_scores or fast_mode == 1 else 'age_mae'
        loss_name = 'mse loss' if not intuitive_training_scores or fast_mode == 1 else 'MAE [years]'
        # dummy score is mean train age predicted for all valid trials
        if not intuitive_training_scores or fast_mode == 1:
            # TODO: check
            y_pred_ = mean_train_age if not standardize_targets else 0
            y_pred = len(tuabn_valid.get_metadata())*[y_pred_]
            y_true = target_transform(tuabn_valid.get_metadata()['target'])
        else:
            y_pred = len(tuabn_valid.description)*[mean_train_age]
            y_true = tuabn_valid.description['age']
        dummy_score = mean_absolute_error(
            y_pred=y_pred,
            y_true=y_true,
        )
    elif target_name in ['gender', 'pathological', 'age_clf']:
        loss_key = 'misclass' if fast_mode == 0 else 'loss'
        loss_name = '1 - Accuracy' if fast_mode == 0 else 'NLLLoss'
        # dummy score is the less frequent class in train predicted for all valid trials
        class_distribution = tuabn_train.description[name].value_counts()/len(tuabn_train.description)
        label = class_distribution.idxmin()
        logger.debug(f'min class dist {label}')
        dummy_score = 1 - balanced_accuracy_score(
            y_true=tuabn_valid.description[name], 
            y_pred=[label]*len(tuabn_valid.description),
        )
    logger.debug(f'learning dummy score {dummy_score}')
    fig, ax = plt.subplots(1, 1, figsize=(15,3))
    ax = plot_learning(
        df=df,
        loss_key=loss_key,
        loss_name=loss_name,  # TODO: update loss name to input huber, mae, mse....
        dummy_score=dummy_score,
        dummy_score_name='dummy',
        test_name=test_name,
        ax=ax,
    )
    ax.set_title(title)
    save_fig(fig, out_dir, 'curves')

    # skip everything below
    if fast_mode == 1:
        return

    # compute final window and rec predictions and store
    # add pathological, age, and gender since it is useful for post-hoc analysis
    # create some plots and also store. can also be created based on saved pred files
    if target_name in ['age', 'age_clf']:
        add_names = ['pathological', 'gender']
    elif target_name in ['pathological', 'gender']:
        add_names = ['age', 'pathological'] if target_name == 'gender' else ['age', 'gender']
        
    if target_name == 'age':
        n_x, n_y = 2, 2
        fig, ax_arr = plt.subplots(n_x, n_y, figsize=(15, 4), sharex=True, sharey=True)
        fig2, ax_arr2 = plt.subplots(n_x, n_y, figsize=(12, 12), sharex=True, sharey=True)
    for x, kind in enumerate(['window', 'trial']):
        for y, (ds_name, ds) in enumerate(zip(['train', 'valid'], [tuabn_train, tuabn_valid])):
            if kind == 'window':
                y_pred = estimator.predict(ds).ravel()
                if target_name == 'age':
                    y_pred = target_transform.invert(y_pred).ravel()
                y_true = ds.get_metadata()['target'].to_numpy().ravel()
                adds = [ds.get_metadata()[add_name].to_numpy().ravel() for add_name in add_names]
            else:
                y_pred, y_true = estimator.predict_trials(ds)
                # make sure there are no predictions outside expected range [0,1]
                miny, maxy = 0, 1
                if any([((y < miny) | (y > maxy)).any() for y in y_pred]):
                    logger.warning(f'Found {kind} prediction values outside expected range [{miny}, {maxy}]')
                # y_pred: batch x n_classes x n_preds_per_input
                y_pred = np.array([np.mean(y_pred_, axis=1) for y_pred_ in y_pred])
                if x == 1 and y == 1:
                    logger.debug(f'some predictions {y_pred[:3]}')
                if target_name == 'age':
                    y_pred = target_transform.invert(y_pred).ravel()
                    # make sure there are no predictions outside expected range [min/max train age]
                    if any([((y < target_transform.add_element) | (y > target_transform.mult_element)).any() for y in y_pred]):
                        logger.warning(f'Found {kind} prediction values outside expected range [{target_transform.add_element}, {target_transform.mult_element}] (min: {y_pred.min()}, max: {y_pred.max()})')
                    y_true = target_transform.invert(y_true).ravel()
                elif target_name in ['pathological', 'gender']:
                    y_pred = y_pred.argmax(axis=1).ravel()
                adds = [ds.description[add_name].to_numpy().ravel() for add_name in add_names]

            # assemble df and store
            d = np.vstack([y_pred, y_true] + adds).T
            pred_df = pd.DataFrame(data=d, columns=['y_pred', 'y_true'] + add_names)
            this_out_dir = os.path.join(out_dir, 'preds')
            if not os.path.exists(this_out_dir):
                os.makedirs(this_out_dir)
            pred_df.to_csv(os.path.join(this_out_dir, f'{ds_name}_{kind}_preds.csv'))

            if target_name == 'age':
                # visualize age gap proxy accuracy
                ax = ax_arr[x, y]
                # error prone. order matters. first train, then valid
                if ds_name == 'train':
                    class_distribution = pred_df.pathological.value_counts()/len(pred_df)
                    label = class_distribution.idxmax()
                    # dummy is the more frequent class label in train assigned to all test examples 
                    #if y == 0:
                        #logger.debug(f"first: y_pred {np.unique(y_pred)}, y_true {np.unique(y_true)}")
                    gap_dummy = balanced_accuracy_score(
                        y_pred=pred_df['pathological'].to_list(),
                        y_true=[label]*len(pred_df)
                    ) * 100
                if x == 0 and y == 0:
                    logger.debug(f'age gap proxy dummy score {gap_dummy}')
                plot_thresh_to_acc(pred_df, ax=ax, dummy=gap_dummy)
                if y != 0:
                    ax.set_ylabel('')
                if x != (n_x-1):
                    ax.set_xlabel('')
                ax.set_title(f'{ds_name}, {kind}')
                fig.subplots_adjust(wspace=.05)
                save_fig(fig, out_dir, 'thresh_to_acc')

                # visualize chronological vs predicted age
                ax = ax_arr2[x, y]
                # error prone. order matters. first train, then valid
                if ds_name == 'train':
                    # dummy is the average train age
                    dummy = pred_df.y_true.mean()
                if x == 0 and y == 0:
                    logger.debug(f'chronological vs predicted dummy {dummy:.2f}')
                plot_chronological_vs_predicted_age(pred_df, ax=ax, dummy=dummy)
                if y != 0:
                    ax.set_ylabel('')
                if x != (n_x-1):
                    ax.set_xlabel('')
                ax.set_ylim(-10,120)
                ax.set_xlim(-10,110) 
                ax.set_title(f'{ds_name}, {kind}')
                fig2.subplots_adjust(wspace=.05)
                fig2.subplots_adjust(hspace=.1)
                save_fig(fig2, out_dir, 'chronological_vs_predicted_age')
    

def predict_longitudinal_datasets(
    data_scaler,
    target_scaler,
    out_dir,
    tmin,
    tmax,
    n_jobs,
    window_size_samples,
    n_channels,
    n_preds_per_input,
    preload,
):
    logger.info(f"predicting longitudinal datasets")
    for point in ['valid_best']:  # 'train_end', 'valid_best', checkpoint etc?
        clf_path = os.path.join(out_dir, 'checkpoint', f'{point}_model.pkl')
        if not os.path.exists(clf_path):
            logger.warning(f"{clf_path} not found")
            continue
        with open(clf_path, 'rb') as f:
            clf = pickle.load(f)
        for kind in ['transition', 'pathological', 'non_pathological']:
            ds_path = f'/work/longitudinal/{kind}.pkl'
            if not os.path.exists(ds_path):
                logger.warning(f"{ds_path} not found")
                continue
            with open(ds_path, 'rb') as f:
                ds = pickle.load(f)
            logger.debug(f"from model at {point} predicting longitudinal {kind}")
            trial_preds = predict_longitudinal_dataset(
                clf,
                ds,
                data_scaler,
                target_scaler,
                tmin,
                tmax,
                n_jobs,
                window_size_samples,
                n_channels,
                n_preds_per_input,
                preload,
            )
            out_path = os.path.join(out_dir, 'preds', f'{point}_longitudinal_{kind}_trial_preds.csv')
            # combine predictions with description dataframe
            df = ds.description
            df['y_pred'] = trial_preds
            df.to_csv(out_path)
            # TODO: plot_joint_scatter, plot_age_gap_hist


def predict_longitudinal_dataset(
    clf,
    ds,
    data_scaler,
    target_scaler,
    tmin,
    tmax,
    n_jobs,
    window_size_samples,
    n_channels,
    n_preds_per_input,
    preload,
    trialwise=True,
    average=True,
):
    # preprocess
    """
    ds = _preprocess(
        ds,
        tmin,
        tmax,
        n_jobs,
    )
    """
    # create windows
    ds = _create_windows(
        mapping=None,
        tuabn=ds,
        window_size_samples=window_size_samples,
        n_channels=n_channels,
        n_jobs=n_jobs,
        preload=preload,
        n_preds_per_input=n_preds_per_input,
    )
    # set data and target scaling transform
    ds.transform = data_scaler
    ds.target_transform = target_scaler

    # make trialswise predictions
    if trialwise:
        preds = clf.predict_trials(ds, return_targets=False)
        if average:
            preds = [np.mean(p, axis=-1).item() for p in preds]
    else:
        preds = clf.predict(ds)
    preds = [target_scaler.invert(p) for p in preds]
    return preds


def plot_learning(
    df,
    loss_key,
    loss_name,
    test_name,
    dummy_score=None,
    dummy_score_name='', 
    ax=None,
):
    # n_restarts, the times when learning rate gets bigger again
    n_restarts = (df['event_lr'].diff() > 0).sum() + 1
    n_epochs = len(df)
    n_epochs_per_restart = n_epochs//n_restarts

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12,4))

    ax.plot(df[f'train_{loss_key}'], 'blue', linewidth=1)
    y = [
        df[f'train_{loss_key}'][i*n_epochs_per_restart:(i+1)*n_epochs_per_restart].min() 
        for i in range(n_restarts)
    ]
    x = [
        df[f'train_{loss_key}'][i*n_epochs_per_restart:(i+1)*n_epochs_per_restart].argmin()+(i*n_epochs_per_restart) 
        for i in range(n_restarts)
    ]
    if n_restarts > 1:
        ax.plot(x, y, c='c', linestyle='--', linewidth=1)

    ax.plot(df[f'{test_name}_{loss_key}'], 'r', linewidth=1)
    y = [
        df[f'{test_name}_{loss_key}'][i*n_epochs_per_restart:(i+1)*n_epochs_per_restart].min() 
        for i in range(n_restarts)
    ]
    x = [
        df[f'{test_name}_{loss_key}'][i*n_epochs_per_restart:(i+1)*n_epochs_per_restart].argmin()+(i*n_epochs_per_restart) 
        for i in range(n_restarts)
    ]
    if n_restarts > 1:
        ax.plot(x, y, c='orange', linestyle='--', linewidth=1)

    if dummy_score is not None:
        ax.axhline(dummy_score, c='m', linewidth=1)

    # plot restarts
    """
    ymin = df[[f'train_{loss_key}',f'{test_name}_{loss_key}']].min().min()
    ymax = df[[f'train_{loss_key}',f'{test_name}_{loss_key}']].max().max()
    ymax_ = dummy_score if dummy_score is not None else ymax
    [ax.plot([x, x], [ymin, ymax_], c='k', linestyle='-', linewidth=1) 
     for x in range(n_epochs_per_restart, n_epochs, n_epochs_per_restart)]
    """
    [ax.axvline(x, c='k', linewidth=1, linestyle='-') 
     for x in range(n_epochs_per_restart, n_epochs, n_epochs_per_restart)]
    
    train_score = df[f'train_{loss_key}'].iloc[-1]
    test_score = df[f'{test_name}_{loss_key}'].iloc[-1]
    if n_restarts == 1:
        legend = [f'train ({train_score:.2f})', f'{test_name} ({test_score:.2f})']
        if dummy_score is not None:
            legend.append(f'{dummy_score_name} ({dummy_score:.2f})')
    else:
        legend = [f'train ({train_score:.2f})', f'train_trend', f'{test_name} ({test_score:.2f})', f'{test_name}_trend']
        if dummy_score is not None:
            legend.append(f'{dummy_score_name} ({dummy_score:.2f})')
        legend.append('restart')

    ax.legend(legend, loc='best', ncol=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{loss_name}')
    #ax.set_xticks(np.arange(len(df)), np.arange(1, len(df)+1))
    #ax.set_title(f'TUAB, {n}, {tmin}s–{tmax}s, age, {model_name}')
    return ax


def plot_thresh_to_acc(
    df,
    ax=None,
    dummy=None,
):
    # if we only decode normals or abnormals, this will raise 
    #if df.pathological.nunique() == 1:
    #    warnings.filterwarnings("ignore", message="y_pred contains classes not")
    sorted_gaps = (df['y_true'] - df['y_pred']).sort_values().to_numpy()
    gaps = df['y_true'] - df['y_pred']
    
    accs = []
    for thresh in sorted_gaps:
        y_true=df['pathological'].to_numpy(dtype=int)
        y_pred=(gaps < thresh).to_numpy(dtype=int)
        #logger.debug(f"second: y_pred {np.unique(y_pred)}, y_true {np.unique(y_true)}")
        accs.append(
            balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        )
    df = pd.DataFrame([sorted_gaps, accs, y_true]).T
    df.columns = ['thresh', 'acc', 'pathological']
    df['acc'] *= 100
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10,2))
    ax.plot(df['thresh'], df['acc'], c='b' if 0 in df.pathological.unique() else 'g')  # does not make sense in mixed case
    ax.set_ylabel('Accuracy [%]')
    ax.set_xlabel('Decoded Age – Chronological Age [years]')
    if dummy is not None:
        ax.axhline(dummy, c='m', linewidth=1)
    return ax


def plot_chronological_vs_predicted_age(
    df,
    dummy=None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    
    if df.pathological.nunique() == 1:
        color = 'b' if 0 in df.pathological.unique() else 'r'
        color2 = 'b' if 0 in df.pathological.unique() else 'r'
        label = 'non-pathological' if 0 in df.pathological.unique() else 'pathological' 
        ax.scatter(df.y_true.to_numpy('int'), df.y_pred.to_numpy('float'), marker='.', s=5, c=color, label=label)
        # plot a trend line
        m, b = np.polyfit(df.y_true.to_numpy('int'), df.y_pred.to_numpy('float'), 1)
        ax.plot(df.y_true, m*df.y_true + b, c=color2, label='trend', linewidth=1) 

    # order of plotting is a bit weird here to get the legend in the desired order...
    color_map = OrderedDict({'non-pathological': 'b', 'pathological': 'r'})
    if df.pathological.nunique() > 1:
        for patho_key, (patho, color) in enumerate(color_map.items()):
            this_df = df[df.pathological==patho_key]
            ax.scatter(this_df.y_true.to_numpy('int'), this_df.y_pred.to_numpy('float'), marker='.', s=5,
                       c=color, label=patho)

    ax.plot([0, 100], [0, 100], c='k', linewidth=1, label='identity')

    if df.pathological.nunique() > 1:
        for patho_key, (patho, color) in enumerate(color_map.items()):
            this_df = df[df.pathological==patho_key]
            m, b = np.polyfit(this_df.y_true.to_numpy('int'), this_df.y_pred.to_numpy('float'), 1)
            ax.plot(this_df.y_true, m*this_df.y_true + b, c=color, 
                    linewidth=1, label=f'{patho} trend')

    ax.set_xlabel('Chronological Age [years]')
    ax.set_ylabel('Decoded Age [years]')

    if dummy is not None:
        ax.axhline(dummy, label='dummy', c='m', linewidth=1)   

    ax.legend(ncol=2, loc='upper center')
    return ax


def plot_joint_scatter(
    all_preds_df,
):
    grid = sns.jointplot(data=all_preds_df, x='y_pred', y='y_true', hue='pathological', alpha=.75, height=4.85)
#     sns.scatterplot(
#         data=all_preds_df.groupby('pathological', as_index=False).mean(),  # median looks better
#         ax=grid.ax_joint, hue='pathological', x='y_pred', y='y_true', marker='^',
#         edgecolor='black', s=100, legend=False, alpha=.75,
#     )
    df = all_preds_df[all_preds_df.pathological]
    if not df.empty:
        m, b = np.polyfit(df.y_true.to_numpy('int'), df.y_pred.to_numpy('float'), 1)
        grid.ax_joint.plot(df.y_true, m*df.y_true + b, label='trend', linewidth=1, c='g')
        sns.scatterplot(
            data=df.groupby('pathological', as_index=False).mean(),  # median looks better
            ax=grid.ax_joint, hue='pathological', x='y_pred', y='y_true', marker='^',
            edgecolor='black', s=100, legend=False, alpha=.75, palette=['g'],
        )
    df = all_preds_df[~all_preds_df.pathological]
    if not df.empty:
        m, b = np.polyfit(df.y_true.to_numpy('int'), df.y_pred.to_numpy('float'), 1)
        grid.ax_joint.plot(df.y_true, m*df.y_true + b, label='trend', linewidth=1, c='b')
        sns.scatterplot(
            data=df.groupby('pathological', as_index=False).mean(),  # median looks better
            ax=grid.ax_joint, hue='pathological', x='y_pred', y='y_true', marker='^',
            edgecolor='black', s=100, legend=False, alpha=.75, palette=['b'],
        )
    
    #grid.ax_joint.set_xlim(-25, 125)
    #grid.ax_joint.set_ylim(-25, 125)
    grid.ax_joint.plot([0,100],[0,100], c='m', linestyle='--')
    #grid.ax_joint.plot([0, 100], [0, 0], c='k')
    #grid.ax_joint.plot([0, 0], [0, 100], c='k')
    #grid.ax_joint.plot([0, 100], [100, 100], c='k')
    #grid.ax_joint.plot([100, 100], [0, 100], c='k')
    grid.ax_joint.set_xlabel('Decoded age [years]')
    grid.ax_joint.set_ylabel('Chronological age [years]')
    return grid
    
    
def plot_age_gap_hist(
    df,
    ax=None,
):
    df['gap'] = df.y_pred - df.y_true
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12,3))
    ax = sns.histplot(data=df, x='gap', 
                      hue='pathological' if df.pathological.nunique() != 1 else None,
                      #stat='percent', 
                      color='b' if 0 in df.pathological.unique() else 'g',
                      ax=ax, kde=True)#, ax=ax, binwidth=5)
    ax.set_xlabel('Decoded age - Chronological age [years]')
    ax.set_title(f'Brain age gap')
    return ax


def save_fig(
    fig,
    out_dir, 
    title,
):
    for file_type in ['pdf', 'png', 'jpg']:
        out_path = os.path.join(out_dir, 'plots', f'{file_type}')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        fig.savefig(
            os.path.join(out_path, f'{title}.{file_type}'),
            bbox_inches='tight',
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # args for decoding
    parser.add_argument('--augment', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--debug', type=int)
    parser.add_argument('--fast-mode', type=int)
    parser.add_argument('--final-eval', type=int)
    parser.add_argument('--intuitive-training-scores', type=int)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--n-jobs', type=int)
    parser.add_argument('--n-restarts', type=int)
    parser.add_argument('--n-train-recordings', type=int)
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--preload', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--shuffle-data-before-split', type=int)
    parser.add_argument('--split-kind', type=str)
    parser.add_argument('--squash-outs', type=int)
    parser.add_argument('--standardize-data', type=int)
    parser.add_argument('--standardize-targets', type=int)
    parser.add_argument('--subset', type=str)
    parser.add_argument('--target-name', type=str)
    parser.add_argument('--tmin', type=int)
    parser.add_argument('--tmax', type=int)
    parser.add_argument('--valid-set-i', type=int)
    parser.add_argument('--window-size-samples', type=int)
    # args for storing run details
    parser.add_argument('--run-name', type=str)
    args, unknown = parser.parse_known_args()
    args = vars(args)
    s = pd.Series(args)
    if unknown:
        raise ValueError(f'There are unknown input parameters: {unknown}')

    run_name = args.pop('run_name')
    logger.info(f"This is run {run_name}")
    # run the actual code
    decode_tueg(**args, config=s)
