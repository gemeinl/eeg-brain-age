import os
import re
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
from sklearn.metrics import mean_absolute_error, balanced_accuracy_score, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, Checkpoint, TrainEndCheckpoint, ProgressBar, BatchScoring
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from skorch.utils import valid_loss_score, noop

from braindecode.datasets import BaseDataset, BaseConcatDataset
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
# TODO: look at the (negative) outliers reports. why are they outliers?
def decode_tueg(
    batch_size,
    condition,
    config,
    data_path,
    date,
    debug,
    final_eval,
    intuitive_training_scores,
    max_age,
    min_age,
    model_name,
    n_epochs,
    n_jobs,
    n_restarts,
    n_train_recordings,
    out_dir,
    preload,
    seed,
    shuffle_data_before_split,
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
    out_dir = os.path.join(out_dir, date, str(seed), str(valid_set_i))
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
        batch_size, condition, config, data_path, debug, final_eval, intuitive_training_scores,
        max_age, min_age, model_name, n_epochs, n_jobs, n_restarts, n_train_recordings, 
        out_dir, preload, seed, shuffle_data_before_split, squash_outs, 
        standardize_data, standardize_targets, subset, target_name, tmax, tmin, 
        valid_set_i, window_size_samples, augment, loss, logger,
    )

    #log_capture_string = get_log_capturer(logger, debug)
    level = logging.DEBUG if debug == 1 else logging.INFO
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

    competition = 1
    if competition == 1:
        tuabn_train, tuabn_valid, mapping, valid_rest, valid_rest_name = get_competition_datasets(
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
            min_age,
            max_age,
            condition,
        )
    else:
        tuabn_train, tuabn_valid, mapping, valid_rest, valid_rest_name  = get_datasets(
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
            min_age,
            max_age,
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
    n_preds_per_input = get_n_preds_per_input(
        model,
        n_channels,
        window_size_samples,
    )
    tuabn_train, tuabn_valid = create_windows(
        mapping, 
        tuabn_train,
        tuabn_valid,
        window_size_samples,
        n_jobs,
        preload,
        n_preds_per_input,
        test_name(final_eval),
    )
    tuabn_train, tuabn_valid = standardize(
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
        tuabn_train.target_transform,
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
    estimator = set_augmentation(
        augment,
        ch_names,
        seed,
        estimator,
        sfreq,
        len(tuabn_train),
        batch_size,
        competition,
    )
    logger.info(title)
    logger.info(f'starting training')
    estimator.fit(tuabn_train, y=None)
    logger.info(f'finished training')
    # generate simple output
    df = pd.DataFrame(estimator.history)
    save_csv(df, out_dir, 'history.csv')
    # there is one transform per dataset and one target_transform per concat dataset
    with open(os.path.join(out_dir, 'data_scaler.pkl'), 'wb') as f:
        pickle.dump(tuabn_train.transform[0], f)
    with open(os.path.join(out_dir, 'target_scaler.pkl'), 'wb') as f:
        pickle.dump(tuabn_train.target_transform, f)
    train_preds, valid_preds, scores = create_final_scores(
        estimator,
        tuabn_train,
        tuabn_valid,
        test_name(final_eval),
        target_name,
        tuabn_train.target_transform,
        tuabn_train.transform[0],
        n_jobs,
    )
    pred_path = os.path.join(out_dir, 'preds')
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    save_csv(train_preds, pred_path, 'train_end_train_preds.csv')
    save_csv(valid_preds, pred_path, f'train_end_{test_name(final_eval)}_preds.csv')
    save_csv(scores, out_dir, 'train_end_scores.csv')
    
    if competition == 1:
        return

    # TODO: rename valid_rest to smth meaningfull
    # TODO: move stuff below into function
    # predict valid rest and longitudinal datasets
    for ds_name in [valid_rest_name]:#, 'transition', 'non_pathological', 'pathological']:
        if ds_name == valid_rest_name:
            ds = valid_rest
        else:
            ds = get_longitudinal_ds(ds_name, (min_age, max_age))
        if ds is None:
            continue
        logger.debug(f"dataset {ds_name} n={len(ds.datasets)}")
        logger.debug('preprocessing')
        ds = preprocess(
            ds, 
            preprocessors=get_preprocessors(tmin, tmax), 
            n_jobs=n_jobs,
        )
        logger.debug('windowing')
        ds = _create_windows(
            ds,
            window_size_samples,
            n_jobs, 
            preload,
            n_preds_per_input,
            mapping,
        )
        logger.debug('predicting')
        ds_preds, ds_score = _create_final_scores(
            estimator,
            ds,
            ds_name,
            target_name,
            tuabn_train.target_transform,
            tuabn_train.transform[0],
            n_jobs,
        )
        save_csv(ds_preds, pred_path, f'train_end_{ds_name}_preds.csv')
        scores = pd.concat([scores, ds_score], axis=1)
        save_csv(scores, out_dir, 'train_end_scores.csv')
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
    condition,
    config,
    data_path,
    debug,
    final_eval,
    intuitive_training_scores,
    max_age,
    min_age,
    model_name, 
    n_epochs,
    n_jobs,
    n_restarts,
    n_train_recordings,
    out_dir,
    preload,
    seed,
    shuffle_data_before_split,
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
    if target_name != 'age' and standardize_targets:
        logger.warning(f"'standardize_targets' without effect with this target ({target_name})")
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
    if augment not in [
        'dropout', 'shuffle', 'fliplr', 'random', 
        'reverse', 'sign', 'noise', 'mask', 'flipfb',
        'identity', '0',
    ]:
        raise ValueError(f"Unknown augmentation {augment}.")
    if loss not in ['mse', 'mae', 'log_cosh', 'huber', 'nll']:
        raise ValueError(f'Unkown loss {loss}')
    if target_name in ['pathological', 'gender', 'age_clf'] and loss != 'nll':
        raise ValueError(f"loss '{loss}' cannot be used with this target ({target_name})")
    if squash_outs not in [0, 1]:
        raise ValueError
    if squash_outs == 1 and target_name != 'age':
        logger.warning(f"'squash_outs' without effect with this target ({target_name})")
    if final_eval == 0:
        assert valid_set_i in [0, 1, 2, 3, 4]
    if max_age != -1:
        assert isinstance(max_age, int)
    if min_age != -1:
        assert isinstance(min_age, int)
    if condition not in ['EC', 'EO', 'all']:
        raise ValueError


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
        # if we crop to a specific time, we don't inlcude tmax
        # if we crop to the full rec aka tmax=None, we do include tmax, because otherwise we loose a sample
        include_tmax = True if tmax is None else False
        preprocessors.append(Preprocessor('crop', tmin=tmin, tmax=tmax, include_tmax=include_tmax))
    if ch_names is not None:
        logger.debug(f"adding channel picker {ch_names}")
        preprocessors.append(
            Preprocessor('pick_channels', ch_names=['EEG ' + ch for ch in ch_names], ordered=True))
    if lfreq is not None or hfreq is not None:
        logger.debug(f"adding filter {lfreq} – {hfreq}")
        preprocessors.append(Preprocessor('filter', l_freq=lfreq, h_freq=hfreq))
    return preprocessors


def get_train_eval_datasets(tuabn_train, target_name):
    if target_name in ['age', 'age_clf']:
        logger.debug(f"into train (0.9) and eval (0.1).")
        train_ids, eval_ids = train_eval_split(tuabn_train.description)
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


def get_train_valid_datasets(tuabn_train, target_name, valid_set_i):
    logger.info(f"validation run, removing eval from dataset with {len(tuabn_train.description)} recordings")
    tuabn_train, _ = get_train_eval_datasets(tuabn_train, target_name)
    shuffle = True if target_name in ['age', 'age_clf'] else False
    return _get_train_valid_datasets(tuabn_train, target_name, valid_set_i, shuffle)


def _get_train_valid_datasets(tuabn_train, target_name, valid_set_i, shuffle):
    logger.debug(f"splitting dataset with {len(tuabn_train.description)} recordings")
    logger.debug(f"into train (.8) and valid (.2).")
    # for pathology decoding, turn off shuffling and make time series split chronologically
    train_ids, valid_ids = train_valid_split(tuabn_train.description, valid_set_i, shuffle)
    intersection = np.intersect1d(train_ids, valid_ids)
    if intersection.any():
        raise RuntimeError(f"leakage train / valid detected {intersection}")
    tuabn_valid = tuabn_train.split(valid_ids)['0']
    tuabn_train = tuabn_train.split(train_ids)['0']
    return tuabn_train, tuabn_valid


def match_pathological_distributions(d):
    d_p = d[d.pathological]
    d_n = d[~d.pathological]
#     print(d_p.shape, d_n.shape)

    max_age = max([d_p.age.max(), d_n.age.max()])

    subsampled = []
    for age in range(max_age+1):
        d_p_n = d_p[d_p.age==age]
        d_n_n = d_n[d_n.age==age]
        min_n = min(len(d_p_n), len(d_n_n))
#         print(age, min_n)
        subsampled.append(d_p_n.head(min_n))
        subsampled.append(d_n_n.head(min_n))
    subsampled = pd.concat(subsampled)
    return subsampled


def subsample_uniformly(d):
    min_age = 18
    max_age = 86
    n_per_age = 6
    d_uniform = pd.concat([g.head(n_per_age) for n, g in d.groupby(['pathological', 'age']) if (n[1] >= min_age) and (n[1] <= max_age)], axis=0)
    return d_uniform


def add_ages_from_additional_sources(ds):
    for d_ in ds.datasets:
        # manually add back missing reports here! -.-
        d_.description['report'] = TUHAbnormal._read_physician_report(
            d_.description.path.replace('TUH_PRE', 'TUH').replace('.edf', '.txt'),
        )
        rec_year = d_.raw.info['meas_date'].year
        # seems like one (?) header broke in preprocessing. read header of original unpreprocessed reocrding
        header = TUHAbnormal._read_edf_header(d_.description.path.replace('TUH_PRE', 'TUH'))
        pattern = r'\d\d-\w\w\w-(\d\d\d\d)'
        matches = re.findall(pattern, str(header))
        if len(matches) != 1:
            birthyear = np.nan
        else:
            birthyear = int(matches[0])
        d_.description['date_age'] = int(rec_year) - birthyear
        pattern = r'(\d+)[ -]+?[years]{3,5}[ -]+?[old]{3}'
        matches = re.findall(pattern, d_.description.report)
        if len(matches) >= 1:
            # assume report always starts with 'XX year old ...'
            match = int(matches[0])
        elif len(matches) == 0:
            match = np.nan
        d_.description['report_age'] = match
    return ds


def get_competition_datasets(
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
    min_age,
    max_age,
    condition,
):
    train_subj = 1200  # use 10 instead of 1200 training subjects, for demonstration purpose
    test_subj = 400  # use 10 instead of 400 testing subjects, for demonstration purpose
    sfreq = 100
    with open(os.path.join(data_path, 'train.pkl'), 'rb') as f:
        tuabn_train = pickle.load(f)

    # order dataset from all EC, then all EO to
    # EC, EO of subject 1 to EC, EO of last subjects
    # to easily redruce subject leakage in splitting
    splits = tuabn_train.split('subject')
    tuabn_train = BaseConcatDataset([d for s, d in splits.items()])
    if not tuabn_train.description['condition'][0] != tuabn_train.description['condition'][1]:
        raise RuntimeError

    # train_raws = {}
    # for condition in ["EC", "EO"]:
    #     train_raws[condition] = []
    #     train_subjs = list(range(1, train_subj + 1))
    #     for s in train_subjs:
    #         fname = os.path.join(data_path, f"subj{s:04}_{condition}{sfreq}_hz_raw.fif.gz")
    #         raw = mne.io.read_raw(fname, preload=False, verbose='error')
    #         train_raws[condition].append(raw)
    # meta = pd.read_csv(os.path.join(data_path, "train_subjects.csv"), index_col=0)
    # meta = pd.concat([meta, meta])
    # meta['condition'] = len(train_raws['EC']) * ['EC'] + len(train_raws['EO']) * ['EO']
    # train_raws = train_raws['EC'] + train_raws['EO']
    # tuabn_train = BaseConcatDataset([
    #     BaseDataset(raw, target_name=target_name) for raw in train_raws
    # ])
    # meta['subject'] = meta['id']
    # tuabn_train.set_description(meta)
    
    # TODO: implement splitting if both conditions should be used
    if condition in ['EC', 'EO']:
        logger.debug(f'using condition {condition} data only')
        tuabn_train = tuabn_train.split('condition')[condition]
    if final_eval == 1:
        with open(os.path.join(data_path.replace('training', 'testing'), 'test.pkl'), 'rb') as f:
            tuabn_valid = pickle.load(f)
    else:
        tuabn_train, tuabn_valid = _get_train_valid_datasets(
            tuabn_train, target_name, valid_set_i, False)
    if n_train_recordings != -1:
        tuabn_train = tuabn_train.split([list(range(n_train_recordings))])['0']
    if tmin != -1 or tmax != -1:
        preprocessors = get_preprocessors(tmin, tmax)
        [tuabn_train, tuabn_valid] = [
            preprocess(ds, preprocessors=preprocessors, n_jobs=n_jobs) for ds in [tuabn_train, tuabn_valid]
        ]
    return tuabn_train, tuabn_valid, None, None, None
    

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
    min_age,
    max_age,
):
    logger.debug("indexing files")
    tuabn_train = TUHAbnormal(
        path=data_path,
        preload=False,
        add_physician_reports=True,
        n_jobs=n_jobs,
        target_name = 'age' if target_name in ['age', 'age_clf'] else target_name,
    )
    
    exclude_derivating_ages = 1
    if exclude_derivating_ages != -1:
        logger.info(f'rejecting recordings with > 1 year derivation in header, dates, and report')
        d = tuabn_train.description
        logger.debug(f'there are {d.pathological.sum()} patho and {len(d)-d.pathological.sum()} non-patho recordings in total')
        # add ages parsed from medical report and computed from recording export date and birth year of patient
        tuabn_train = add_ages_from_additional_sources(tuabn_train)
        # if the difference is bigger than one year, possible due to anonymization, reject the recording
        c1 = (tuabn_train.description.age - tuabn_train.description.report_age).abs() < 2
        c2 = (tuabn_train.description.age - tuabn_train.description.date_age).abs() < 2
        c3 = (tuabn_train.description.date_age - tuabn_train.description.report_age).abs() < 2
        ids = tuabn_train.description[c1 & c2 & c3].index.to_list()
        tuabn_train = tuabn_train.split([ids])['0']
        d = tuabn_train.description
        logger.debug(f'there are {d.pathological.sum()} patho and {len(d)-d.pathological.sum()} non-patho recordings left')
    
    subsample = -1
    if subsample != -1:
        logger.info(f'subsampling age distributions for pathological and non-pathological recordings ({subsample})')
        d = tuabn_train.description
        logger.debug(f'there are {d.pathological.sum()} patho and {len(d)-d.pathological.sum()} non-patho recordings in total')
        if subsample == 'match':
            d = match_pathological_distributions(d)
        elif subsample == 'uniform':
            d = subsample_uniformly(d)
        else:
            raise ValueError
        tuabn_train = tuabn_train.split([d.sort_index().index.to_list()])['0']
        d = tuabn_train.description
        logger.debug(f'there are {d.pathological.sum()} patho and {len(d)-d.pathological.sum()} non-patho recordings left')
        
    if final_eval == 1:
        logger.debug(f"splitting dataset with {len(tuabn_train.description)} recordings")
        tuabn_train, tuabn_valid = get_train_eval_datasets(tuabn_train, target_name)
    else:
        tuabn_train, tuabn_valid = get_train_valid_datasets(tuabn_train, target_name, valid_set_i)

    # select normal/abnormal only
    logger.info(f"from train ({len(tuabn_train.datasets)}) and {test_name(final_eval)}"
                f" ({len(tuabn_valid.datasets)}) selecting {subset}")
    tuabn_train, _ = subselect(tuabn_train, subset)
    tuabn_valid, valid_rest = subselect(tuabn_valid, subset)
    logger.debug(f"selected train ({len(tuabn_train.datasets)}) and {test_name(final_eval)}"
                 f" ({len(tuabn_valid.datasets)})")
    logger.debug(f"valid_rest (aka not {subset}) has {len(valid_rest.datasets)}")

    # TODO: add male /female subselection?

    # select based on age
    if min_age != -1 or max_age != -1:
        logger.info(f"removing recordings of underage subjects")
        tuabn_train = subselect(dataset=tuabn_train, subset=(min_age, max_age))
        tuabn_valid = subselect(dataset=tuabn_valid, subset=(min_age, max_age))
        valid_rest = subselect(dataset=valid_rest, subset=(min_age, max_age))
        logger.debug(f"selected train ({len(tuabn_train.datasets)}) and {test_name(final_eval)}"
                     f" ({len(tuabn_valid.datasets)})")

    valid_rest_name = f'valid_not_{subset}'
    logger.debug(f"valid_rest (aka not {subset}) has {len(valid_rest.datasets)}")

    # reduce number of train recordings
    if n_train_recordings != -1:
        tuabn_train = tuabn_train.split([list(range(n_train_recordings))])['0']
        logger.debug(f"selected {n_train_recordings} train recordings")

    some_durations = [ds.raw.n_times/ds.raw.info['sfreq'] for ds in tuabn_train.datasets][:3]
    logger.debug(f'some raw durations {some_durations}')
    if tmin != -1 or tmax != -1:
        logger.debug("preprocessing")
        preprocessors = get_preprocessors(tmin, tmax)
        [tuabn_train, tuabn_valid] = [
            preprocess(ds, preprocessors=preprocessors, n_jobs=n_jobs) for ds in [tuabn_train, tuabn_valid]
        ]
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
    return tuabn_train, tuabn_valid, mapping, valid_rest, valid_rest_name


def subselect(
    dataset,
    subset,
):
    if subset in ['normal', 'abnormal', 'mixed']:
        return subselect_pathology(dataset=dataset, subset=subset)
    elif subset in ['M', 'F']:
        return subselect_gender(dataset=dataset, subset=subset)
    elif isinstance(subset, tuple):
        assert len(subset) == 2
        return subselect_age(dataset=dataset, min_age=subset[0], max_age=subset[1])
    else:
        raise ValueError
    
    
def subselect_pathology(
    dataset,
    subset,
):
    # select normal / abnormal only
    rest_dataset = None
    if subset != 'mixed':
        k = 'pathological'
        v = 'False' if subset == 'normal' else 'True'
        not_v = 'True' if subset == 'normal' else 'False'
        splits = dataset.split(k)
        dataset = splits[v]
        rest_dataset = splits[not_v]
    return dataset, rest_dataset


def subselect_gender(
    dataset,
    subset,
):
    return dataset.split('gender')[subset]


def subselect_age(
    dataset,
    min_age,
    max_age,
):
    ages = dataset.description['age']
    if min_age != -1:
        ids = ages[(ages >= min_age)].index.to_list()
        dataset = dataset.split(ids)['0']
    if max_age != -1:
        ids = ages[(ages <= max_age)].index.to_list()
        dataset = dataset.split(ids)['0']
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
            n_classes = 1
            final_conv_length = 35
        model = ShallowFBCSPNet(
            in_chans=n_channels,
            n_classes=n_classes,
            input_window_samples=window_size_samples,
            final_conv_length=final_conv_length,
        )
    elif model_name == 'deep':
        lr = 0.01
        weight_decay = 0.5 * 0.001
        final_conv_length = 1
        if target_name == 'age':
#             lr = 0.005  # magical robin notebook  # smoother curves, worse valid mae score
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


def get_n_preds_per_input(
        model,
        n_channels,
        window_size_samples,
    ):
    n_preds_per_input = model(
        torch.ones(1, n_channels, window_size_samples, 1).to(next(model.parameters()).device)
    ).size()[2]
    logger.debug(f"model produces {n_preds_per_input} preds for every input of size {window_size_samples}")
    return n_preds_per_input


def create_windows(
    mapping,
    tuabn_train,
    tuabn_valid,
    window_size_samples,
    n_jobs, 
    preload,
    n_preds_per_input,
    test_name,
):
    logger.debug("windowing")
    [tuabn_train, tuabn_valid] = [
        _create_windows(
            ds,
            window_size_samples,
            n_jobs,
            preload,
            n_preds_per_input,
            mapping,
        )
        for ds in [tuabn_train, tuabn_valid]
    ]
    logger.debug(f'train windows {len(tuabn_train)}')
    logger.debug(f'{test_name} windows {len(tuabn_valid)}')
    return tuabn_train, tuabn_valid


def _create_windows(
    ds,
    window_size_samples,
    n_jobs, 
    preload,
    n_preds_per_input,
    mapping,
):
    return create_fixed_length_windows(
        ds,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples-n_preds_per_input,
        n_jobs=min(n_jobs, 4),
        preload=bool(preload),
        mapping=mapping,
        drop_last_window=False,
        drop_bad_windows=True,
        reject=None,
        flat=None,
    )


def train_eval_split(df):
    # fix the seed to always get identical splits
    seed = 20220429
    train, eval_ = train_test_split(df, test_size=1/10, random_state=seed)
    return sorted(train.index.to_list()), sorted(eval_.index.to_list())


def train_valid_split(df, valid_set_i, shuffle):
    # fix the seed to always get identical splits
    seed = 20220429
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


# def sine_scale(y, miny, maxy):
#     y = y - miny
#     y = y / (maxy - miny)
#     y = y * np.pi
#     y = y + np.pi
#     y = np.cos(y)
#     y = y + 1 
#     y = y / 2
#     return y


# def inv_sine_scale(y, miny, maxy):
#     y = y * 2
#     y = y - 1
#     # y could be outside of [-1,1]
#     # in those cases, arccos fails
#     # how to handle?
#     y = np.arccos(y)
#     y = y - np.pi
#     y = y / np.pi
#     y = y * (maxy + miny)
#     y = y + miny
#     return -y

    
class TargetScaler():
    def __init__(self, add_element=0, mult_element=1, kind='minmax'):
        self.add_element = add_element
        self.mult_element = mult_element
        self.kind = kind

    def __call__(self, y):
#         if self.kind == 'standard':
#             return (y - self.add_element) / self.mult_element
#         elif self.kind == 'exponential':
#             return np.power(((y - self.add_element) / self.mult_element), self.exp)
#             raise NotImplementedError
#         elif self.kind == 'sine':
#             return sine_scale(y, self.add_element, self.mult_element)
#         elif self.kind == 'percentage':
#             raise NotImplementedError
#         elif self.kind == 'sigmoid':
#             torch.special.expit((y - self.add_element) / self.mult_element)
#             raise NotImplementedError
        if self.kind == 'minmax':
            scale = (self.add_element / ((self.mult_element - self.add_element)))
            return (y / (self.mult_element - self.add_element)) - scale
        else:
            raise NotImplementedError

    def invert(self, y):
#         if self.kind == 'standard':
#             return y * self.mult_element + self.add_element
#         elif self.kind == 'exponential':
#             raise NotImplementedError
#             return (np.power(y, 1/self.exp) * self.mult_element) + self.add_element
#         elif self.kind == 'sine':
#             raise NotImplementedError
#             return inv_sine_scale(y, self.add_element, self.mult_element) 
#         elif self.kind == 'percentage':
#             raise NotImplementedError
#         elif self.kind == 'sigmoid':
#             torch.special.logit(y * self.mult_element + self.add_element)
#             raise NotImplementedError
        if self.kind == 'minmax':
            scale = (self.add_element / ((self.mult_element - self.add_element)))
            return (y + scale) * (self.mult_element - self.add_element)
        # TODO: check inversion of minmax scale! seemed ok
        else:
            raise NotImplementedError


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
    return tuabn_train, tuabn_valid


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
    for df, csv_name in [
        (config, 'config.csv'),
#         (train_description, 'train_description.csv'),
#         (valid_description, f'{test_name}_description.csv'),
    ]:
        save_csv(df, out_dir, csv_name)
    
    
def save_csv(df, out_dir, csv_name):
    df.to_csv(os.path.join(out_dir, csv_name))


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
        # TODO: average here, too?
        y_pred = model.predict(X)
    # TODO: derive target_scaler from X?
    y_true = target_scaler.invert(y_true)
    y_pred = target_scaler.invert(y_pred)
    return mae(y_true, y_pred)


def mae(
    y_true,
    y_pred,
):
    return float(mean_absolute_error(y_true=y_true, y_pred=y_pred))


def acc(
    y_true,
    y_pred,
):
    return balanced_accuracy_score(y_true=y_true.astype(int), y_pred=y_pred.astype(int))


def window_acc(
    model,
    X,
    y,
    return_y_yhat,
):
    return trial_acc(y, y_pred, trialwise=False, return_y_yhat=return_y_yhat)


# TODO: implement a binned acc / mae on classifiction?
def trial_acc(
    model,
    X,
    y,
    trialwise,
    return_y_yhat,
):
    if trialwise:
        y_pred, y_true = model.predict_trials(X)
        y_pred = np.array([np.mean(y_pred_, axis=1) for y_pred_ in y_pred])
    else:
        y_pred = model.predict(X)
    y_pred = y_pred.argmax(axis=1)
    return acc(y_true, y_pred, return_y_yhat)


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
    callbacks = []
    if test_name == 'eval':
        callbacks.extend([
            (f"{test_name}_loss", BatchScoring(
                valid_loss_score, name=f"{test_name}_loss", target_extractor=noop)),
        ])
    # add callbacks for cosine annealing with warm restarts and a checkpointer
    n_epochs_per_restart = int(n_epochs/(n_restarts+1))
    callbacks.extend([
        ("lr_scheduler", LRScheduler(
            CosineAnnealingWarmRestarts, T_0=n_epochs_per_restart, T_mult=1)),
        # ("progress_bar", ProgressBar()),  # unfortunately not working in jupyter?
    ])
    # compute the mean and std on train ages
    # scale both train and valid set ages accordingly
    # set up scoring functions that invert the scaling for both
    # such that intuitive performance tracking during training can take place
    if target_name == 'age':
        if intuitive_training_scores and fast_mode == 0:
            callbacks.extend([
                (f"MAE_{test_name}", CroppedTrialEpochScoring(
                    partial(trial_age_mae, target_scaler=target_scaler, return_y_yhat=False),
                    name=f'{test_name}_age_mae', on_train=False, 
                    lower_is_better=True)),#, avg_axis=2)),
                ("MAE_train", CroppedTrialEpochScoring(
                    partial(trial_age_mae, target_scaler=target_scaler, return_y_yhat=False),
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
        # TODO: eval_loss_best not found as name for default callback is 'valid_loss'
        monitor = f'{test_name}_loss_best'
    callbacks.extend([
        (f"best_{test_name}", Checkpoint(
            dirname=os.path.join(out_dir, 'checkpoint'), monitor=monitor, fn_prefix=f'{test_name}_best_',
            f_pickle='model.pkl')),  # load_best=True?
        ("after_train", TrainEndCheckpoint(
            dirname=os.path.join(out_dir, 'checkpoint'), f_pickle='model.pkl')),
    ])
    return callbacks


# def mean_percentage_error(input, target):
#     # TODO: need to use absolute value?
#     # does not seem to work well
#     e = target-input
#     return (target/e).mean()


# def mean_squared_percentage_error(input, target):
#     # does not seem to work well
#     e = target-input
#     return (target/e*target/e).mean()


# def log_cosh_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     # https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch
#     def _log_cosh(x: torch.Tensor) -> torch.Tensor:
#         return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
#     return torch.mean(_log_cosh(input - target))


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
    if loss == 'nll':
        loss_function = torch.nn.functional.nll_loss
    elif loss == 'mse':
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
#         logger.debug(f"With {n_examples} examples and batch size of {batch_size}")
#         logger.debug(f"expecting {self.calls_per_epoch} batches/calls per epoch")
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
    competition,
):
    # TODO: make augment a list of transformations
    if augment == '0':
        return estimator
    probability = 1
    augmentations = {
        'dropout': ChannelsDropout(probability=probability, p_drop=.2, random_state=seed),
        'mask': SmoothTimeMask(probability=probability, mask_len_samples=int(sfreq), random_state=seed),
        'noise': GaussianNoise(probability=probability, std=.1, random_state=seed),
        'reverse': TimeReverse(probability=probability, random_state=seed),
        'shuffle': ChannelsShuffle(probability=probability, p_shuffle=.2, random_state=seed),
        'sign': SignFlip(probability=probability, random_state=seed),
        'identity': IdentityTransform(),
        'identity': IdentityTransform(),
    }
    if competition != 1:
        augmentations.update({
            'flipfb': ChannelsSymmetryFB(probability=probability, ordered_ch_names=ch_names, random_state=seed),
            'fliplr': ChannelsSymmetry(probability=probability, ordered_ch_names=ch_names, random_state=seed),
        })

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
    return estimator


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
    data_scaler,
    n_jobs,
):
    train_preds, train_score = _create_final_scores(
        estimator,
        tuabn_train,
        'train',
        target_name,
        target_scaler,
        data_scaler,
        n_jobs,
    )
    valid_preds, valid_score = _create_final_scores(
        estimator,
        tuabn_valid,
        test_name,
        target_name,
        target_scaler,
        data_scaler,
        n_jobs,
    )
    scores = pd.concat([train_score, valid_score], axis=1)
    return train_preds, valid_preds, scores


def _create_final_scores(
    estimator,
    ds,
    ds_name,
    target_name,
    target_scaler,
    data_scaler,
    n_jobs,
):
    logger.info(f"on {ds_name} reached")
    preds, targets = predict_ds(
        estimator,
        ds, 
        target_name,
        target_scaler,
        data_scaler, 
        n_jobs,
        mem_efficient=True if target_name in ['age_clf'] else False,
        trialwise=True,
        average_time_axis=True,
    )
    preds = pd.DataFrame({'y_true': targets, 'y_pred': preds})  
    preds = pd.concat([preds, ds.description], axis=1)
    # always aggregate to subject-wise score
    subject_wise = True
    if subject_wise:
        # TODO: y_true are no ints after this operation anymore. pandas introduces floating point imprecisions...
        y_true = preds.groupby('subject')['y_true'].mean()
        y_pred = preds.groupby('subject')['y_pred'].mean()
        if target_name == 'age':
            epsilon = np.finfo(np.float64).eps
            scorings = [
                ('mae', mae),
                ('mdae', lambda y_true, y_pred: np.nanmedian(np.abs(y_true - y_pred))), 
                ('r2', r2_score),
                ('mape', lambda y_true, y_pred: np.nanmean(np.abs(np.abs(y_true - y_pred) / y_true))),
                ('mdape', lambda y_true, y_pred: np.nanmedian(np.abs(np.abs(y_true - y_pred) / y_true))), 
            ]
        else:
            scorings = [('acc', acc)]
    else:
        raise NotImplementedError
    scores = {ds_name: {}}
    for score_name, scoring_func in scorings:
        score = scoring_func(y_true=y_true, y_pred=y_pred)
        scores[ds_name].update({score_name: score})
        logger.info(f"{score:.2f} {score_name}")
    scores = pd.DataFrame(scores)
    return preds, scores


def generate_splits(n_datasets, n_jobs):
    n_splits = n_datasets/n_jobs if n_datasets % n_jobs == 0 else n_datasets/n_jobs+1
    return {str(i): list(b) for i, b in enumerate(np.array_split(list(range(n_datasets)), n_splits))}


def predict_ds(
    clf,
    ds, 
    target_name,
    target_scaler,
    data_scaler, 
    n_jobs,
    mem_efficient,
    trialwise=True,
    average_time_axis=True,
):
    ds.target_transform = target_scaler
    ds.transform = data_scaler
    if mem_efficient:
        splits = generate_splits(len(ds.datasets), n_jobs)
        splits = {i: ds.split(ids)['0'] for i, ids in splits.items()}
    else:
        splits = {'0': ds}
    all_preds, all_targets = [], []
    for d_i, d in splits.items():
        preds, targets = _predict_ds(
            clf,
            d,
            trialwise=trialwise,
            average_time_axis=average_time_axis,
        )
        # get class label from predictions
        if target_name != 'age':
            # TODO: preds currently not an ndarray here
            preds = np.argmax(preds, axis=-1)
        all_preds.append(preds)
        all_targets.append(targets)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return all_preds, all_targets


def _predict_ds(
    clf,
    ds, 
    trialwise=True,
    average_time_axis=True,
):
    if trialwise:
        preds, targets = clf.predict_trials(ds, return_targets=True)
    else:
        preds = clf.predict(ds)
        targets = ds.get_metadata['target'].to_numpy()
    if average_time_axis:
        preds = [np.mean(p, axis=-1).squeeze() for p in preds]
    if hasattr(ds, 'target_transform'):
        preds = [ds.target_transform.invert(p) for p in preds]
        targets = [ds.target_transform.invert(t) for t in targets]
    return preds, targets


def get_longitudinal_ds(kind, subset):
    try:
        ds_path = f'/work/longitudinal/{kind}.pkl'
        with open(ds_path, 'rb') as f:
            ds = pickle.load(f)
    except:
        try:
            ds_path = f'/home/jovyan/longitudinal/{kind}.pkl'
            with open(ds_path, 'rb') as f:
                ds = pickle.load(f)
        except:
            raise RuntimeError
    if subset not in ['normal', 'abnormal', 'mixed']:
        ds = subselect(ds, subset)
    return ds


def load_exp(
    base_dir, 
    exp, 
    checkpoint,
):
    with open(os.path.join(base_dir, exp, f'checkpoint/{checkpoint}_model.pkl'), 'rb') as f:
        clf = pickle.load(f)
    with open(os.path.join(base_dir, exp, 'data_scaler.pkl'), 'rb') as f:
        data_scaler = pickle.load(f)
    with open(os.path.join(base_dir, exp, 'target_scaler.pkl'), 'rb') as f:
        target_scaler = pickle.load(f)
    config = pd.read_csv(os.path.join(base_dir, exp, 'config.csv'), index_col=0).squeeze()
    return clf, data_scaler, target_scaler, config


def plot_learning_curves(histories, loss_name, valid_or_eval='Valid', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(12,3))
    for history in histories:
        ax = sns.lineplot(data=history, y='train_loss', x='epoch', ax=ax, c='g', linewidth=.5)
        ax = sns.lineplot(data=history, y='valid_loss', x='epoch', ax=ax, c='orange', linewidth=.5)

    mean_train_loss = np.mean([history['train_loss'] for history in histories], axis=0)
    sns.lineplot(x=history['epoch'], y=mean_train_loss, linestyle='--', c='g', 
                 label=f'Train ({mean_train_loss[-1]:.5f})')
    mean_valid_loss = np.mean([history['valid_loss'] for history in histories], axis=0)
    sns.lineplot(x=history['epoch'], y=mean_valid_loss, linestyle='--', c='orange', 
                 label=f'{valid_or_eval} ({mean_valid_loss[-1]:.5f})')
    ax.set_ylabel(loss_name)
    ax.set_xlabel('Epoch')
    ax.legend(title='Subset')
    return ax


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


# TODO: accept a thresh as input (valid thresh for eval preds)
def plot_thresh_to_acc(
    df,
    ax=None,
):
    # if we only decode normals or abnormals, this will raise 
    #if df.pathological.nunique() == 1:
    #    warnings.filterwarnings("ignore", message="y_pred contains classes not")
    sorted_gaps = df['gap'].sort_values().to_numpy()
    gaps = df['gap']

    accs = []
    for thresh in sorted_gaps:
        y_true=df['pathological'].to_numpy(dtype=int)
        y_pred=(gaps > thresh).to_numpy(dtype=int)
        #logger.debug(f"second: y_pred {np.unique(y_pred)}, y_true {np.unique(y_true)}")
        accs.append(
            balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        )
    gap_df = pd.DataFrame([sorted_gaps, accs, y_true]).T
    gap_df.columns = ['thresh', 'acc', 'pathological']
    gap_df['acc'] *= 100

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(12,3))

    # combine age gap hist with thresh to acc curve by creating a twin axis
    ax = plot_age_gap_hist(df, bin_width=1, ax=ax)
    ax.legend(title='Pathological', loc='upper left')
    # ax is thresh curve, ax1 is histogram
    ax1 = ax.twinx()

    df = gap_df
    if 0 not in df.pathological.unique():
        c='g' 
    elif 1 not in df.pathological.unique():
        c='b'
    else:
        c='k'

    ax1.plot(df['thresh'], df['acc'], c='lightgreen', label='Thresholds')#, zorder=3)  # does not make sense in mixed case

    ax1.set_ylabel('Accuracy [%]')
    ax.set_xlabel('Chronological Age − Decoded Age [years]')
#     ax.legend(title='Pathological', loc='lower left')

    xlim = max(abs(sorted_gaps)) * 1.1
    ax1.set_xlim(-xlim, xlim)

    ax1.scatter(
        sorted_gaps[(df.acc - 50).abs().argmax()], 
        df.acc[(df.acc - 50).abs().argmax()], 
        zorder=4, marker='*',  c='orange', s=50,
#         label=f'Best ({sorted_gaps[df.acc.argmax()]:.2f} years, {df.acc.max():.2f} %)'
        label=f'Best ({sorted_gaps[(df.acc - 50).abs().argmax()]:.2f} , {df.acc[(df.acc - 50).abs().argmax()]:.2f})'
    )
    ax1.legend(loc='upper right')
    
#     # color the background
#     ax1.set_facecolor('white')
#     ax.set_facecolor('white')
    ylim = ax1.get_ylim()
    ax1.axhspan(
        float(ax1.get_yticks()[0]), float(ax1.get_yticks()[-1]), xmax=1, 
        xmin=.5 + sorted_gaps[(df.acc - 50).abs().argmax()] / (ax.get_xlim()[1]*2),
        facecolor='teal', alpha=.1, zorder=-100,
    )
    ax1.axhspan(
        float(ax1.get_yticks()[0]), float(ax1.get_yticks()[-1]), xmin=0, 
        xmax=.5 + sorted_gaps[(df.acc - 50).abs().argmax()] / (ax.get_xlim()[1]*2),
        facecolor='orange', alpha=.1, zorder=-100,
    )
    ax1.set_ylim(ylim)
    
    # in the center of left and right half, add text what it means
    x1 = ax1.get_xlim()[0] / 2
    x2 = ax1.get_xlim()[1] / 2
    y = ax1.get_ylim()[0] + np.diff(ax1.get_ylim())/2
    ax1.text(x1, y, "Predict Pathological", ha='right')
    ax1.text(x2, y, "Predict Non-Pathological", ha='left')
    
    ax.set_title('Brain Age Gap Pathology Proxy')
    ax1.set_title('')
    
    # manually force last ytick to show (as multiple of 10). 78.75 did not create a 80 ticklabel, s.t.
    # labels in twin axis did not match
    ax.set_ylim((ax.get_ylim()[0], int(math.ceil(ax.get_ylim()[1] / 10.0)) * 10))
    # https://stackoverflow.com/questions/26752464/how-do-i-align-gridlines-for-two-y-axis-scales-using-matplotlib
    # order matters. do this after data was plotted to this axis
    ax1.set_yticks(np.linspace(ax1.get_yticks()[0], ax1.get_yticks()[-1], len(ax.get_yticks())))
    ax1.grid(None)
    return ax


# def plot_thresh_to_acc(
#     df,
#     ax=None,
#     dummy=None,
# ):
#     # if we only decode normals or abnormals, this will raise 
#     #if df.pathological.nunique() == 1:
#     #    warnings.filterwarnings("ignore", message="y_pred contains classes not")
#     sorted_gaps = (df['y_true'] - df['y_pred']).sort_values().to_numpy()
#     gaps = df['y_true'] - df['y_pred']

#     accs = []
#     for thresh in sorted_gaps:
#         y_true=df['pathological'].to_numpy(dtype=int)
#         y_pred=(gaps > thresh).to_numpy(dtype=int)
#         #logger.debug(f"second: y_pred {np.unique(y_pred)}, y_true {np.unique(y_true)}")
#         accs.append(
#             balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
#         )
#     df = pd.DataFrame([sorted_gaps, accs, y_true]).T
#     df.columns = ['thresh', 'acc', 'pathological']
#     df['acc'] *= 100

#     if ax is None:
#         fig, ax = plt.subplots(1,1,figsize=(12,3))
#     if 0 not in df.pathological.unique():
#         c='g' 
#     elif 1 not in df.pathological.unique():
#         c='b'
#     else:
#         c='k'

#     ax.plot(df['thresh'], df['acc'], c=c)#, zorder=3)  # does not make sense in mixed case
#     ax.fill_between(df['thresh'][df['thresh'] < sorted_gaps[df.acc.argmax()]],
#                     df['acc'][df['thresh'] < sorted_gaps[df.acc.argmax()]], 50, 
#                     color='b', label='False', alpha=.5)
#     ax.fill_between(df['thresh'][df['thresh'] > sorted_gaps[df.acc.argmax()]], 
#                     df['acc'][df['thresh'] > sorted_gaps[df.acc.argmax()]], 50,
#                     color='r', label='True', alpha=.5)

#     ax.set_ylabel('Accuracy [%]')
#     ax.set_xlabel('Chronological Age – Decoded Age [years]')
#     ax.legend(title='Pathological')

#     xlim = max(abs(sorted_gaps)) * 1.1
#     ax.set_xlim(-xlim, xlim)

#     ax.plot([sorted_gaps[df.acc.argmax()], sorted_gaps[df.acc.argmax()]], 
#             [ax.get_ylim()[0],  df.acc.max()], c='lightgreen', linewidth=1)
#     ax.plot([ax.get_xlim()[0], sorted_gaps[df.acc.argmax()]], 
#             [df.acc.max(), df.acc.max()], c='lightgreen', linewidth=1)
# #     ax.axvline(sorted_gaps[df.acc.argmax()], c='lightgreen', linewidth=1)
# #     ax.axhline(df.acc.max(), c='lightgreen', linewidth=1)
#     ax.scatter(sorted_gaps[df.acc.argmax()], df.acc.max(), zorder=4, marker='*', 
#                c='lightgreen', s=20)

#     ax.text(sorted_gaps[df.acc.argmax()], ax.get_ylim()[0]-1.75, f"{sorted_gaps[df.acc.argmax()]:.2f}",
#             ha='center', va='top', fontweight='bold')#, c='lightgreen')
#     ax.text(ax.get_xlim()[0]-1.25, df.acc.max(), f"{df.acc.max():.2f}",
#             ha='right', va='center', fontweight='bold')#, c='lightgreen')
    
#     ax.set_yticks(ax.get_yticks()[:-2])
    
#     if dummy is not None:
#         ax.axhline(dummy, c='m', linewidth=1)
#     return ax


def create_grid(hist_max_count, max_age):
    #https://stackoverflow.com/questions/10388462/matplotlib-different-size-subplots
    fig, ax = plt.subplots(1, 1, figsize=(18,18))
    gridx = 14
    gridy = 28
    ax0 = plt.subplot2grid((gridx, gridy), (0, 4), rowspan=2, colspan=10)
    ax1 = plt.subplot2grid((gridx, gridy), (0, 16), rowspan=2,  colspan=10)
    ax2 = plt.subplot2grid((gridx, gridy), (2, 0), rowspan=5, colspan=4)
    ax2.invert_xaxis()
    ax3 = plt.subplot2grid((gridx, gridy), (2, 4), rowspan=5, colspan=10)
    ax4 = plt.subplot2grid((gridx, gridy), (2, 16), rowspan=5, colspan=10)
    ax5 = plt.subplot2grid((gridx, gridy), (2, 14), rowspan=5, colspan=1)
    ax6 = plt.subplot2grid((gridx, gridy), (2, 26), rowspan=5, colspan=1)
    ax7 = plt.subplot2grid((gridx, gridy), (0, 1), rowspan=1, colspan=1)

    facecolor = 'white'
    ax0.set_title('')
    ax0.set_xlim(0, max_age)
    ax0.set_ylim([0, hist_max_count])
    ax0.set_xticklabels([])
    ax0.set_xlabel(' ')
    ax0.set_facecolor(facecolor)
    ax1.set_title('')
    ax1.set_xlim(0, max_age)
    ax1.set_ylim([0, hist_max_count])
    ax1.set_xticklabels([])
    ax1.set_xlabel(' ')
    ax1.set_yticklabels([])
    ax1.set_ylabel(' ')
    ax1.set_facecolor(facecolor)
    ax2.set_ylim(0, max_age)
    ax2.set_xlim([hist_max_count, 0])
    ax2.set_facecolor(facecolor)
    ax2.set_ylabel('Decoded Age [years]')
    ax3.set_ylim(0, max_age)
    ax3.set_yticklabels([])
    ax3.set_ylabel(' ')
    ax3.set_xlabel('Chronological Age [years]')
    ax4.set_ylim(0, max_age)
    ax4.set_yticklabels([])
    ax4.set_ylabel(' ')
    ax4.set_xlabel('Chronological Age [years]')
    ax7.set_facecolor(facecolor)
    ax7.set_xticks([])
    ax7.set_yticks([])
    return fig, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7


def plot_heatmap(H, df, bin_size, max_age, cmap, cbar_ax, vmax, ax=None):
    from matplotlib.colors import LinearSegmentedColormap
    # https://stackoverflow.com/questions/67605719/displaying-lowest-values-as-white
#     cmap_ = LinearSegmentedColormap.from_list('', ['white', *getattr(plt.cm, cmap)(np.arange(255))])
    # make discrete colorbar
    colors = ['white'] + [getattr(plt.cm, cmap)(i) for i in np.linspace(0, 255, vmax-1, dtype=int)]
    cmap_ = LinearSegmentedColormap.from_list('discrete_reds', colors, N=vmax)

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(7,6))

    ax = sns.heatmap(H, ax=ax, cmap=cmap_, vmin=0, vmax=vmax,
                     cbar_ax=cbar_ax, cbar_kws={'aspect': 50, 'fraction': 0.05})
    ax.invert_yaxis()

    # add cbar max as text
#     print([t.text for t in cbar_ax.get_yticklabels()])
#     cbar_ax.set_yticks(list(cbar_ax.get_yticks())[:-1] + [cbar_ax.get_ylim()[1]])
    # avoid floating point number labels in discrete colorbar
    cbar_ax.set_yticks(cbar_ax.get_yticks()[:-1])
    cbar_ax.set_yticklabels([
        '' if '.' in t._text and t._text.split('.')[1] != '0' else str(int(float(t._text)))
        for t in cbar_ax.get_yticklabels()
    ])
    cbar_ax.set_ylabel('Count')

    ax.scatter(
        df.y_true.mean()/bin_size, df.y_pred.mean()/bin_size, 
        marker='*', c='magenta' if cmap == 'Reds' else 'cyan',
        s=250, edgecolor='k', zorder=3)

    # TODO: double and triple check why y_true and y_pred x and y here need to be swapped
    m, b = np.polyfit(df.y_true.to_numpy('int')/bin_size, df.y_pred.to_numpy('float')/bin_size, 1)
    ax.plot(df.y_true/bin_size, m*df.y_true/bin_size + b, linewidth=1, #linestyle='--',
            c='magenta' if cmap == 'Reds' else 'cyan')

    # add error to trendline
    # does not really make sense
#     mae = mean_absolute_error(df.y_true, df.y_pred)
#     ax.plot(df.y_true/bin_size, m*df.y_true/bin_size + b + mae/bin_size, linewidth=.2, #linestyle=':',
#         c='magenta' if cmap == 'Reds' else 'cyan')
#     ax.plot(df.y_true/bin_size, m*df.y_true/bin_size + b - mae/bin_size, linewidth=.2, #linestyle=':',
#         c='magenta' if cmap == 'Reds' else 'cyan')

    # for every chronological age plot mean decoded age
#     ax.plot((df[['y_true', 'y_pred']]/bin_size).sort_values('y_true').groupby('y_true', as_index=False).mean().y_true, 
#             (df[['y_true', 'y_pred']]/bin_size).sort_values('y_true').groupby('y_true', as_index=False).mean().y_pred,
#             c='magenta' if cmap == 'Reds' else 'cyan', linewidth=.5)

#     ax.axvline(df.y_true.mean()/bin_size, linestyle='--', color='r' if cmap == 'Reds' else 'b')
#     ax.axhline(df.y_pred.mean()/bin_size, linestyle='--', color='r' if cmap == 'Reds' else 'b')

#     ticklabels = [t.get_text() for t in ax.get_xticklabels()]
#     ticklabels = [str(int(t)*bin_size) for t in ticklabels]
#     ax.set_xticklabels(ticklabels)
#     ax.set_yticklabels(ticklabels)
    ax.set_xlabel('Chronological Age [years]')
    ax.set_xticks([int(i/bin_size) for i in np.linspace(0, 100, 11)])
    ax.set_xticklabels([str(i) for i in np.linspace(0, 100, 11, dtype=int)], rotation=0)
    
    ax.set_ylabel(' ')
    ax.set_yticklabels([])
    ax.text(ax.get_xlim()[0], ax.get_ylim()[1], 'Overestimated', ha='left', va='top', weight='bold')
    ax.text(ax.get_xlim()[1], ax.get_ylim()[0], 'Underestimated', ha='right', va='bottom', weight='bold')
    return ax


def plot_heatmaps(df, bin_size, max_age, hist_max_count):
    assert max_age == 100
    assert max_age % bin_size == 0
    fig, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = create_grid(hist_max_count, max_age)
    
    df_p = df[df.pathological == 1]
    df_np = df[df.pathological == 0]
    import matplotlib.patches as mpatches
    patches = []
    if not df_np.empty:
        mae_non_patho = mean_absolute_error(df_np.y_true, df_np.y_pred)
        patches.append(mpatches.Patch(color='b', label=f'False (n={len(df_np)})\n({mae_non_patho:.2f} years mae)', alpha=.5))
    if not df_p.empty:
        mae_patho = mean_absolute_error(df_p.y_true, df_p.y_pred)
        patches.append(mpatches.Patch(color='r', label=f'True (n={len(df_p)})\n({mae_patho:.2f} years mae)', alpha=.5))
    ax7.legend(handles=patches, title='Pathological')

    bins = np.arange(0, 100, bin_size)
    sns.histplot(df_np.y_true, ax=ax0, color='b', kde=True, bins=bins)
    ax0.axvline(df_np.y_true.mean(), c='cyan')
    sns.histplot(df_p.y_true, ax=ax1, color='r', kde=True, bins=bins)
    ax1.axvline(df_p.y_true.mean(), c='magenta')

    sns.histplot(data=df_np, y='y_pred', ax=ax2, color='b', kde=True, bins=bins)
    sns.histplot(data=df_p, y='y_pred', ax=ax2, color='r', kde=True, bins=bins)
    ax2.axhline(df_np.y_pred.mean(), c='cyan')
    ax2.axhline(df_p.y_pred.mean(), c='magenta')
    ax2.set_xticks(ax0.get_yticks()[:-1])
    ax2.set_yticks(np.linspace(0, 100, 11, dtype='int'))
    ax2.legend()
    
    sns.lineplot(x=[0, 100], y=[0, 100], ax=ax3, c='k', linewidth=1)
    sns.scatterplot(data=df_np[['y_pred', 'y_true']].mean().to_frame().T, 
                    x='y_true', y='y_pred', ax=ax3, c='cyan', marker='*', s=300)
    sns.lineplot(x=[0, 100], y=[0, 100], ax=ax4, c='k', linewidth=1)
    sns.scatterplot(data=df_p[['y_pred', 'y_true']].mean().to_frame().T, 
                    x='y_true', y='y_pred', ax=ax4, c='magenta', marker='*', s=300)

    Hs = []
    dfs = [df_np, df_p]
    for this_df in dfs:
        if this_df.empty:
            Hs.append(None)
            continue
        # TODO: double and triple check why i need to swap y_true and y_pred x and y here
        H, xedges, yedges = np.histogram2d(
            this_df.y_pred, this_df.y_true, 
            bins=max_age//bin_size, range=[[0, max_age], [0, max_age]],
        )
        Hs.append(H)
    Hmax = np.max([H.max() for H in Hs if H is not None]).astype(int)

#     fig, ax_arr = plt.subplots(1, 2, figsize=(15,6), sharex=True, sharey=True)
#     fig.tight_layout()
    axs = [ax3, ax4]
    axs2 = [ax0, ax1]
    for i, (H, this_df, cmap, cbar_ax) in enumerate(zip(Hs, dfs, ['Blues', 'Reds'], [ax5, ax6])):
        if this_df.empty:
            continue
        ax = plot_heatmap(
            H,
            this_df,
            bin_size=bin_size,
            max_age=max_age,
            cmap=cmap,
            ax=axs[i],
            vmax=Hmax,
            cbar_ax=cbar_ax,
        )
        mae = mean_absolute_error(this_df.y_true, this_df.y_pred)
        # add error to diagonal
#         sns.lineplot(x=[0, 100-mae/bin_size], y=[mae/bin_size, 100], ax=axs[i], c='k', linewidth=1, linestyle='--')
#         sns.lineplot(x=[mae/bin_size, 100], y=[0, 100-mae/bin_size], ax=axs[i], c='k', linewidth=1, linestyle='--')
#         axs2[i].set_title(
#             f'Non-pathological\n({mae:.2f} years mae)' if i == 0 else f'Pathological\n({mae:.2f} years mae)')
    return fig


# TODO: merge hists functions
def plot_hist():
    pass


def plot_age_gap_hist(
    df,
    bin_width=5,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12,3))
    bins = np.concatenate([
        np.arange(0, - df.gap.min() + bin_width, bin_width, dtype=int)[::-1]*-1,
        np.arange(bin_width, df.gap.max() + bin_width, bin_width, dtype=int)
    ])
    patho_df = df[df.pathological == 1]
    non_patho_df = df[df.pathological == 0]
    ax = sns.histplot(data=non_patho_df, x='gap', color='b', ax=ax, kde=True, bins=bins, label='False')
    ax = sns.histplot(data=patho_df, x='gap', color='r', ax=ax, kde=True, bins=bins, label='True')
    mean_non_patho_gap = non_patho_df.gap.mean()
    mean_patho_gap = patho_df.gap.mean()
    ax.axvline(mean_non_patho_gap, c='cyan')
    ax.axvline(mean_patho_gap, c='magenta')
#     if mean_patho_gap > mean_non_patho_gap:
#         ax.text(mean_non_patho_gap + (mean_patho_gap - mean_non_patho_gap)/2, ax.get_ylim()[1], 
#                 f"{(mean_patho_gap - mean_non_patho_gap):.2f}", fontweight='bold',
#                 ha='center', va='bottom')
#     else:
#         ax.text(mean_patho_gap + (mean_non_patho_gap - mean_patho_gap)/2, ax.get_ylim()[1], 
#                 f"{(mean_non_patho_gap - mean_patho_gap):.2f}", fontweight='bold',
#                 ha='center', va='bottom')
    max_abs_gap = max(abs(df.gap))*1.1
    ax.set_xlim(-max_abs_gap, max_abs_gap)
    ax.set_xlabel('Chronological Age − Decoded Age [years]')
    ax.set_title(f'Brain age gap')
    ax.legend(title='Pathological')
    return ax


def accuracy_perumtations(df, n_repetitions):
    accs = []
    for thresh in df.gap:
        acc = balanced_accuracy_score(df.pathological, df.gap > thresh) * 100
        accs.append(acc)
    orig_acc = max(accs)
    
    accs = []
    for n in range(n_repetitions):
        choices = np.random.choice(2, len(df))
        acc = balanced_accuracy_score(df.pathological, choices)*100
        accs.append(acc)
    return orig_acc, accs


def age_gap_diff_permutations(df, n_repetitions, subject_wise):
    if subject_wise:
        df = df.groupby(['subject', 'pathological'], as_index=False).mean(numeric_only=True)
    gaps = df['gap']
    # averaging above changes dtype of pathological from bool to float....
    patho_df = df[df.pathological == 1]
    non_patho_df = df[df.pathological == 0]
    mean_patho_gap = patho_df['gap'].mean()
    mean_non_patho_gap = non_patho_df['gap'].mean()
    mean_gap_diff = mean_patho_gap - mean_non_patho_gap

    mean_gap_diffs = []
    for n in range(n_repetitions):
        choices = np.random.choice(2, len(gaps))
        gaps0 = gaps[choices==0]
        gaps1 = gaps[choices==1]
        mean_gap_diffs.append(gaps0.mean() - gaps1.mean())
    return mean_gap_diff, mean_gap_diffs


def plot_violin(y, sampled_y, xlabel, center_value=0):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.axvline(y, c='lightgreen')
    ax = sns.violinplot(x=sampled_y, kde=True, color='g', inner="quartile")
    # set violin alpha = .5
    # https://github.com/mwaskom/seaborn/issues/622
    from matplotlib.collections import PolyCollection
    for art in ax.get_children():
        if isinstance(art, PolyCollection):
            art.set_alpha(.5)
    ax.set_xlabel(xlabel)
    ax.legend(['Observed', 'Sampled'])#, title='Mean Chronological Age - Predicted Age')
    max_abs_lim = max([abs(center_value - i) for i in ax.get_xlim()])
    ax.set_xlim(center_value-max_abs_lim, center_value+max_abs_lim)
    
    p = np.min([
        ((sampled_y >= y).sum() / len(sampled_y)),
        ((sampled_y <= y).sum() / len(sampled_y)),
    ])
    if p == 0:
        p = 1/len(sampled_y)

    ax.text(y, ax.get_ylim()[1], f'{y:.2f} (p={p:.2E})',
            ha='center', va='bottom', fontweight='bold')
    return ax


def age_pyramid(df_of_ages_genders_and_pathology_status, train_or_eval, alpha=.5,
                fs=24, ylim=20, bins=np.linspace(0, 100, 101), out_dir=None,
                show_title=True, show_pathology_legend=True):
    df = df_of_ages_genders_and_pathology_status
    male_df = df[(df["gender"] == 0) | (df["gender"] == 'M')]
    female_df = df[(df["gender"] == 1) | (df["gender"] == 'F')]

    male_abnormal_df = male_df[male_df["pathological"] == 1]
    male_normal_df = male_df[male_df["pathological"] == 0]
    female_abnormal_df = female_df[female_df["pathological"] == 1]
    female_normal_df = female_df[female_df["pathological"] == 0]

    f, ax_arr = plt.subplots(ncols=2, sharey=True, sharex=False, figsize=(15, 18))
    ax1, ax2 = ax_arr

    if show_title:
        plt.suptitle(train_or_eval+" Histogram", y=.9, fontsize=fs+5)

    sns.histplot(
        y=male_normal_df["age"], bins=bins, alpha=alpha, color="g",
        orientation="horizontal", ax=ax1, kde=True,
        label="Non-pathological ({:.1f}%)".format(
            len(male_normal_df) / len(male_df) * 100) if show_pathology_legend else None,
    )
    ax1.axhline(male_normal_df["age"].mean(), color='g')

    sns.histplot(
        y=male_abnormal_df["age"], bins=bins, alpha=alpha, color="b",
        orientation="horizontal", ax=ax1, kde=True,
        label="Pathological ({:.1f}%)".format(
            len(male_abnormal_df) / len(male_df) * 100) if show_pathology_legend else None,
    )
    ax1.axhline(male_abnormal_df["age"].mean(), color='b')

    ax1.axhline(np.mean(male_df["age"]), color="black",
                # label="mean age {:.2f} $\pm$ {:.2f}".format(
                #     np.mean(male_df["age"]), np.std(male_df["age"])))
                label="Mean age {:.1f} ($\pm$ {:.1f})"
                .format(np.mean(male_df["age"]), np.std(male_df["age"])))
    ax1.barh(np.mean(male_df["age"]), height=2 * np.std(male_df["age"]),
             width=ylim, color="black", alpha=.25)
    ax1.set_xlim(0, ylim)

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [2, 1, 0]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    ax1.legend(fontsize=fs, loc="lower left")
    ax1.set_title("Male ({:.1f}%)".format(100 * float(len(male_df) / len(df))),
                  fontsize=fs, loc="left", y=.95, x=.05)
    ax1.invert_xaxis()

    # second axis
    sns.histplot(
        y=female_normal_df["age"], bins=bins, alpha=alpha, color="y",
        orientation="horizontal", ax=ax2, kde=True, line_kws= {'linestyle': '--'}, #kde_kws={'bw_adjust': 3},
        label="Non-pathological ({:.1f}%)".format(
            len(female_normal_df) / len(female_df) * 100) if show_pathology_legend else None,
    )
    ax2.axhline(female_normal_df["age"].mean(), color='y', linestyle="--")

    sns.histplot(
        y=female_abnormal_df["age"], bins=bins, alpha=alpha, color="r",
        orientation="horizontal", ax=ax2, kde=True, line_kws= {'linestyle': '--'}, #kde_kws={'bw_adjust': 3},
        label="Pathological ({:.1f}%)".format(
            len(female_abnormal_df) / len(female_df) * 100) if show_pathology_legend else None,
    )
    ax2.axhline(female_abnormal_df["age"].mean(), color='r', linestyle="--")

    ax2.axhline(np.mean(female_df["age"]), color="black", linestyle="--",
                # label="mean age {:.2f} $\pm$ {:.2f}"
                # .format(np.mean(female_df["age"]), np.std(female_df["age"])))
                label="Mean age {:.1f} ($\pm$ {:.1f})"
                .format(np.mean(female_df["age"]), np.std(female_df["age"])))
    ax2.barh(np.mean(female_df["age"]), height=2 * np.std(female_df["age"]),
             width=ylim, color="black",
             alpha=.25)
    ax2.legend(fontsize=fs, loc="lower right")
    ax2.set_xlim(0, ylim)
    # ax1.invert_yaxis()
    ax2.set_title("Female ({:.1f}%)".format(100 * len(female_df) / len(df)),
                  fontsize=fs, loc="right", y=.95, x=.95)  # , y=.005)

    plt.ylim(0, 100)
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.set_ylabel("Age [years]", fontsize=fs)
    ax1.set_xlabel("Count", fontsize=fs, x=1)
    # ax1.yaxis.set_label_coords(-.025, 0)
    plt.yticks(np.linspace(0, 100, 11), fontsize=fs - 5)
    ax1.tick_params(labelsize=fs - 5)
    ax2.tick_params(labelsize=fs - 5)
    ax2.set_xlabel("")
    if out_dir is not None:
        plt.savefig(out_dir+"tuh_{}.png".format(train_or_eval),
                    bbox_inches="tight")
    return ax_arr


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


def read_result(
    exps_dir,
    result,
):
    exp_dirs = os.listdir(exps_dir)
    exp_results = []
    for exp_dir in exp_dirs:
        exp_dir = os.path.join(exps_dir, exp_dir)
        folds = sorted(os.listdir(exp_dir))
        cv_result = []
        for valid_set_i in folds:
            this_exp_dir = os.path.join(exp_dir, valid_set_i)
            this_result = _read_result(this_exp_dir, result)
            cv_result.append(this_result)
        cv_result = pd.concat(cv_result, ignore_index=int(result == 'config'))
        exp_results.append(cv_result)
    exp_results = pd.concat(exp_results).sort_values(['seed', 'valid_set_i'])
    return exp_results


def _read_result(
    exp_dir,
    result,
):
    config_path = os.path.join(exp_dir, 'config.csv')
    config = pd.read_csv(config_path, index_col=0).T
    if result == 'history':
        hist_path = os.path.join(exp_dir, 'history.csv')
        this_result = pd.read_csv(hist_path, index_col=0)
    elif result == 'score':
        score_path = os.path.join(exp_dir, 'train_end_scores.csv')
        this_result = pd.read_csv(score_path, index_col=0)
    elif result == 'preds':
        try:
            subset = 'valid'
            pred_path = os.path.join(exp_dir, 'preds', f'train_end_{subset}_preds.csv')
            preds1 = pd.read_csv(pred_path, index_col=0)
        except:
            subset = 'eval'
            pred_path = os.path.join(exp_dir, 'preds', f'train_end_{subset}_preds.csv')
            preds1 = pd.read_csv(pred_path, index_col=0)
        preds1['subset'] = subset
        try:
            pred_path = os.path.join(exp_dir, 'preds', f'train_end_valid_not_{config.squeeze()["subset"]}_preds.csv')
            preds2 = pd.read_csv(pred_path, index_col=0)
        except:
            try:
                pred_path = os.path.join(exp_dir, 'preds', f'train_end_valid_rest_preds.csv')
                preds2 = pd.read_csv(pred_path, index_col=0)
            except:
                raise FileNotFoundError
        preds2['subset'] = 'valid_rest'
        # TODO: add longitudinal preds
        this_result = pd.concat([preds1, preds2])
        this_result['gap'] = this_result.y_true - this_result.y_pred
    elif result == 'config':
        this_result = config
    else:
        raise ValueError
    if result != 'config':
        this_result['seed'] = int(config['seed'])
        this_result['valid_set_i'] = int(config['valid_set_i'])
    return this_result


# TODO: add std error bar around mean?
def plot_recording_interval_hist(df, clip_value, c, ax=None):
    all_day_diffs = []
    for subj, g in df.groupby('subject'):
        diff = g[['year', 'month', 'day']].sort_values(['year', 'month', 'day']).diff()
        day_diff = diff['year'] * 365 + diff['month'] * 30 + diff['day']
        day_diff = day_diff.iloc[1:]
        assert np.isfinite(day_diff).all()
        all_day_diffs.append(day_diff)
    day_diffs = pd.DataFrame(pd.concat(all_day_diffs, axis=0)).astype(int)

    bin_width = 30
    bins = np.arange(0, 4748, bin_width)
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(12,3))
    mean = day_diffs.mean().squeeze()
    std = day_diffs.std().squeeze()
    ax.axvline(mean, c=c, label='')
    day_diffs[day_diffs>clip_value] = clip_value
    ax = sns.histplot(day_diffs, ax=ax, bins=bins, kde=True, palette=[c])
    ax.set_yscale('log')
    ax.set_xticks([0, 30, 90, 180, 365] + [365*i for i in range(2, 14)]);
    ax.set_xticklabels([str(i) for i in [0, 30, 90, 180, 365]] + [f'365*{i}' for i in range(2, 14)])
    ax.set_xlim((0-bin_width, clip_value+bin_width))
    ax.set_xlabel('Days')
    print(df.pathological.unique(), mean, std)
    return ax, mean, std


def plot_longitudinal_interval_hists(description):
    fig, ax = plt.subplots(1,1,figsize=(12,3))
    df_pathological = extract_longitudinal_dataset(description, kind='pathological', load=False)
    ax, mean_patho, std_patho = plot_recording_interval_hist(df_pathological, 365*5, ax=ax, c='r')
    df_non_pathological = extract_longitudinal_dataset(description, kind='non_pathological', load=False)
    ax, mean_non_patho, std_non_patho = plot_recording_interval_hist(df_non_pathological, 365*5, ax=ax, c='b')
    df_transition = extract_longitudinal_dataset(description, kind='transition', load=False)
    ax, mean_transition, std_transition = plot_recording_interval_hist(df_transition, 365*5, 'g', ax=ax)
    ax.legend([
        f'Pathological ({mean_patho:.0f} $\pm$ {std_patho:.0f} days)',
        f'Non-Pathological ({mean_non_patho:.0f} $\pm$ {std_non_patho:.0f} days)',
        f'Transition ({mean_transition:.0f} $\pm$ {std_transition:.0f} days)',
    ])
    ax.get_legend().legendHandles[0].set_color('r')
    ax.get_legend().legendHandles[1].set_color('b')
    ax.get_legend().legendHandles[2].set_color('g')
    ax.set_title('Recording Intervals in TUH Longitudinal Datasets')
    return ax


def extract_longitudinal_dataset(description, kind, load):
    assert kind in ['transition', 'pathological', 'non_pathological']
    dfs = []
    for s, g in description.groupby('subject'):
        if len(g) > 1:
            if g.pathological.nunique() == 2:
                if kind == 'transition':
                        dfs.append(g)
            else:
                if kind == 'pathological':
                    if g.pathological.unique() == 1:
                        dfs.append(g)
                elif kind == 'non_pathological':
                    if g.pathological.unique() == 0:
                        dfs.append(g)
    dfs = pd.concat(dfs)
    dfs = dfs.reset_index(drop=True)
    if kind == 'transition':
        assert all(dfs.groupby('subject').pathological.nunique() == 2)
    else:
        if kind in ['pathological', 'non_pathological']:
            assert all(dfs.groupby('subject').pathological.nunique() == 1)
    print("n recs", len(dfs), "n subj", dfs.subject.nunique())
    if not load:
        return dfs
    else:
        dfs = dfs.T
        ds = []
        for i, s in dfs.iteritems():
            p = s.path
            p = p.replace('/data/datasets/TUH/EEG/tuh_eeg/', '/home/jovyan/mne_data/TUH_PRE/tuh_eeg/')
            if not os.path.exists(p):
                raise RuntimeError("rec not found")
            raw = mne.io.read_raw_edf(p, preload=False, verbose='error')
            d = BaseDataset(raw, s, target_name='age')
            ds.append(d)
        ds = BaseConcatDataset(ds)
        return ds


if __name__ == "__main__":
    # TODO: add exp to arguments?
    parser = argparse.ArgumentParser()
    # args for decoding
    parser.add_argument('--augment', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--condition', type=str)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--date', type=str)
    parser.add_argument('--debug', type=int)
    parser.add_argument('--fast-mode', type=int)
    parser.add_argument('--final-eval', type=int)
    parser.add_argument('--intuitive-training-scores', type=int)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--max-age', type=int)
    parser.add_argument('--min-age', type=int)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--n-jobs', type=int)
    parser.add_argument('--n-restarts', type=int)
    parser.add_argument('--n-train-recordings', type=int)
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--preload', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--shuffle-data-before-split', type=int)
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
