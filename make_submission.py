#!/usr/bin/env python3
import argparse
from collections import defaultdict
from functools import partial
import multiprocessing.pool
from pathlib import Path
import pickle
from typing import Tuple, List, Dict

import eli5
from eli5.formatters.text import format_as_text
import numpy as np
import pandas as pd
from skimage.feature import blob_log
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.externals import joblib
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn import metrics
import tqdm
from xgboost import XGBRegressor

import utils


STEP_RATIO = 2
PRED_SCALE = 4
PATCH_SIZE = 60
SUM_THRESHOLDS = [0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.5]
BLOB_THRESHOLDS = [0.02, 0.04, 0.08, 0.16, 0.24, 0.5]
SUM_FEATURES = ['sum'] + ['sum-{:.2}'.format(th) for th in SUM_THRESHOLDS]
BLOB_FEATURES = [f.format(th) for th in BLOB_THRESHOLDS
                 for f in ['blob-{:.2f}', 'blob-{:.2f}-sum']]
FEATURE_NAMES = SUM_FEATURES + BLOB_FEATURES
ALL_FEATURE_NAMES = ['x0', 'x1', 'y0', 'y1'] + FEATURE_NAMES
FEATURE_NAMES = ['sum', 'sum-0.04', 'sum-0.08', 'sum-0.24']
FEATURE_NAMES += ['blob-0.04', 'blob-0.04-sum', 'blob-0.08', 'blob-0.08-sum']


def load_xs_ys(pred_path: Path, coords: pd.DataFrame,
               ) -> Tuple[int, float, np.ndarray, np.ndarray, List, List]:
    pred = utils.load_pred(pred_path)
    path_parts = pred_path.name.split('-')[:-1]
    if len(path_parts) == 2:
        img_id, img_scale = path_parts
    else:
        img_id, = path_parts
        img_scale = 1
    img_id, img_scale = int(img_id), float(img_scale)
    scale = PRED_SCALE / img_scale
    all_features, all_targets = [], []
    all_cls_blobs, all_blob_ids = [], []
    for cls in range(utils.N_CLASSES):
        cls_features, cls_targets = [], []
        all_features.append(cls_features)
        all_targets.append(cls_targets)
        cls_pred = pred[cls]
        try:
            img_coords = coords.loc[[img_id]]
            cls_coords = img_coords[img_coords.cls == cls]
            cls_coords = list(zip(cls_coords.col / scale,
                                  cls_coords.row / scale))
        except KeyError:
            cls_coords = []
        cls_blobs = [[(x, y, cls_pred[int(np.round(y)), int(np.round(x))])
                      for y, x, _ in blob_log(cls_pred, threshold=blob_threshold,
                                              min_sigma=1, max_sigma=4, num_sigma=4)]
                     for blob_threshold in BLOB_THRESHOLDS]
        all_cls_blobs.append(cls_blobs)
        cls_blob_ids = []
        all_blob_ids.append(cls_blob_ids)
        max_y, max_x = cls_pred.shape
        step = PATCH_SIZE // STEP_RATIO
        steps = lambda m: range(-PATCH_SIZE + step, m + PATCH_SIZE - step, step)
        for x0 in steps(max_x):
            for y0 in steps(max_y):
                x1 = min(max_x, x0 + PATCH_SIZE)
                y1 = min(max_y, y0 + PATCH_SIZE)
                x0 = max(0, x0)
                y0 = max(0, y0)
                patch = cls_pred[y0: y1, x0: x1]
                features = [x0, x1, y0, y1, patch.sum()]
                cls_features.append(features)
                for i, threshold in enumerate(SUM_THRESHOLDS):
                    bin_mask = patch > threshold
                    features.append(bin_mask.sum())
                target = sum(x0 <= x < x1 and y0 <= y < y1 for x, y in cls_coords)
                cls_targets.append(target)
                blob_ids = []
                for i, th_blobs in enumerate(cls_blobs):
                    blob_ids.append(i)
                    blob_count = blob_sum = 0
                    for x, y, value in th_blobs:
                        if x0 <= x < x1 and y0 <= y < y1:
                            blob_count += 1
                            blob_sum += value
                    features.extend([blob_count, blob_sum])
                cls_blob_ids.append(blob_ids)
    return (img_id, img_scale, np.array(all_features), np.array(all_targets),
            all_cls_blobs, all_blob_ids)


def load_all_features(root: Path, only_valid: bool, args) -> Dict[str, np.ndarray]:
    features_path = root.joinpath('features.npz')  # type: Path
    coords = utils.load_coords()
    pred_paths = list(root.glob('*-pred.npy'))
    get_id = lambda p: int(p.name.split('-')[0])
    if only_valid:
        valid_ids = {int(p.stem) for p in utils.labeled_paths()}
        pred_paths = [p for p in pred_paths if get_id(p) in valid_ids]
    if args.limit:
        pred_paths = pred_paths[:args.limit]
    if not args.new_features and features_path.exists():
        print('Loading features...')
        data = dict(np.load(str(features_path)))
        clf_features_path = root.joinpath('clf_features.npz')
        if clf_features_path.exists():
            clf_features = np.load(str(clf_features_path))['xs']
            data['xs'] = np.concatenate([data['xs'], clf_features], axis=2)
            for i in range(clf_features.shape[2]):
                feature_name = 'clf-{}'.format(i)
                ALL_FEATURE_NAMES.append(feature_name)
                FEATURE_NAMES.append(feature_name)
        print('done.')
        ids = [get_id(p) for p in pred_paths]
        assert set(ids) == set(data['ids'][0])
        return data
    print('{} total'.format(len(pred_paths)))
    data = {k: [[] for _ in range(utils.N_CLASSES)]
            for k in ['ids', 'scales', 'xs', 'ys']}
    blob_data = {k: [[] for _ in range(utils.N_CLASSES)]
                 for k in ['blobs', 'blob_ids']}
    with multiprocessing.pool.Pool(processes=24) as pool:
        for id, scale, xs, ys, blobs, blob_ids in tqdm.tqdm(
                pool.imap(partial(load_xs_ys, coords=coords), pred_paths, chunksize=2),
                total=len(pred_paths)):
            for cls in range(utils.N_CLASSES):
                data['ids'][cls].extend([id] * len(ys[cls]))
                data['scales'][cls].extend([scale] * len(ys[cls]))
                data['xs'][cls].extend(xs[cls])
                data['ys'][cls].extend(ys[cls])
                blob_data['blobs'][cls].append((id, scale, blobs[cls]))
                blob_data['blob_ids'][cls].extend(blob_ids[cls])
    data = {k: np.array(v, dtype=np.int32 if k in {'ids', 'ys'} else np.float32)
            for k, v in data.items()}
    with features_path.open('wb') as f:
        np.savez(f, **data)
    with root.joinpath('blobs.pkl').open('wb') as f:
        pickle.dump(blob_data, f)
    return data


def get_pred_by_id(ids: np.ndarray, pred: np.ndarray, unique_ids) -> np.ndarray:
    pred_by_id = np.zeros(len(unique_ids))
    id_idx = {img_id: i for i, img_id in enumerate(unique_ids)}
    for img_id, x in zip(ids, pred):
        pred_by_id[id_idx[img_id]] += x
    return np.divide(pred_by_id, STEP_RATIO**2)


def train(data, *regs,
          save_to=None, concat_features=False, explain=False):
    coords = utils.load_coords()
    concated_xs = np.concatenate(data['xs'], axis=1)
    all_rmse, all_patch_rmse, all_baselines = [], [], []
    regs_name = ', '.join(type(reg).__name__ for reg in regs)
    fitted_regs = []
    expl_by_cls = defaultdict(list)
    for cls in range(utils.N_CLASSES):
        ids = data['ids'][cls]
        scales = data['scales'][cls]
        ys = data['ys'][cls]
        xs = input_features(concated_xs if concat_features else data['xs'][cls])
        # indices = np.array(sorted(range(len(ids)), key=lambda i: (scales[i], ids[i])))
        # ids, xs, ys = ids[indices], xs[indices], ys[indices]
        pred, fitted = train_predict(regs, xs, ys, ids)
        ys_by_id, pred_by_id = [], []
        unique_ids = sorted(set(ids))
        pred_by_id = get_pred_by_id(ids, pred, unique_ids)
        for img_id in unique_ids:
            try:
                ys_by_id.append((coords.loc[[img_id]].cls == cls).sum())
            except KeyError:
                ys_by_id.append(0)
        pred_by_id = round_prediction(pred_by_id)
        patch_rmse = np.sqrt(metrics.mean_squared_error(ys, pred))
        rmse = np.sqrt(metrics.mean_squared_error(ys_by_id, pred_by_id))
        baseline_rmse = np.sqrt(metrics.mean_squared_error(
            cross_val_predict(DummyRegressor(), [[0]] * len(ys_by_id), ys_by_id, cv=5),
            ys_by_id))
        print('cls {}, patch mean {:.3f}, patch RMSE {:.3f}, '
              'image mean {:.2f}, image RMSE {:.2f}, baseline RMSE {:.2f}'
              .format(cls, np.mean(ys), patch_rmse,
                      np.mean(ys_by_id), rmse, baseline_rmse))
        all_rmse.append(rmse)
        all_patch_rmse.append(patch_rmse)
        all_baselines.append(baseline_rmse)
        if save_to:
            fitted_regs.append(fitted)
        if explain:
            for reg in fitted:
                expl = eli5.explain_weights(reg, feature_names=FEATURE_NAMES)
                expl_by_cls[cls].append(expl)
                print(type(reg).__name__, format_as_text(
                    expl, show=('method', 'targets', 'feature_importances')))
    print('{} with {} features: mean patch RMSE {:.3f}, mean image RMSE {:.2f}, '
          'mean baseline RMSE {:.2f}'
          .format(regs_name, ', '.join(FEATURE_NAMES),
                  np.mean(all_patch_rmse), np.mean(all_rmse),
                  np.mean(all_baselines)))
    if save_to:
        joblib.dump(fitted_regs, save_to)
        print('Saved to', save_to)

    if explain:
        dfs = []
        for cls, expls in expl_by_cls.items():
            for expl in expls:
                df = eli5.format_as_dataframe(expl)
                df['cls'] = cls
                df['estimator'] = expl.estimator.split('(')[0]
                dfs.append(df)
        df = pd.concat(dfs)
        df.reset_index(inplace=True)
        df['feature'] = df['index']
        del df['index']
        df = df[['feature', 'cls', 'estimator', 'std', 'weight']]
        df.to_csv('feature_importances.csv', index=None)


def train_predict(regs, xs, ys, ids):
    cv = GroupKFold(n_splits=4)
    fitted = []
    cv_preds = []
    pred_ids = []
    for train_ids, valid_ids in cv.split(xs, groups=ids):
        reg_preds = []
        for reg in regs:
            reg = clone(reg)
            if isinstance(reg, XGBRegressor):
                reg.fit(xs[train_ids], ys[train_ids],
                        eval_metric='rmse',
                        eval_set=[(xs[valid_ids], ys[valid_ids])],
                        verbose=False,
                        early_stopping_rounds=3)
            else:
                reg.fit(xs[train_ids], ys[train_ids])
            fitted.append(reg)
            reg_preds.append(reg.predict(xs[valid_ids]))
        cv_preds.append(average_predictions(reg_preds))
        pred_ids.append(valid_ids)
    pred_ids = np.array([
        idx for idx, v in sorted(enumerate(np.concatenate(pred_ids)),
                                 key=lambda x: x[1])])
    return np.concatenate(cv_preds)[pred_ids], fitted


def average_predictions(preds: List[np.ndarray]) -> np.ndarray:
    pred = np.mean(preds, axis=0)
    return np.clip(pred, 0, None)


def round_prediction(pred: np.ndarray) -> np.ndarray:
    return pred.round().astype(np.int32)


def input_features(xs):
    xs = xs[:, tuple(ALL_FEATURE_NAMES.index(f) for f in FEATURE_NAMES)]
    assert xs.shape[1] == len(FEATURE_NAMES)
    return xs


def predict(root: Path, model_path: Path, all_ids, all_xs, concat_features=False):
    all_regs = joblib.load(model_path)
    if concat_features:
        concated_xs = np.concatenate(all_xs, axis=1)
    classes = [
        'adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']
    unique_ids = sorted(set(all_ids[0]))
    all_preds = pd.DataFrame(index=unique_ids, columns=classes)
    for cls, (cls_name, cls_regs) in tqdm.tqdm(list(enumerate(zip(classes, all_regs)))):
        ids = all_ids[cls]
        xs = input_features(concated_xs if concat_features else all_xs[cls])
        pred = average_predictions([reg.predict(xs) for reg in tqdm.tqdm(cls_regs)])
        all_preds[cls_name] = round_prediction(get_pred_by_id(ids, pred, unique_ids))
    out_path = root.joinpath(root.name + '.csv')
    all_preds.to_csv(str(out_path), index_label='test_id')
    print('Saved submission to {}'.format(out_path))


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('root', type=Path)
    arg('mode', choices=['train', 'predict'])
    arg('--concat-features', action='store_true')
    arg('--predict-train', action='store_true')
    arg('--limit', type=int)
    arg('--new-features', action='store_true')
    arg('--explain', action='store_true')
    arg('--test-root', type=Path)
    args = parser.parse_args()
    model_path = args.root.joinpath('regressor.joblib')  # type: Path
    if args.mode == 'train':
        data = load_all_features(args.root, only_valid=True, args=args)
        train(data,
              ExtraTreesRegressor(
                  n_estimators=300, max_depth=5, min_samples_split=10, n_jobs=8,
                  criterion='mse'),
              ExtraTreesRegressor(
                  n_estimators=200, max_depth=4, min_samples_split=10, n_jobs=8,
                  criterion = 'mse'),
              XGBRegressor(n_estimators=200, max_depth=4, nthread=16),
             #Lasso(alpha=1.0, normalize=False, max_iter=100000),
              save_to=model_path,
              concat_features=args.concat_features,
              explain=args.explain,
              )
    elif args.mode == 'predict':
        if args.predict_train:
            data = load_all_features(args.root, only_valid=True, args=args)
        else:
            test_root = args.test_root or args.root.joinpath('test')
            data = load_all_features(test_root, only_valid=False, args=args)
        predict(args.root, model_path, all_ids=data['ids'], all_xs=data['xs'],
                concat_features=args.concat_features)


if __name__ == '__main__':
    main()
