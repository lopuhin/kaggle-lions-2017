#!/usr/bin/env python
import argparse
from functools import partial
import multiprocessing.pool
from pathlib import Path
from typing import Tuple, List

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
FEATURE_NAMES = ['sum', 'sum-0.02', 'sum-0.25', 'blob-0.02', 'blob-0.04']


def load_xs_ys(pred_path: Path, coords: pd.DataFrame,
               thresholds=(0.02, 0.25), patch_size=80,
               ) -> Tuple[int, np.ndarray, np.ndarray]:
    pred = utils.load_pred(pred_path)
    img_id = int(pred_path.name.split('-')[0])
    all_features, all_targets = [], []
    for cls in range(utils.N_CLASSES):
        cls_features, cls_targets = [], []
        all_features.append(cls_features)
        all_targets.append(cls_targets)
        cls_pred = pred[cls]
        try:
            img_coords = coords.loc[[img_id]]
            cls_coords = img_coords[img_coords.cls == cls]
            cls_coords = list(zip(cls_coords.col / PRED_SCALE,
                                  cls_coords.row / PRED_SCALE))
        except KeyError:
            cls_coords = []
        cls_blobs = [[(x, y)
                      for y, x, _ in blob_log(cls_pred, threshold=blob_threshold,
                                              min_sigma=1, max_sigma=4, num_sigma=4)]
                     for blob_threshold in [0.02, 0.04]]
        max_y, max_x = cls_pred.shape
        step = patch_size // STEP_RATIO
        steps = lambda m: range(-patch_size + step, m + patch_size - step, step)
        for x0 in steps(max_x):
            for y0 in steps(max_y):
                x1 = min(max_x, x0 + patch_size)
                y1 = min(max_y, y0 + patch_size)
                x0 = max(0, x0)
                y0 = max(0, y0)
                patch = cls_pred[y0: y1, x0: x1]
                features = [x0, x1, y0, y1]
                features.append(patch.sum())
                cls_features.append(features)
                for i, threshold in enumerate(thresholds):
                    bin_mask = patch > threshold
                    if i + 1 < len(thresholds):
                        bin_mask &= patch < thresholds[i + 1]
                    features.append(bin_mask.sum())
                # TODO - lookup
                target = sum(x0 <= x < x1 and y0 <= y < y1 for x, y in cls_coords)
                cls_targets.append(target)
                for th_blobs in cls_blobs:
                    # TODO - lookup
                    features.append(
                        sum(x0 <= x < x1 and y0 <= y < y1 for x, y in th_blobs))
    return img_id, np.array(all_features), np.array(all_targets)


def load_all_features(root: Path, only_valid: bool, args,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        data = np.load(str(features_path))
        ids = [get_id(p) for p in pred_paths]
        assert set(ids) == set(data['ids'][0])
        return data['ids'], data['xs'], data['ys']
    print('{} total'.format(len(pred_paths)))
    all_xs, all_ys, all_ids = [[[] for _ in range(utils.N_CLASSES)] for _ in range(3)]
    with multiprocessing.pool.Pool(processes=16) as pool:
        for id, xs, ys in tqdm.tqdm(
                pool.imap(partial(load_xs_ys, coords=coords), pred_paths, chunksize=2),
                total=len(pred_paths)):
            for cls in range(utils.N_CLASSES):
                all_xs[cls].extend(xs[cls])
                all_ys[cls].extend(ys[cls])
                all_ids[cls].extend([id] * len(ys[cls]))
    ids, xs, ys = [np.array(lst) for lst in [all_ids, all_xs, all_ys]]
    with features_path.open('wb') as f:
        np.savez(f, ids=ids, xs=xs, ys=ys)
    return ids, xs, ys


def train(all_ids, all_xs, all_ys, *regs,
          save_to=None, concat_features=False, explain=False):
    coords = utils.load_coords()
    concated_xs = np.concatenate(all_xs, axis=1)
    all_rmse, all_patch_rmse, all_baselines = [], [], []
    regs_name = ', '.join(type(reg).__name__ for reg in regs)
    fitted_regs = []
    for cls in range(utils.N_CLASSES):
        ids = all_ids[cls]
        ys = all_ys[cls]
        xs = input_features(concated_xs if concat_features else all_xs[cls])
        cv = GroupKFold(n_splits=5)
        pred = average_predictions(
            [cross_val_predict(reg, xs, ys, cv=cv, groups=ids) for reg in regs])
        ys_by_id, pred_by_id = [], []
        unique_ids = sorted(set(ids))
        for img_id in unique_ids:
            idx = ids == img_id
            try:
                ys_by_id.append((coords.loc[[img_id]].cls == cls).sum())
            except KeyError:
                ys_by_id.append(0)
            pred_by_id.append(pred[idx].sum() / STEP_RATIO**2)
        pred_by_id = round_prediction(np.array(pred_by_id))
        patch_rmse = np.sqrt(metrics.mean_squared_error(ys, round_prediction(pred)))
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
            fitted = []
            for reg in regs:
                reg = clone(reg)
                reg.fit(xs, ys)
                fitted.append(reg)
                if explain:
                    print(type(reg).__name__, format_as_text(
                        eli5.explain_weights(reg, feature_names=FEATURE_NAMES),
                        show=('method', 'targets', 'feature_importances')))
            fitted_regs.append(fitted)
    print('{}: mean patch RMSE {:.3f}, mean image RMSE {:.2f}, '
          'mean baseline RMSE {:.2f}'
          .format(regs_name, np.mean(all_patch_rmse), np.mean(all_rmse),
                  np.mean(all_baselines)))
    if save_to:
        joblib.dump(fitted_regs, save_to)
        print('Saved to', save_to)


def average_predictions(preds: List[np.ndarray]) -> np.ndarray:
    pred = np.mean(preds, axis=0)
    return np.clip(pred, 0, None)


def round_prediction(pred: np.ndarray) -> np.ndarray:
    return pred.round().astype(np.int32)


def input_features(xs):
    xs = xs[:, 4:]  # strip coords
    assert xs.shape[1] == len(FEATURE_NAMES)
    return xs


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
    args = parser.parse_args()
    model_path = args.root.joinpath('regressor.joblib')  # type: Path
    if args.mode == 'train':
        ids, xs, ys = load_all_features(args.root, only_valid=True, args=args)
        train(ids, xs, ys,
              ExtraTreesRegressor(
                  n_estimators=100, max_depth=3, min_samples_split=10, n_jobs=8,
                  criterion='mse'),
              XGBRegressor(n_estimators=100, max_depth=3, nthread=16),
             #Lasso(alpha=1.0, normalize=False, max_iter=100000),
              save_to=model_path,
              concat_features=args.concat_features,
              explain=args.explain,
              )
    elif args.mode == 'predict':
        if args.predict_train:
            ids, all_xs, _ = load_all_features(args.root, only_valid=True, args=args)
        else:
            ids, all_xs, _ = load_all_features(
                args.root.joinpath('test'), only_valid=False, args=args)
        all_regs = joblib.load(model_path)
        concated_xs = np.concatenate(all_xs, axis=1)
        classes = [
            'adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']
        all_preds = pd.DataFrame(index=ids, columns=classes)
        for cls, (cls_name, cls_regs) in enumerate(zip(classes, all_regs)):
            xs = input_features(concated_xs if args.concat_features else all_xs[cls])
            preds = [reg.predict(xs) for reg in cls_regs]
            all_preds[cls_name] = average_predictions(preds)
        out_path = args.root.joinpath(args.root.name + '.csv')
        all_preds.to_csv(str(out_path), index_label='test_id')
        print('Saved submission to {}'.format(out_path))


if __name__ == '__main__':
    main()
