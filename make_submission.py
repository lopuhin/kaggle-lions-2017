#!/usr/bin/env python
import argparse
from functools import partial
import multiprocessing.pool
from pathlib import Path
from typing import Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.externals import joblib
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import tqdm
# xgboost <= 0.6a2 shows a warning when used with scikit-learn 0.18+
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor

import utils


XsYs = Tuple[np.ndarray, np.ndarray]


def load_xs_ys(pred_path: Path, coords: pd.DataFrame, thresholds=(0.02, 0.25),
               ) -> XsYs:
    pred = utils.load_pred(pred_path)
    img_id = int(pred_path.name.split('-')[0])
    xs, ys = [], []
    for cls in range(utils.N_CLASSES):
        try:
            ys.append((coords.loc[img_id].cls == cls).sum())
        except KeyError:
            ys.append(0)
        cls_pred = pred[cls]
        x = [cls_pred.sum()]
        for i, threshold in enumerate(thresholds):
            bin_mask = cls_pred > threshold
            if i + 1 < len(thresholds):
                bin_mask &= cls_pred < thresholds[i + 1]
            x.append(bin_mask.sum())
        xs.append(x)
    return np.array(xs), np.array(ys)


def load_all_features(root: Path, only_vald: bool) -> XsYs:
    coords = utils.load_coords()
    pred_paths = list(root.glob('*-pred.npy'))
    if only_vald:
        valid_ids = {int(p.stem) for p in utils.labeled_paths()}
        pred_paths = [p for p in pred_paths
                      if int(p.name.split('-')[0]) in valid_ids]
    print('{} total'.format(len(pred_paths)))
    all_xs = [[] for _ in range(utils.N_CLASSES)]
    all_ys = [[] for _ in range(utils.N_CLASSES)]
    with multiprocessing.pool.ThreadPool(processes=16) as pool:
        for xs, ys in tqdm.tqdm(
                pool.imap_unordered(partial(load_xs_ys, coords=coords),
                                    pred_paths, chunksize=2),
                total=len(pred_paths)):
            for cls in range(utils.N_CLASSES):
                all_xs[cls].append(xs[cls])
                all_ys[cls].append(ys[cls])
    return np.array(all_xs), np.array(all_ys)


def evaluate_stacked(all_xs, all_ys, *regs, save_to=None):
    print()
    stacked_xs = np.concatenate(all_xs, axis=1)
    all_rmse = []
    regs_name = ', '.join(type(reg).__name__ for reg in regs)
    fitted_regs = []
    for cls in range(utils.N_CLASSES):
        ys = all_ys[cls]
        xs = stacked_xs
        preds = [cross_val_predict(reg, xs, ys, cv=5) for reg in regs]
        pred = np.mean(preds, axis=0)
        rmse = np.sqrt(metrics.mean_squared_error(ys, pred))
        print('cls {}, RMSE {:.2f}'.format(cls, rmse))
        all_rmse.append(np.mean(rmse))
        if save_to:
            fitted = []
            for reg in regs:
                reg = clone(reg)
                reg.fit(xs, ys)
                fitted.append(reg)
            fitted_regs.append(fitted)
    print('Average RMSE for {}: {:.2f}'.format(regs_name, np.mean(all_rmse)))
    if save_to:
        joblib.dump(fitted_regs, save_to)
        print('Saved to', save_to)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('root', type=Path)
    arg('mode', choices=['train', 'predict'])
    args = parser.parse_args()

    model_path = args.root.joinpath('regressor.joblib')  # type: Path
    features_path = args.root.joinpath('features.npz')  # type: Path

    if args.mode == 'train':
        if features_path.exists():
            data = np.load(str(features_path))
            xs, ys = data['xs'], data['ys']
        else:
            xs, ys = load_all_features(args.root, only_vald=True)
            with features_path.open('wb') as f:
                np.savez(f, xs=xs, ys=ys)
        evaluate_stacked(
            xs, ys,
            ExtraTreesRegressor(n_estimators=50),
            XGBRegressor(n_estimators=50, max_depth=2),
            Lasso(alpha=1.0, normalize=False, max_iter=100000),
            save_to=model_path)
    elif args.mode == 'predict':
        all_xs, _ = load_all_features(args.root.joinpath('test'), only_vald=False)
        all_regs = joblib.load(model_path)
        stacked_xs = np.concatenate(all_xs, axis=1)
        for cls, cls_regs in enumerate(all_regs):
            preds = [reg.predict(stacked_xs) for reg in cls_regs]
            pred = np.mean(preds, axis=0)
            # TODO


if __name__ == '__main__':
    main()
