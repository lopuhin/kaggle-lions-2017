#!/usr/bin/env python3
import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('output')
    args = parser.parse_args()
    inputs = [pd.read_csv(fname, index_col=0) for fname in args.inputs]
    output = sum(df / len(inputs) for df in inputs)  # type: pd.DataFrame
    output.round().astype(int).to_csv(args.output)


if __name__ == '__main__':
    main()
