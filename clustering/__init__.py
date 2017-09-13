"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 04 September, 2017 @ 9:58 PM.
  Copyright (c) 2017. Victor. All rights reserved.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def process_data(df):
    """
    Processes a dataframe in order to handle for non-numeric data
    :type df: pd.DataFrame
    :param df: the dataframe containing the data
    :return df: pd.DataFrame
    """
    columns = df.columns.values

    def convert(val):
        return text_digit[val]

    for col in columns:
        text_digit = {}  # {"Female": 0}
        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            uniques = set(df[col].values.tolist())
            x = 0
            for unique in uniques:
                if unique not in text_digit:
                    text_digit[unique] = x
                    x += 1
            df[col] = list(map(convert, df[col]))
    return df


def main():
    df = pd.read_excel("../datasets/titanic.xls")
    df.drop(['body', 'name'], 1, inplace=True)
    df.convert_objects(convert_numeric=True)
    df.fillna(0, inplace=True)
    df = process_data(df)
    print(df.head())


if __name__ == '__main__':
    main()
