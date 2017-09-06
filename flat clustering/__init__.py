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
    """
    columns = df.columns.values
    for col in columns:
        pass


def main():
    df = pd.read_excel("../datasets/titanic.xls")
    df.drop(['body', 'name'], 1, inplace=True)
    df.convert_objects(convert_numeric=True)
    df.fillna(0, inplace=True)
    # print(df.head())
    process_data(df)


if __name__ == '__main__':
    main()
