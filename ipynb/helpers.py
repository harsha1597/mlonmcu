#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import glob
import os
import ast
import pandas as pd

# Given a list of columns, this function will add those columns to the dataframe
def add_column(df,cols):
    df["Config"] = df["Config"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    for col in cols:
        df[col] = df["Config"].apply(lambda x: x.get(col, None) if isinstance(x, dict) else None)
    return df

def find_newest_report():
    home = os.getenv("MLONMCU_HOME")
    list_of_files = glob.glob(os.path.join(home, "results", "*"))  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def tabularize_latest_report(add_cols=None):
    report_file = find_newest_report()
    df = pd.read_csv(report_file, sep=",")
    df = add_column(df,add_cols) if add_cols else df
    return df
