
from datetime import datetime
import numpy as np
import pandas as pd
from os.path import join
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

from bnp_paribas_util import *


# ----------------------------------------------------------------------------------
#                        DATA FUNCTIONS
# ----------------------------------------------------------------------------------

def get_fraud(df, col_extract=None, verbose=0):
    extract1 = df.copy()
    if col_extract is not None:
        extract1 = extract1[col_extract]
    extract1 = extract1[extract1['fraud_flag']==1]
    extract1.loc["TOTAL"] = extract1.sum(axis=0)

    return extract1

def transpose_categories(df, remove_prefix="item_", verbose=0):
    df_temp = df.loc["TOTAL"]
    type(df_temp)
    dd = pd.DataFrame(data=df_temp)
    dd = dd.sort_values(by=['TOTAL'], ascending=False)
    dict_temp = defaultdict(list)
    
    ever_proceed = {'ID', "Nb_of_items", 'fraud_flag'}

    for idx in dd.index:
        cat_name = idx.replace("_cash", "")
        cat_name = cat_name.replace("_nb", "")
        if remove_prefix:
            cat_name = cat_name.replace(remove_prefix, "")
        else:
            remove_prefix = ""

        if cat_name not in ever_proceed:
            dict_temp['cat_name'].append(cat_name)
            ever_proceed.add(cat_name)
            cash_val = dd.loc[remove_prefix+cat_name+"_cash", 'TOTAL']
            nb_val = dd.loc[remove_prefix+cat_name+"_nb", 'TOTAL']
            dict_temp['total_nb'].append(nb_val)
            dict_temp['total_cash'].append(cash_val)

    dd2 = pd.DataFrame.from_dict(dict_temp)
    # garde uniquement les données des catégories qui ont été dans une fraude
    the_most_steel = dd2[(dd2['total_nb']>0)&(dd2['total_cash']>0)]
    the_most_steel = the_most_steel.sort_values(by=['total_nb', 'total_cash'], ascending=False)
    the_most_steel = the_most_steel.reset_index(drop=True)

    return the_most_steel

def extract_most_steel(df, cat_warn, col_extract=None, verbose=0):
    short_name = "extract_most_steel"
    cols1 = []
    cols_cash = []
    cols_nb = []

    if col_extract is None:
        col_extract = col_names_without_numeroted_card_col(df, verbose=0)

    for c in col_extract:
        for cc in cat_warn.values:
            if cc in c:
                cols1.append(c)
                if c.endswith("cash"):
                    cols_cash.append(c)
                else:
                    cols_nb.append(c)
                
    cols = deepcopy(cols1)
    cols.insert(0,'fraud_flag')
    cols.insert(0,'Nb_of_items')
    cols.insert(0,'ID')
    the_most = df[cols]
    the_most['TOTAL_cash'] = the_most[cols_cash].sum(axis=1)
    the_most['TOTAL_nb'] = the_most[cols_nb].sum(axis=1)
    if verbose>0:
        info(short_name, f"{the_most.shape}")
    most_reduce = the_most[the_most['TOTAL_nb']>0]
    if verbose>0:
        info(short_name, f"{most_reduce.shape}")
    return most_reduce
