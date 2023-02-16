
from datetime import datetime
import numpy as np
import pandas as pd
from os.path import join
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

"""
Module :
    1. pre_processing
    2. encode_data
    3. Dispatch UNKNOWN ?


------
Result :
    Final.1 dump
    Final.2 load
    Final.3 reduce_data_by_typing if need
"""

from bnp_paribas_util import *

CARD_COLS_START = ['item', 'model', 'goods_code', 'cash_price', 'make','Nbr_of_prod_purchas']

SUPER_CATEGORIES = {
        "COMPUTER"              :['APPLE S','2HP ELITEBOOK 850V6','COMPUTERS','HP ELITEBOOK 850V6',],
        "COMPUTER ACCESSORIES"  :['TARGUS GEOLITE ESSENTIAL CASE','TOSHIBA PORTABLE HARD DRIVE', 'PRINTERS & SCANNERS', 'PRINTERS SCANNERS','LOGITECH PEBBLE M350 BLUETOOTH MOUSE','BLANK MEDIA & MEDIA STORAGE', 'BLANK MEDIA MEDIA STORAGE', 'COMPUTER NETWORKING', 'COMPUTER PERIPHERALS & ACCESSORIES', 'COMPUTER PERIPHERALS ACCESSORIES','APPLE PRODUCTDESCRIPTION','2TARGUS GEOLITE ESSENTIAL CASE','AERIALS REMOTE CONTROLS','6  SPACE GREY 32GB','2TOSHIBA PORTABLE HARD DRIVE','2LOGITECH PEBBLE M350 BLUETOOTH MOUSE'],
        "SOFTWARE"              :['MICROSOFT OFFICE HOME AND STUDENT 2019,','2MICROSOFT OFFICE HOME AND STUDENT 2019,','COMPUTER SOFTWARE',],
        'AUDIO ACCESSORIES'     :['AUDIO ACCESSORIES'],
        'BABY'                  :['BABY CHANGING','BABY & CHILD TRAVEL','BABY CHILD TRAVEL','BABY FEEDING','BABY PLAY EQUIPMENT','BABYWEAR',],
        'BAGS'                  :['LUGGAGE','BAGS & CARRY CASES', 'BAGS CARRY CASES', 'BAGS WALLETS ACCESSORIES', 'BAGS, WALLETS & ACCESSORIES',],
        'OUTDOOR ACCESSORIES'   :['OUTDOOR ACCESSORIES','OUTDOOR FURNITURE','GARDENING EQUIPMENT','BARBECUES & ACCESSORIES', 'BARBECUES ACCESSORIES',],
        'BEAUTY AND SAFETY'     :['SUNCARE', 'SUNGLASSES & READING GLASSES', 'SUNGLASSES READING GLASSES','MAKEUP','HAIRCARE', 'HEALTH & BEAUTY ELECTRICAL', 'HEALTH BEAUTY ACCESSORIES', 'HEALTH BEAUTY ELECTRICAL','FRAGRANCE','BATH & BODYCARE', 'BATH BODYCARE', 'FACIAL SKINCARE',],
        'HOUSE LINEN'           :['TABLE LINEN','SOFT FURNISHINGS','LINGERIE & HOISERY', 'LINGERIE HOISERY','LAUNDRY & CLOTHESCARE', 'LAUNDRY CLOTHESCARE','BATH LINEN','BED LINEN','CARPETS RUGS FLOORING', 'CARPETS, RUGS & FLOORING',],
        'HOUSE ACCESSORIES'     :['TABLEWARE', 'WINDOW DRESSING','STANDS & BRACKETS', 'STANDS BRACKETS','PRESERVING & BAKING EQUIPMENT', 'PRESERVING BAKING EQUIPMENT','LIVING & DINING FURNITURE', 'LIVING DINING FURNITURE','LIGHTING','BARWARE','KITCHEN ACCESSORIES', 'KITCHEN SCALES & MEASURES', 'KITCHEN SCALES MEASURES', 'KITCHEN STORAGE', 'KITCHEN UTENSILS & GADGETS', 'KITCHEN UTENSILS GADGETS','HOUSEHOLD CLEANING','HOME AND PERSONAL SECURITY', 'HOME OFFICE', 'HOME SAFETY EQUIPMENT','HEATING & COOLING APPLIANCES', 'HEATING COOLING APPLIANCES','FOOD STORAGE','FITTED KITCHENS','BATHROOM', 'BATHROOM ACCESSORIES', 'BATHROOM FIXTURES', 'BEDROOM FURNITURE','CHRISTMAS DECORATIONS','COOKING APPLIANCES', 'COOKWARE','DECORATING', 'DECORATIVE ACCESSORIES','DOOR FURNITURE','DISPOSABLE TABLEWARE CUTLERY','EASTER DECORATIONS',],
        'CHILDRENS'             :['TOYS','SCHOOLWEAR','GIRLSWEAR','CHILDREN S ACCESSORIES', 'CHILDREN S FOOTWEAR', 'CHILDREN S FURNITURE', 'CHILDRENS FOOTWEAR','BOYSWEAR',],
        'NURSERY'               :['NURSERY ACCESSORIES', 'NURSERY EQUIPMENT FURNITURE', 'NURSERY FURNITURE', 'NURSERY LINEN', 'NURSERY TOYS',],
        'WOMEN'                 :['WOMEN S ACCESSORIES', 'WOMEN S CLOTHES', 'WOMEN S FOOTWEAR', 'WOMEN S NIGHTWEAR', 'WOMENS ACCESSORIES', 'WOMENS CLOTHES', 'WOMENS FOOTWEAR'],
        'MEN'                   :['MEN S ACCESSORIES', 'MEN S CLOTHES', 'MEN S FOOTWEAR', 'MEN S NIGHTWEAR', 'MEN S SPORTSWEAR', 'MEN S UNDERWEAR SOCKS', 'MENS CLOTHES', 'MENS NIGHTWEAR', 'MENS UNDERWEAR & SOCKS',],
        'CABLES ADAPTERS'       :['CABLES & ADAPTERS', 'CABLES ADAPTERS',],
        'BOOKS'                 :['BOOKS',],
        'OFFICE ACCESSORIES'    :['STATIONERY SUNDRIES', 'STORAGE & ORGANISATION', 'STORAGE ORGANISATION','PENS PENCILS','PAPER NOTEBOOKS','DIARIES & ORGANISERS', 'DIARIES ORGANISERS','FILING DESK ACCESSORIES'],
        'GAMES'                 :['GAMES', 'GAMING',],
        'SPORT'                 :['SPORTS EQUIPMENT','GYM EQUIPMENT',],
        'POWER ACCESSORIES'     :['FULFILMENT CHARGE','POWER & BATTERIES', 'POWER BATTERIES',],
        'HI-FI'                 :['TELEVISIONS & HOME CINEMA', 'TELEVISIONS HOME CINEMA','VIDEOS DVD DIGITAL EQUIPMENT','HI-FI','PORTABLE AUDIO EQUIPMENT',],
        'FOOD'                  :['PICNICWARE','FOOD PREPARATION','GIFT FOOD DRINK','HOT DRINK PREPARATION',],
        'IMAGING EQUIPMENT'     :['IMAGING ACCESSORIES', 'IMAGING EQUIPMENT',],
        'JEWELLERY'             :['JEWELLERY & WATCHES', 'JEWELLERY WATCHES',],
        'TELEPHONE ACCESSORIES' :['TECHNOLOGY ACCESSORIES','TELEPHONE ACCESSORIES', 'TELEPHONES FAX MACHINES TWO-WAY RADIOS', 'TELEPHONES, FAX MACHINES & TWO-WAY RADIOS','TELEPHONE ACCESSORIES'],
        'OTHER'                 :['THEMED GIFTS','PRODUCT','PARTY DECORATIONS','CRAFT','DRESSMAKING','GIFT WRAP', 'GREETING CARDS & PERSONALISED STATIONERY', 'GREETING CARDS PERSONALISED STATIONERY',],
        'UNKNOWN'               :['UNKNOWN',],
        'SERVICE'               :['SERVICE','WARRANTY',],  
}

ITEM_CATEGORIES = {}

def initITEM_CATEGORIES():
    if len(ITEM_CATEGORIES)==0:
        for key, val in SUPER_CATEGORIES.items():
            ITEM_CATEGORIES[val] = key
            ITEM_CATEGORIES[_clean_categories(val)] = key
    return ITEM_CATEGORIES
# ----------------------------------------------------------------------------------
#                        PRE-PROCESSING
# ----------------------------------------------------------------------------------
def pre_processing(X, y=None,save_file_path = None, verbose=0):
    """1. Fusion des DF X et Y
       2. Uniformisation des écritures des items, make et model
       3. Suppression des colonnes de code
       4. Sauvegarde du fichier

    Args:
        X (DataFrame): _description_
        y (DataFrame, optional): _description_. Defaults to None.
        save_file_path (str, optional): _description_. Defaults to None.
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        DataFrame: _description_
    """
    short_name = "pre_processing"
    dataset_train = None

    # 1. Fusion des DF X et Y
    if y is not None:
        dataset_train = pd.merge(left=X, right=y, on='ID', indicator=True)
        if verbose>0:
            info(short_name, f'\n{dataset_train["_merge"].value_counts()}')
        
        dataset_train = dataset_train.drop(columns=["_merge"])
        # On déplace les colonnes intéressantes au début de la DF
        cols = list(dataset_train.columns)
        for c_n in ['fraud_flag', 'Nb_of_items']:
            try:
                cols.remove(c_n)
                cols.insert(1, c_n)
            except :
                pass

        dataset_train = dataset_train[cols]
        # print(dataset_train.columns)
    else:
        dataset_train = X.copy()
        
    # 2. Uniformisation des écritures des items
    # Certains items sont identiques mais écris différemment, une étape d'uniformisation est nécessaire...
    for i in tqdm(range(1, 25), desc="clean", disable=verbose<1):
        # à l'origine il y avait 173 items, après nettoyage => 162 items
        col = f'item{i}'
        dataset_train[col] = dataset_train[col].apply(lambda x: _clean_categories(input_str=x))
        # à l'origine il y avait 829 maker, après nettoyage => 827 makers
        col = f'make{i}'
        dataset_train[col] = dataset_train[col].apply(lambda x: _clean_categories(input_str=x))
        col = f'model{i}'
        dataset_train[col] = dataset_train[col].apply(lambda x: _clean_categories(input_str=x))

    # 3. Suppression des colonnes de code
    if verbose>0:
        info(short_name, f'input {dataset_train.shape}')

    dataset_train = drop_numeroted_data_col(df=dataset_train, cols_name=["goods_code"], verbose=verbose)
    if verbose>0:
        info(short_name, f'output {dataset_train.shape}')

    dump_file(data_set_path=save_file_path, df=dataset_train, verbose=verbose)
        
    return dataset_train


def encode_data(data_set_path,df, file_name= "encoded_data.csv", force_reloading=0, with_drop=1, to_encode_data=['item', 'make'], add_item_group=1,verbose=0):
    """
    1. drop unused columns
    2. Encode `to_encode_data` columns
    3. Typing datas to reduce memory space
    4. drop encoded data
    5. Add amount column
    6. Groups items if `add_item_group` 
    Final. Save result and return if
    
    Note :  If target file ever exist and not `force_reloading` load the file.

    Args:
        data_set_path (str): _description_
        df (DataFrame or None): _description_
        file_name (str, optional): _description_. Defaults to "encoded_data.csv".
        force_reloading (int, optional): _description_. Defaults to 0.
        with_drop (int, optional): _description_. Defaults to 1.
        to_encode_data (list, optional): _description_. Defaults to ['item', 'make'].
        add_item_group (int, optionnal) : 1 to group item, 2 to remove items columns after, 0 to do nothing, deefault 1
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        DataFrame: _description_
    """
    short_name = "encode_data"
    df_res = load_dump_file(data_set_path=data_set_path, file_name=file_name, force_reloading=force_reloading, verbose=verbose)
    save_file_path = join(data_set_path, file_name)

    if df_res is None:

        timing = {
            'item' : '> 30 min',
            'make' : '~= 270 min ~= 4h30'
        }

        df_res = df.copy()
        if verbose>0:
            info(short_name, f"{df_res.shape} datas to encode")

        # 1. on réduit la taille de la df aux données que l'on souhaite conserver uniquement pour réduire le taille en mémoire et le temps de traitement
        if verbose>1:
            debug(short_name, f"STEP 1 => remove unsed columns")
        to_del_col = deepcopy(CARD_COLS_START)
        for e in to_encode_data:
            try:
                to_del_col.remove(e)
            except:
                pass
        
        for to_del in to_del_col:
            df_res = drop_numeroted_data_col(df=df_res, cols_name=[to_del], verbose=verbose)
            
        if verbose>0:
            info(short_name, f"STEP 1 => {len(to_del_col)} unsed columns {surligne_text('DROP')} --> {df_res.shape}")

        for to_enc in tqdm(to_encode_data, disable=verbose<1):

            # /!\ 42 min de traitement lors du forçage
            if verbose>0:
                warn(short_name, f"STEP 2 => encoding {to_enc.upper()} to features {surligne_text(timing.get(to_enc, 'UNKNOWN duration'), 'red')}...")

            list_data = _get_data_list(df=df_res, col_name=to_enc, verbose=verbose-1)
            df_res = _encode_numeroted_data_to_features(df=df_res, data_list=list_data, data_col_name=to_enc, with_drop=with_drop, verbose=verbose)

            df_res = df_res.reset_index()
            df_res = df_res.set_index('index')
            df_res = drop_numeroted_data_col(df=df_res, cols_name=[to_enc], verbose=verbose)
            f_name = save_file_path.replace("complete", to_enc.upper())
            dump_file(data_set_path=f_name, df=df_res, verbose=verbose)
                        
            if verbose>0:
                info(short_name, f"STEP 2 => encoding {to_enc.upper()} to features {surligne_text('SAVE')}")
                if verbose>1:
                    debug(short_name, f"{f_name}")
        
        df_res = add_amounts(x_df_input=df_res, verbose=verbose)
        if verbose>0:
            info(short_name, f"STEP 3 => Add Amount {surligne_text('DONE')}")

        if add_item_group:
            if verbose>1: debug(short_name, f"STEP 4 => Group items")
            with_drop= 1 if add_item_group == 2 else 0
            f_name = save_file_path.replace("complete", "encoded_data_group")
            df_res = add_item_group_datas_to_features(df=df_res,data_set_path=f_name, with_drop=with_drop, verbose=verbose)
            if verbose>0:
                info(short_name, f"STEP 4 => Group items {surligne_text('DONE')}")

        # Sauvegarde du fichier
        if verbose>1: debug(short_name, f"Final STEP => Saving File ...")
        dump_file(data_set_path, file_name, df=df_res, verbose=verbose)
        
    return df_res

def _encode_numeroted_data_to_features(df, data_list, data_col_name='item', with_drop=0, verbose=0):
    short_name = "encode_numeroted_data_to_features"
    if verbose > 0: print(f"[{short_name}]\tINFO: Conversion {data_col_name.upper()} to feature... START")
    # Il faut prévoir le cas inconnu pour y ajouter des données inconnues.
    data_list.add("UNKNOWN")
    df_res = df.copy()
    
    for current_name in tqdm(data_list, desc=f"{data_col_name}_list", disable=verbose<1):
        col = data_col_name+"_"+current_name+"_nb"
        df_res[col] = df_res.apply(lambda x : nb_by_col(current_name=current_name, data_col_name=data_col_name, row=x, col_addition='Nbr_of_prod_purchas', verbose=verbose-1), axis=1)
        # Typage des données
        df_res[col] = df_res[col].round(decimals=0)
        df_res[col] = df_res[col].astype("int8")
        col = data_col_name+"_"+current_name+"_cash"
        df_res[col] = df_res.apply(lambda x : nb_by_col(current_name=current_name, data_col_name=data_col_name, row=x, col_addition='cash_price', verbose=verbose-1), axis=1)
        # Typage des données
        df_res[col] = df_res[col].round(decimals=2)
        df_res[col] = df_res[col].astype("float16")

    if verbose > 0: print(f"[{short_name}]\tINFO: Conversion {data_col_name.upper()} to feature............ END")
    
    if with_drop:
        if verbose>0:
            info(short_name, f"Drop {data_col_name} columns ... ")
        df_res = drop_numeroted_data_col(df=df_res, cols_name=[data_col_name], verbose=verbose)
       
    return df_res

def add_item_group_datas_to_features(df,data_set_path, file_name= "encoded_data_group.csv",force_reloading=0, with_drop=0, verbose=0):
    """Group items by categories

    Args:
        df (DataFrame): _description_
        with_drop (int, optional): To drop the items columns. Defaults to 0.
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        DataFrame: _description_
    """
    short_name = "add_item_group_datas_to_features"
    df_res = load_dump_file(data_set_path=data_set_path, file_name=file_name, force_reloading=force_reloading, verbose=verbose)
            
    if df_res is None:

        data_col_name='item'
        df_columns = list(df.columns)

        # Il faut prévoir le cas inconnu pour y ajouter des données inconnues.
        for cate, cols in SUPER_CATEGORIES.items():
            cols_names_nb = []
            cols_names_cash = []
            for col in cols:
                curr = data_col_name+"_"+col+"_nb"
                if curr in df_columns:
                    cols_names_nb.append(curr)
                curr = data_col_name+"_"+col+"_cash"
                if curr in df_columns:
                    cols_names_cash.append(curr)
            if len(cols_names_cash) != len(cols_names_nb):
                warn(short_name, f"{cate} Le nombre de données est différent : {cols_names_nb} vs {cols_names_cash}")
            if len(cols_names_nb)>0:
                df["group_"+data_col_name+"_"+cate+"_nb"] = df[cols_names_nb].sum(axis=0)
                if with_drop:
                    df = df.drop(cols_names_nb, axis=1)
            if len(cols_names_cash)>0:
                df["group_"+data_col_name+"_"+cate+"_cash"] = df[cols_names_cash].sum(axis=0)
                if with_drop:
                    df = df.drop(cols_names_cash, axis=1)
        df_res = df
        dump_file(data_set_path, file_name, df=df_res, verbose=verbose)

    if verbose > 0: info(short_name, f"{df_res.shape} result data.")
    return df_res


def reduce_data_by_typing(df, verbose=0):
    """ Round float data to x.xx and convert float64 nb to int8 and float data float16

    Args:
        df (DataFrame): _description_
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    short_name = "reduce_data_by_typing"
    if verbose>0:info(short_name, f"{df.dtypes}")
    for col in df.columns:
        if df[col].dtype == 'float64':
            if col.endswith('_nb') or col in ['Nb_of_items']:
                df[col] = df[col].round(decimals=0)
                df[col] = df[col].astype("int8")
            else:
                df[col] = df[col].round(decimals=2)
                df[col] = df[col].astype("float16")
        elif df[col].dtype == 'int64' and col not in ['index', 'ID']:
            df[col] = df[col].astype("int8")
    if verbose>0:info(short_name, f"{df.dtypes}")
    return df


def load_test_df(file_path, train_columns, force=False, verbose=0):
    """Load the official test df and add : amount and reducre data by typing, then save the new df.
    If the df have been save, just load it.

    Args:
        file_path (str): _description_
        train_columns (list(str)): _description_
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        DataFrame: _description_
    """
    short_name = "load_test_df"
    save_path = file_path.replace(".csv", "_rounded.csv")

    test_origin = load_dump_file(data_set_path=save_path, force_reloading=force, verbose=verbose)
    
    if test_origin is None:
        test_origin = load_dump_file(data_set_path=file_path, force_reloading=force, verbose=verbose)
        
        # Ajout des colonnes manquantes
        test_origin = add_amounts(test_origin, verbose=verbose)

        test_cols = test_origin.columns
        for col in train_columns:
            if col not in test_cols:
                test_origin[col] = 0
        
        test_origin = test_origin[train_columns]
        test_origin = reduce_data_by_typing(df=test_origin, verbose=verbose)
        if verbose>0:
            info(short_name, f"{test_origin.shape} test données mises à jour")

        dump_file(data_set_path=save_path, df=test_origin, verbose=verbose)

    if verbose>0:
        info(short_name, f"{test_origin.shape} test données chargées")
    return test_origin

def add_amounts(x_df_input, verbose=0):
    """Add the amount column, sum of all item cas column

    Args:
        x_df_input (DataFrame): _description_
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        DataFrame: new df
    """
    ncol_item = "amount"
    
    x_df = x_df_input.copy()
    cols = sorted(list(x_df.columns))
    start = False
    if ncol_item not in cols:
        x_df[ncol_item] = 0
                
        for col in cols:
            if col.lower().startswith("item_"):
                start = True
                if col.lower().endswith("_cash"):
                    x_df[col] = x_df[col].fillna(0)
                    x_df[ncol_item] = x_df[ncol_item] + x_df[col]
            # On sort de la boucle dès que le nom de colonne ne commence pas par item
            elif start:
                break
    return x_df

# ----------------------------------------------------------------------------------
#                        DATA FUNCTIONS
# ----------------------------------------------------------------------------------

def _clean_categories(input_str):
    # BLANK MEDIA & MEDIA STORAGE => BLANK MEDIA MEDIA STORAGE
    output_str = input_str
    if isinstance(output_str, str):
        output_str = output_str.replace(" & ", "")
        # CHILDREN S ACCESSORIES => CHILDRENS ACCESSORIES
        output_str = output_str.replace(" S ","S ")
        # TELEPHONES, FAX MACHINES & TWO-WAY RADIOS => TELEPHONES FAX MACHINES TWO-WAY RADIOS
        output_str = output_str.replace(",", "")
        # 2HP ELITEBOOK 850V6 => HP ELITEBOOK 850V6
        if output_str.startswith("2"):
            output_str = output_str[1:]
        
        output_str = output_str.strip()
    return output_str


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


def col_names_without_numeroted_card_col(df, verbose=0):
    short_name = "col_names_without_numeroted_card_col"
    col_extract = []
    # On enlève toutes les colonnes liées à la place dans le panier, ce n'est pas ce qui nous intéresse
    for col_name in list(df.columns):
        if not col_name.startswith('item') and not col_name.startswith('model') and not col_name.startswith('goods_code') and not col_name.startswith('cash_price') and not col_name.startswith('make') and not col_name.startswith('Nbr_of_prod_purchas'):
            col_extract.append(col_name)
    if verbose>0:
        info(short_name, f"{col_extract}")
    return col_extract


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

def nb_by_col(current_name, data_col_name, row, col_addition='Nbr_of_prod_purchas', verbose=0):
    nb_item = 0
    for i in range(1, 25):
        item_val = row[f'{data_col_name}{i}']
        if isinstance(item_val, str) and current_name == item_val:
            nb_item += row[f'{col_addition}{i}']
    return round(nb_item, 2)

def drop_numeroted_data_col(df, cols_name, verbose=0):
    short_name = "drop_numeroted_data_col"
    n_df = df.copy()
    nb_col_droped = 0 
    if verbose>1:
        debug(short_name, f"input shape {n_df.shape}")  
        
    for i in tqdm(range(1, 25), desc="Drop column", disable=verbose<1)  :
        for col_name in cols_name:
            try:
                n_df = n_df.drop(columns=[f'{col_name}{i}'])
                nb_col_droped += 1
            except:
                pass
    
    if verbose>0:
        info(short_name, f"{nb_col_droped} columns droped")
        if verbose>1:
            debug(short_name, f"output shape {n_df.shape}")   
    return n_df

def _get_data_list(df, col_name, verbose=0):
    short_name = f'get_{col_name}_list'
    
    items_list = set()
    for i in tqdm(range(1, 25), desc=f"{col_name}_list", disable=verbose<1):
        items_list = items_list | set(df[f'{col_name}{i}'].unique())
    
    if verbose>1:
        debug(short_name, f'{len(items_list)} with NA value')
    
    items_list.remove(np.nan)

    if verbose>0:
        info(short_name, f'{len(items_list)} {col_name} without NA value')
        if verbose>1:
            debug(short_name, f'',items_list)
    return items_list