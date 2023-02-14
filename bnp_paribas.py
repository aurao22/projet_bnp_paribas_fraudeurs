
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join, exists, getsize
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

from pr_auc_score_SB05ixL import complete_y_cols, evaluate

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
            ITEM_CATEGORIES[clean_categories(val)] = key
    return ITEM_CATEGORIES
# ----------------------------------------------------------------------------------
#                        PRE-PROCESSING
# ----------------------------------------------------------------------------------
def pre_processing(X, y=None,save_file_path = None, verbose=0):
    short_name = "pre_processing"
    dataset_train = None

    # 1. Fusion des DF X et Y
    if y is not None:
        dataset_train = pd.merge(left=X, right=y, on='ID', indicator=True)
        if verbose>0:
            print(f'[{short_name}]\tINFO : \n{dataset_train["_merge"].value_counts()}')
        
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
        dataset_train[col] = dataset_train[col].apply(lambda x: clean_categories(input_str=x))
        # à l'origine il y avait 829 maker, après nettoyage => 827 makers
        col = f'make{i}'
        dataset_train[col] = dataset_train[col].apply(lambda x: clean_categories(input_str=x))
        col = f'model{i}'
        dataset_train[col] = dataset_train[col].apply(lambda x: clean_categories(input_str=x))


    # 3. Suppression des colonnes de code
    if verbose>0:
        print(f'[{short_name}]\tINFO : input {dataset_train.shape}')

    dataset_train = drop_numeroted_data_col(df=dataset_train, cols_name=["goods_code"], verbose=verbose)
    if verbose>0:
        print(f'[{short_name}]\tINFO : output {dataset_train.shape}')

    if save_file_path is not None:
        # Sauvegarde du fichier
        dataset_train.to_csv(save_file_path)
        if verbose > 0: print(f"[{short_name}]\tINFO: {save_file_path} => SAVED")

    return dataset_train

def encode_data(data_set_path,df, file_name= "encoded_data.csv", force_reloading=0, with_drop=1,verbose=0):
    short_name = "encode_data"
    df_res = None
    save_file_path = join(data_set_path, file_name)

    if exists(save_file_path) and getsize(save_file_path) > 0 and not force_reloading:
        if verbose > 0: print(f"[{short_name}]\tINFO: {file_name} => Exist")
        # Chargement de la DF fichier
        df_res = pd.read_csv(save_file_path, sep=",", index_col="index",low_memory=False)
        
    if df_res is None:
        if verbose>0:
            print(f"[{short_name}]\tINFO : STEP 1 => encoding ITEM to features about 30 min ... START")
        # /!\ 42 min de traitement lors du forçage
        df_res = _encode_item_to_features(df=df, with_drop=with_drop,verbose=verbose)
        df_res = df_res.reset_index()
        df_res = df_res.set_index('index')
        df_res.to_csv(save_file_path.replace("complete", 'ITEMS'))
        df_temp = drop_numeroted_data_col(df=df_res, cols_name=["cash_price", "make", "model", "Nbr_of_prod_purchas"], verbose=verbose)
        df_temp.to_csv(save_file_path.replace("complete", 'ITEMS_light'))

        if verbose>0:
            print(f"[{short_name}]\tINFO : STEP 2 => encoding MARK to features about 270 min soit 4h30 ... START")
        df_res = _encode_make_to_features(df=df_res, with_drop=with_drop, verbose=verbose)

        # Sauvegarde du fichier
        if verbose>0: print(f"[{short_name}]\tINFO : STEP 3 => Saving File ...")
        df_res.to_csv(save_file_path)
        df_temp = drop_numeroted_data_col(df=df_res, cols_name=["cash_price", "make", "model", "Nbr_of_prod_purchas"], verbose=verbose)
        df_temp.to_csv(save_file_path.replace("complete", 'complete_light'))
        if verbose > 0: print(f"[{short_name}]\tINFO: {file_name} => SAVED")

    return df_res

def _encode_item_to_features(df, with_drop=0,verbose=0):
    items_list = get_item_list(df=df, verbose=verbose-1)
    return _encode_numeroted_data_to_features(df=df, data_list=items_list, data_col_name='item', with_drop=with_drop, verbose=verbose)

def _encode_make_to_features(df, with_drop=0, verbose=0):
    make_list = get_maker_list(df=df, verbose=verbose-1)
    return _encode_numeroted_data_to_features(df=df, data_list=make_list, data_col_name='make', with_drop=with_drop, verbose=verbose)

def _encode_numeroted_data_to_features(df, data_list, data_col_name='item', with_drop=0, verbose=0):
    short_name = "encode_numeroted_data_to_features"
    if verbose > 0: print(f"[{short_name}]\tINFO: Conversion {data_col_name.upper()} to feature... START")
        
    df_res = df.copy()
    
    for current_name in tqdm(data_list, desc=f"{data_col_name}_list", disable=verbose<1):
        df_res[data_col_name+"_"+current_name+"_nb"] = df_res.apply(lambda x : nb_by_col(current_name=current_name, data_col_name=data_col_name, row=x, col_addition='Nbr_of_prod_purchas', verbose=verbose-1), axis=1)
        df_res[data_col_name+"_"+current_name+"_cash"] = df_res.apply(lambda x : nb_by_col(current_name=current_name, data_col_name=data_col_name, row=x, col_addition='cash_price', verbose=verbose-1), axis=1)
    
    if verbose > 0: print(f"[{short_name}]\tINFO: Conversion {data_col_name.upper()} to feature............ END")
    
    if with_drop:
        if verbose>0:
            print(f"[{short_name}]\tINFO : Drop {data_col_name} columns ... ")
        df_res = drop_numeroted_data_col(df=df_res, cols_name=[data_col_name], verbose=verbose)
       
    return df_res

def prefixed_cols(df, col_prefix, verbose=0):
    short_name = "prefixed_cols"
    cols = []
    for col in df.columns:
        if col.startswith(col_prefix):
            cols.append(col)
    
    if verbose > 0: print(f"[{short_name}]\tINFO: {len(cols)} columns prefixed by {col_prefix}")        
    return cols


def load_test_df(file_path, train_columns, verbose=0):
    test_origin = pd.read_csv(file_path, sep=',',index_col="index" ,low_memory=False)
    if verbose>0:
        print(f"{test_origin.shape} test données chargées")
    
    # Ajout des colonnes manquantes
    test_origin = add_amounts(test_origin, verbose=verbose)

    test_cols = test_origin.columns
    for col in train_columns:
        if col not in test_cols:
            test_origin[col] = 0
    
    test_origin = test_origin[train_columns]
    if verbose>0:
        print(f"{test_origin.shape} test données mises à jour")
    return test_origin

def add_amounts(x_df_input, verbose=0):
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
#                        DATASET FUNCTION
# ----------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from os.path import exists

_DATASET_KEYS = ["X_train", "X_test", "y_train", "y_test"]

def load_splited_data(dataset_path, test_size = 0.2, random_state = 42, save_it=True, force=False, verbose=0):
    short_name="load_splited_data"
    
    dataset_dict = {}

    if not force:
        for set_name in _DATASET_KEYS:
            file_name = dataset_path.replace(".csv", f"_split_{set_name}.csv")
            if exists(file_name):
                if verbose>0:info(short_name, f"Loading {set_name}...")
                dataset_dict[set_name] = pd.read_csv(file_name, sep=',', low_memory=False)
                if set_name.startswith("X_"):
                    dataset_dict[set_name] = add_amounts(dataset_dict[set_name], verbose=verbose)
            else:
                break
        
    # si les fichiers split n'existent pas :
    if force or len(dataset_dict) < 4:       
        
        if verbose > 0: info(short_name, f"Chargement des données train sources...")
        train_origin = pd.read_csv(dataset_path, sep=',', index_col="index" ,low_memory=False)
        dataset_dict['train_origin'] = train_origin
        if verbose > 0: info(short_name,f"Train d'originr {train_origin.shape} {surligne_text('LOAD')}")

        if verbose > 0: info(short_name, f"Séparation de X et y...")
        target = 'fraud_flag'
        columns = list(train_origin.columns)
        if verbose > 1: debug(short_name, len(columns))
        columns.remove(target)
        if verbose > 1: debug(short_name, len(columns))
        X = train_origin[columns]
        y = train_origin[target]

        if verbose > 0: info(short_name, f"Split du dataset_path...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Ajout des données manquantes
        X_train = add_amounts(X_train, verbose=verbose)
        X_test  = add_amounts(X_test, verbose=verbose)

        y_train = complete_y_cols(X_test=X_train, y_param=y_train) 
        y_test  = complete_y_cols(X_test=X_test, y_param=y_test) 

        for dataset_name, dataset in zip(_DATASET_KEYS, [X_train, X_test, y_train, y_test]):
            dataset_dict[dataset_name] = dataset
            if save_it and dataset is not None:
                dataset.to_csv(dataset_path.replace(".csv", f"_split_{dataset_name}.csv"))
                    
    if verbose > 0:
        info(short_name, f"{len(dataset_dict)} dataset {surligne_text('LOAD')}")
        if len(dataset_dict)>0: info(short_name, f"{dataset_dict.get(_DATASET_KEYS[0]).shape} / {dataset_dict.get(_DATASET_KEYS[1]).shape}")
    return dataset_dict
# ----------------------------------------------------------------------------------
#                        DATA FUNCTIONS
# ----------------------------------------------------------------------------------

def clean_categories(input_str):
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
    col_extract = []
    # On enlève toutes les colonnes liées à la place dans le panier, ce n'est pas ce qui nous intéresse
    for col_name in list(df.columns):
        if not col_name.startswith('item') and not col_name.startswith('model') and not col_name.startswith('goods_code') and not col_name.startswith('cash_price') and not col_name.startswith('make') and not col_name.startswith('Nbr_of_prod_purchas'):
            col_extract.append(col_name)
    if verbose>0:
        print(col_extract)
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
        print(the_most.shape)
    most_reduce = the_most[the_most['TOTAL_nb']>0]
    if verbose>0:
        print(most_reduce.shape)
    return most_reduce

def nb_by_col(current_name, data_col_name, row, col_addition='Nbr_of_prod_purchas', verbose=0):
    nb_item = 0
    for i in range(1, 25):
        item_val = row[f'{data_col_name}{i}']
        if isinstance(item_val, str) and current_name == item_val:
            nb_item += row[f'{col_addition}{i}']
    return nb_item

def get_item_list(df, verbose=0):
    """Récupération de la liste des items pour en créer une catégorie

    Args:
        df (_type_): _description_
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    return _get_data_list(df=df, col_name="item", verbose=verbose)

def get_maker_list(df, verbose=0):
    return _get_data_list(df=df, col_name="make", verbose=verbose)

def _get_data_list(df, col_name, verbose=0):
    short_name = f'get_{col_name}_list'
    items_list = set()
    for i in tqdm(range(1, 25), desc=f"{col_name}_list", disable=verbose<1):
        items_list = items_list | set(df[f'{col_name}{i}'].unique())
    
    if verbose>1:
        print(f'[{short_name}]\tDEBUG : ',len(items_list), "with NA value")
    
    items_list.remove(np.nan)

    if verbose>0:
        print(f'[{short_name}]\tINFO : ',len(items_list), col_name, "without NA value")
        if verbose>1:
            print(f'[{short_name}]\tDEBUG : ',items_list)
    return items_list

def drop_numeroted_data_col(df, cols_name, verbose=0):
    short_name = "drop_numeroted_data_col"
    n_df = df.copy()
    nb_col_droped = 0 
    if verbose>1:
        print(f"[{short_name}]\tDEBUG : input shape {n_df.shape}")  
        
    for i in tqdm(range(1, 25), desc="Drop column", disable=verbose<1)  :
        for col_name in cols_name:
            try:
                n_df = n_df.drop(columns=[f'{col_name}{i}'])
                nb_col_droped += 1
            except:
                pass
    
    if verbose>0:
        print(f"[{short_name}]\tINFO : {nb_col_droped} columns droped")
        if verbose>1:
            print(f"[{short_name}]\tDEBUG : output shape {n_df.shape}")   
    return n_df

# ----------------------------------------------------------------------------------
#                        MODEL FUNCTION
# ----------------------------------------------------------------------------------
def train_model(model, model_name, 
                dataset_dict, data_set_path, 
                scores,score_path, 
                params, features="ALL", add_data=np.nan,commentaire=np.nan,target="fraud_flag", 
                verbose=0):
    short_name = "train_model"
    if verbose>1:debug(short_name, f"{model_name} model fiting...")
    model.fit(dataset_dict.get("X_train"), dataset_dict.get("y_train")[target])

    if verbose>1:debug(short_name,f"Model evaluation...")
    score_accuracy = model.score(dataset_dict.get("X_test"), dataset_dict.get("y_test")[target])
    y_pred = model.predict(dataset_dict.get("X_test"))
    pr_auc_score_, _, _ = evaluate(X_test=dataset_dict.get("X_test"), y_test=dataset_dict.get("y_test"), y_pred=y_pred, verbose=verbose)
    if verbose>0:info(short_name,f"accuracy score : {score_accuracy}, pr_auc score : {pr_auc_score_}")

    test_origin = dataset_dict.get("test_origin", None)
    if test_origin is not None:
        if verbose>0:info(short_name,f"Prediction for test orgine...")
        y_pred_test = model.predict(test_origin)
        y_pred_test_complete = complete_y_cols(X_test=test_origin, y_param=y_pred_test)

        res_path = join(data_set_path, 'official_test_predictions', model_name+"_"+features+"_"+datetime.now().strftime('%Y-%m-%d-%H_%M')+".csv")
        y_pred_test_complete.to_csv(res_path)
    elif verbose>0:info(short_name,f"No prediction for test orgine")

    if verbose>1:debug(short_name,f"Add score...")
    n_scores = add_score(scores_param=scores, modele=model_name, features=features, add_data=add_data, 
            params=params, 
            accuracy_score=score_accuracy, pr_auc_score_TEST_perso=pr_auc_score_,
            commentaire=commentaire,
            score_path=score_path.replace(".csv", f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"),
            verbose=verbose)

    return model, n_scores

from sklearn.linear_model import LogisticRegression
def train_LogisticRegression(dataset_dict, data_set_path, 
                            scores, score_path, 
                            features="ALL", add_data=np.nan,commentaire=np.nan, 
                            verbose=0):
    
    if verbose>1:print(f"Model creation...")
    my_fist_model = LogisticRegression(penalty="l2", fit_intercept=True,solver='liblinear')
    params="penalty='l2', fit_intercept=True,solver='liblinear'"

    model, n_scores = train_model(model=my_fist_model, model_name="LogisticRegression", 
                                dataset_dict=dataset_dict, data_set_path=data_set_path, scores=scores, score_path=score_path, params=params, features=features,
                                add_data=add_data,commentaire=commentaire, verbose=verbose)
    
    return model, n_scores

import lightgbm as lgb
def train_LGBMClassifier(dataset_dict, data_set_path, 
                            scores, score_path, 
                            features="ALL", add_data=np.nan,commentaire=np.nan, 
                            verbose=0):
    
    if verbose>1:print(f"Model creation...")
    lgb_classifier = lgb.LGBMClassifier(boosting_type='goss',  
                                    max_depth=5, 
                                    learning_rate=0.1,
                                    n_estimators=1000, 
                                    subsample=0.8,  
                                    colsample_bytree=0.6,
                                   )
    params="boosting_type='goss', max_depth=5,learning_rate=0.1,n_estimators=1000.subsample=0.8,colsample_bytree=0.6"
    model, n_scores = train_model(model=lgb_classifier, model_name="LGBMClassifier", 
                                dataset_dict=dataset_dict, data_set_path=data_set_path, scores=scores, score_path=score_path, params=params, features=features,
                                add_data=add_data,commentaire=commentaire, verbose=verbose)
    return model, n_scores

from imblearn.over_sampling import RandomOverSampler
def over_dataset_with_RandomOverSampler(X_train,y_train,data_set_path, expected_val, random_state=42, target = 'fraud_flag', verbose=0):
    short_name = "over_RandomOverSampler"
    over_file_name = "train_complete_OVER_RandomOverSampler_"
    res_path = data_set_path.replace("train_complete_", over_file_name)
    dataset_over = None
    if exists(res_path):
        dataset_over = pd.read_csv(res_path) 
        if verbose > 0: info(short_name, f"{res_path} {surligne_text('LOAD')}")
    else:
        # Choix de la taille du nouveau dataset 
        distribution_of_samples = {0:expected_val, 1:expected_val}
        # Sur-Echantillonnage en utilisant la méthode SMOTE
        smote = RandomOverSampler(sampling_strategy = distribution_of_samples, random_state = random_state)
        # X_over_sample, y_over_sample = smote.fit_resample(X,y)
        dataset_over, y_train_over = smote.fit_resample(X_train,y_train[target])
        if verbose > 0: info(short_name, f"{dataset_over.shape}, {y_train_over.shape}")
        if verbose > 1: debug(short_name, f"{y_train_over.value_counts()}")

        dataset_over[target] = y_train_over
        dataset_over.to_csv(res_path)
        if verbose > 0: info(short_name, f"{res_path} {surligne_text('SAVE')}")
    return dataset_over
    
from imblearn.over_sampling import SMOTE

def over_dataset_with_SMOTE(X_train,y_train,data_set_path, sampling_strategy='minority', random_state=42, target = 'fraud_flag', verbose=0):
    short_name = "over_SMOTE"
    over_file_name = "train_complete_OVER_SMOTE_"+sampling_strategy+"_"
    res_path = data_set_path.replace("train_complete_", over_file_name)
    dataset_over = None
    if exists(res_path):
        dataset_over = pd.read_csv(res_path) 
        if verbose > 0: info(short_name, f"{res_path} {surligne_text('LOAD')}")
    else:
        # Sur-Echantillonnage en utilisant la méthode SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5)
        dataset_over, y_train_over = smote.fit_resample(X_train,y_train[target])
        if verbose > 0: info(short_name, f"{dataset_over.shape}, {y_train_over.shape}")
        if verbose > 1: debug(short_name, f"{y_train_over.value_counts()}")

        dataset_over[target] = y_train_over
        dataset_over.to_csv(res_path)
        if verbose > 0: info(short_name, f"{res_path} {surligne_text('SAVE')}")
    return dataset_over

from imblearn.over_sampling import BorderlineSMOTE

def over_dataset_with_BorderlineSMOTE(X_train,y_train,data_set_path, target = 'fraud_flag', verbose=0):
    short_name = "over_BorderlineSMOTE"
    over_file_name = "train_complete_OVER_BorderlineSMOTE_"
    res_path = data_set_path.replace("train_complete_", over_file_name)
    dataset_over = None
    if exists(res_path):
        dataset_over = pd.read_csv(res_path) 
        if verbose > 0: info(short_name, f"{res_path} {surligne_text('LOAD')}")
    else:
        # Sur-Echantillonnage en utilisant la méthode SMOTE
        smote = BorderlineSMOTE()
        dataset_over, y_train_over = smote.fit_resample(X_train,y_train[target])
        if verbose > 0: info(short_name, f"{dataset_over.shape}, {y_train_over.shape}")
        if verbose > 1: debug(short_name, f"{y_train_over.value_counts()}")

        dataset_over[target] = y_train_over
        dataset_over.to_csv(res_path)
        if verbose > 0: info(short_name, f"{res_path} {surligne_text('SAVE')}")
    return dataset_over
# ----------------------------------------------------------------------------------
#                        SCORE
# ----------------------------------------------------------------------------------
def save_score(scores, score_path):
    index_label ="date"
    if index_label in list(scores.columns):
        index_label ="index"
    scores.to_csv(score_path, sep='|', index_label =index_label)

def load_scores(score_path, save_it=False, verbose=0):
    scores = pd.read_csv(score_path, sep='|', index_col="date")
    # scores = scores[['date', 'Modèle', 'Features', 'Add Data', 'Accuracy Score',  'pr_auc_score TEST perso', 'pr_auc_score TEST officiel', 'Commentaire', 'Params']]
    for col in ['Accuracy Score',  'pr_auc_score TEST perso', 'pr_auc_score TEST officiel']:
        try:
            scores.loc[scores[col]=="", col] = np.nan
        except Exception as err:
            if verbose>1: print(err)
        try:
            scores[col] = scores[col].fillna(0)
        except Exception as err:
            if verbose>1: print(err)
        try:
            scores[col] = scores[col].astype(float)
        except Exception as err:
            if verbose>1: print(err)
    if save_it:
        save_score(scores=scores, score_path=score_path)
    return scores

def add_score(scores_param, modele, features, add_data, params, accuracy_score=0,	pr_auc_score_TEST_perso=0,	pr_auc_score_TEST_officiel=0, commentaire=np.nan, score_path=None,verbose=0):
    short_name = "add_score"
    data_dict = {
        'date' : [datetime.now().strftime('%Y-%m-%d %H:%M')],
        'Modèle' : [modele],
        'Features' : [features], 
        'Add Data' : [add_data],
        'Accuracy Score':[accuracy_score],
        'pr_auc_score TEST perso':[pr_auc_score_TEST_perso],
        'pr_auc_score TEST officiel':[pr_auc_score_TEST_officiel], 
        'Params':[params.replace('"', "'") if isinstance(params, str) else params],
        'Commentaire':[commentaire.replace('"', "'") if isinstance(commentaire, str) else commentaire],
    }
    if verbose>1:
        debug(short_name, data_dict)
    to_add = pd.DataFrame.from_dict(data_dict)
    scores = None
    if scores_param is None:
        scores = to_add
    else:
        scores = pd.concat([scores_param.reset_index(), to_add])
        scores = scores.set_index('date')
    if score_path is not None and len(score_path)>0:
        save_score(scores=scores, score_path=score_path)
    return scores
# ----------------------------------------------------------------------------------
#                        GRAPHIQUES
# ----------------------------------------------------------------------------------
PLOT_FIGURE_BAGROUNG_COLOR = 'white'
PLOT_BAGROUNG_COLOR = PLOT_FIGURE_BAGROUNG_COLOR


def color_graph_background(ligne=1, colonne=1):
    figure, axes = plt.subplots(ligne,colonne)
    figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
    if isinstance(axes, np.ndarray):
        for axe in axes:
            # Traitement des figures avec plusieurs lignes
            if isinstance(axe, np.ndarray):
                for ae in axe:
                    ae.set_facecolor(PLOT_BAGROUNG_COLOR)
            else:
                axe.set_facecolor(PLOT_BAGROUNG_COLOR)
    else:
        axes.set_facecolor(PLOT_BAGROUNG_COLOR)
    return figure, axes

def draw_correlation_graphe(df, title, annot=True, size=(18, 15), minimum_corr=False, limit_na=False, fontsize=5, verbose=0):
    """Dessine le graphe de corrélation des données

    Args:
        df (DataFrame): Données à représenter
        title (str) :
        annot (bool, optional)
        size (tuple(int, int))
        minimum_corr
        limit_na
        fontsize
        verbose (int, optional): Mode debug. Defaults to False.
    """
    short_name = "draw_correlation_graphe"
    if verbose>0: print(f"[{short_name}]\tINFO : ...... START")
    corr_df = round(df.corr(), 2)

    if minimum_corr != False:

        corr_df_not_0 = corr_df[((corr_df>minimum_corr)|(corr_df<(-minimum_corr)))]
        # Nous ne devons pas toucher aux colonnes ci-dessous
        for col in ['Nb_of_items', 'fraud_flag']:
            corr_df_not_0[col] = corr_df[col]
            corr_df_not_0.loc[col] = corr_df.loc[col] 
            
        corr_df_not_0["NB_NAN"] = corr_df_not_0.isna().sum(axis=0)

        corr_df_not_0_clean = corr_df_not_0.copy()

        if limit_na == False:

            if verbose>0:
                limit_na = max(corr_df_not_0["NB_NAN"])
                print(f"[{short_name}]\tINFO : {limit_na} max NA")
            
            dvc = corr_df_not_0["NB_NAN"].value_counts()
            limit_na = min(dvc[dvc>100].index)
            if verbose>0:
                print(f"[{short_name}]\tINFO : {limit_na} min index de NB_NAN > 100")

        idxs = corr_df_not_0[corr_df_not_0['NB_NAN']>limit_na-1].index
        if verbose>0:
            print(f"[{short_name}]\tINFO : {corr_df_not_0_clean.shape}")
        corr_df_not_0_clean = corr_df_not_0_clean.drop(columns=list(idxs))
        corr_df_not_0_clean = corr_df_not_0_clean[corr_df_not_0_clean['NB_NAN']<limit_na]
        corr_df_not_0_clean = corr_df_not_0_clean.drop(columns=['NB_NAN'])
        if verbose>0:
            print(f"[{short_name}]\tINFO : {corr_df_not_0_clean.shape}")
        corr_df = corr_df_not_0_clean

    if verbose>0:
        print(f"[{short_name}]\tINFO : Confusion matrix created")
        if verbose>1:
            print(f"[{short_name}]\tDEBUG : CORR ------------------")
            print(f"[{short_name}]\tDEBUG : {corr_df}")

    figure, ax = color_graph_background(1,1)
    figure.set_size_inches(size[0], size[1], forward=True)
    figure.set_dpi(100)
    figure.suptitle(title + f" (minimum_corr={minimum_corr}, limit_na={limit_na})", fontsize=16)
    sns.heatmap(corr_df, annot=annot, annot_kws={"fontsize":fontsize})
    plt.xticks(rotation=45, ha="right", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()
    return corr_df


def draw_graph_multiple(graph_function, df, column_names, title="Répartition",index_name='cat_name', size=(20, 15), verbose=0):
    """Fonction pour dessiner un graphe pie pour la colonne reçue

    Args:
        df (DataFrame): Données à représenter
        column_name (String): colonne à représenter
        country_name_list (list[String], optional): Liste des pays à traiter. Defaults to [].
        verbose (bool, optional): Mode debug. Defaults to False.
    """
    figure, axes = plt.subplots(1,len(column_names))
    i = 0
    for column_name in column_names:
        if len(column_names) > 1:
            graph_function(df, column_name, axes[i],index_name=index_name, verbose=verbose)
        else:
            graph_function(df, column_name, axes,index_name=index_name, verbose=verbose)
        i += 1
    figure.set_size_inches(size[0], size[1], forward=True)
    figure.set_dpi(100)
    figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
    figure.suptitle(f"{title} : "+column_name, fontsize=16)
    plt.grid()
    plt.show()
    print("draw_graph_multiple", column_name," ................................................. END")


def draw_barh(df_param, column_name,  axe, index_name='cat_name',verbose=0):
    
    df = df_param.copy()
    df = df.sort_values(column_name, ascending=False)
        
    axe.barh(y=df[index_name], width=df[column_name])
    # axe.barh(y=df[index_name], width=df[column_name], labels=df[index_name], autopct='%.0f%%')
    axe.set_facecolor(PLOT_BAGROUNG_COLOR)
    axe.set_title(column_name)
    axe.set_facecolor(PLOT_BAGROUNG_COLOR)

def draw_barh_multiple(df, column_names, title="Répartition",index_name='cat_name', size=(20, 15), verbose=0):
    draw_graph_multiple(graph_function=draw_barh, df=df, column_names=column_names, title=title, index_name=index_name, size=size, verbose=verbose)


def draw_pie(df, column_name,  axe, index_name='cat_name',verbose=0):
    """Fonction pour dessiner un graphe pie pour la colonne reçue
    Args:
        df (DataFrame): Données à représenter
        column_name (String): colonne à représenter
        country_name_list (list[String], optional): Liste des pays à traiter. Defaults to [].
        verbose (bool, optional): Mode debug. Defaults to False.
    """
    
    # Affichage des graphiques
    axe.pie(df[column_name], labels=df[index_name], autopct='%.0f%%')
    axe.legend(df[index_name], loc='best')
    axe.set_title(column_name)
    axe.set_facecolor(PLOT_BAGROUNG_COLOR)


def draw_pie_multiple(df, column_names, title="Répartition",index_name='cat_name', size=(20, 15), verbose=0):
    """Fonction pour dessiner un graphe pie pour la colonne reçue

    Args:
        df (DataFrame): Données à représenter
        column_name (String): colonne à représenter
        country_name_list (list[String], optional): Liste des pays à traiter. Defaults to [].
        verbose (bool, optional): Mode debug. Defaults to False.
    """
    draw_graph_multiple(graph_function=draw_pie, df=df, column_names=column_names, title=title, index_name=index_name, size=size, verbose=verbose)

def plot_scatters(Xs, ys, titles, target="fraud_flag", size=(20, 10), verbose=0):
    labels = ['OK', 'FRAUD']
    figure, axes = color_graph_background(ligne=len(Xs), colonne=1)

    # scatter plot of examples by class label
    i = 0
    for X, y, title in zip(Xs, ys, titles):
        for label in y[target].unique():
            axes[i].scatter(X.loc[y[target]==label, 'amount'], X.loc[y[target]==label, 'Nb_of_items'], label=labels[label])
        axes[i].set_title(title)
        i += 1
    
    figure.set_size_inches(size[0], size[1], forward=True)
    figure.set_dpi(100)
    figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
    figure.suptitle(f"{target} repartition", fontsize=16)
    

def plot_scatter(X, y, target="fraud_flag", figsize=(20, 10), verbose=0):
    labels = ['OK', 'FRAUD']
    # scatter plot of examples by class label
    plt.figure(figsize=figsize, dpi=100)
    for label in y[target].unique():
        plt.scatter(X.loc[y[target]==label, 'amount'], X.loc[y[target]==label, 'Nb_of_items'], label=labels[label])
    plt.legend()
    plt.show()    

# ----------------------------------------------------------------------------------
#                        PRINT
# ----------------------------------------------------------------------------------
from termcolor import colored

COLORS = {
    "default" : (6,42,30),
    "green" : (6,42,30),
    "blue" : (13,36,81),
    "red" : (74,40,34),
    "yellow" : (87,82,9),
}

COLOR_MOD = {
    'DEBUG' : 'blue',
    'WARN'  : 'yellow',
    'ERROR' : 'red',
}

def display(function_name, text, now=None, duration=None, key='INFO', color=None, log_end=""):
    log = f"[{function_name:<20}]\t{key.upper():<5} {text}"
    
    log = f"{log} {log_end:>25}"
    
    if now is not None and duration is not None:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        end_ = f"\t\t END {now} ---> in {duration}"
        log = log + f"{end_:>60}"
    elif now is not None:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        end_ = f"\t\t START {now}"
        log = log + f"{end_:>38}" 
            
    if color is not None:
        log = colored(log, color)
        
    print(log)
    return log

# %% colored
def info(function_name, text, now=None, duration=None, log_end=""):
    key='INFO'
    return display(function_name=function_name, text=text, now=now, duration=duration, key=key, color= COLOR_MOD.get(key, None), log_end=log_end)

def debug(function_name, text, now=None, duration=None, log_end=""):
    key='DEBUG'
    return display(function_name=function_name, text=text, now=now, duration=duration, key=key, color= COLOR_MOD.get(key, None), log_end=log_end)

def warn(function_name, text, now=None, duration=None, log_end=""):
    key='WARN'
    return display(function_name=function_name, text=text, now=now, duration=duration, key=key, color= COLOR_MOD.get(key, None), log_end=log_end)
    
def error(function_name, text, now=None, duration=None, log_end=""):
    key='ERROR'
    return display(function_name=function_name, text=text, now=now, duration=duration, key=key, color= COLOR_MOD.get(key, None), log_end=log_end)

def surligne_text(text, color="green"):
    r = COLORS.get(color,COLORS["default"])[0] 
    g = COLORS.get(color,COLORS["default"])[1] 
    b = COLORS.get(color,COLORS["default"])[2] 
    return f'\x1b[{r};{b};{g}m{text}\x1b[0m'

