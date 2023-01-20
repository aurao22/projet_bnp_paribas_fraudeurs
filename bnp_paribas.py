
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os import listdir
from os.path import isfile, join, exists, getsize
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

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

def transpose_categories(df, verbose=0):
    df_temp = df.loc["TOTAL"]
    type(df_temp)
    dd = pd.DataFrame(data=df_temp)
    dd = dd.sort_values(by=['TOTAL'], ascending=False)
    dict_temp = defaultdict(list)
    
    ever_proceed = {'ID', "Nb", 'fraud'}

    for idx in dd.index:
        cat_name = idx.split("_")[0]
        if cat_name not in ever_proceed:
            dict_temp['cat_name'].append(cat_name)
            ever_proceed.add(cat_name)
            cash_val = dd.loc[cat_name+"_cash", 'TOTAL']
            nb_val = dd.loc[cat_name+"_nb", 'TOTAL']
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
    corr_df = round(df.corr(), 2)

    if minimum_corr != False:
        corr_df_not_0 = corr_df[(corr_df>minimum_corr)|(corr_df<(-minimum_corr))]
        corr_df_not_0["NB_NAN"] = corr_df_not_0.isna().sum(axis=0)
        corr_df_not_0_clean = corr_df_not_0.copy()

        if limit_na == False:
            limit_na = max(corr_df_not_0["NB_NAN"])
            if verbose>0:
                print(f"[{short_name}]\tDEBUG : {limit_na} max NA")

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
        print(f"[{short_name}]\tINFO : CORR ------------------")
        print(f"[{short_name}]\tINFO : {corr_df}")

    figure, ax = color_graph_background(1,1)
    figure.set_size_inches(size[0], size[1], forward=True)
    figure.set_dpi(100)
    figure.suptitle(title, fontsize=16)
    sns.heatmap(corr_df, annot=annot, annot_kws={"fontsize":fontsize})
    plt.xticks(rotation=45, ha="right", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()
    return corr_df