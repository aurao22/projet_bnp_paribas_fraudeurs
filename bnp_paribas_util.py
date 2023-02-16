
import pandas as pd
from os.path import join, exists, getsize, basename, dirname
from tqdm import tqdm
from datetime import datetime

from bnp_paribas_Model_OverData import OVERED_DATASET_FILE_NAME, load_over_dataset
from bnp_paribas_preprocessing import reduce_data_by_typing, add_amounts
from pr_auc_score_SB05ixL import complete_y_cols


# ----------------------------------------------------------------------------------
#                        BACKUP
# ----------------------------------------------------------------------------------
def load_dump_file(data_set_path, file_name, force_reloading=0, short_name = "load_dump_file", verbose=0):   
    save_file_path = data_set_path

    if save_file_path is not None:
        if file_name is not None and len(file_name)>0:
            save_file_path = join(save_file_path, file_name)
        else:
            file_name = basename(save_file_path)
        
    df_res = None
    if exists(save_file_path) and getsize(save_file_path) > 0 and not force_reloading:
        # Chargement de la DF fichier
        if verbose > 1: debug(short_name, f"{file_name} exist")
        df_res = pd.read_csv(save_file_path, sep=",", index_col="index",low_memory=False)
        if verbose > 0: info(short_name, f"{file_name} {surligne_text('LOAD')}")
    return df_res

def dump_file(data_set_path,  df,file_name=None, short_name = "dump_file", verbose=0):   
    save_file_path = data_set_path

    if save_file_path is not None:
        if file_name is not None and len(file_name)>0:
            save_file_path = join(save_file_path, file_name)
        else:
            file_name = basename(save_file_path)
        # Sauvegarde du fichier
        df.to_csv(save_file_path)

        if verbose > 0: print(f"[{short_name}]\tINFO: {file_name} => {surligne_text('SAVED')}")
    elif verbose>0:
        warn(short_name, 'No path to dump')
    return save_file_path


from sklearn.model_selection import train_test_split
from os.path import exists

_DATASET_KEYS = ["X_train", "X_test", "y_train", "y_test"]

def load_splited_data(dataset_path, over_name=None, test_size = 0.2, random_state = 42, save_it=True, force=False, target = 'fraud_flag', verbose=0):
    short_name="load_splited_data"
    if verbose > 0: info(short_name, surligne_text('IN PROGRESS...'))
    dataset_dict = {}
    if not force:
        for set_name in _DATASET_KEYS:
            if set_name not in list(dataset_dict.keys()):
                if over_name is not None and len(over_name)>0 and set_name == "X_train":
                    if verbose > 0: info(short_name, f"over {over_name} Loading {set_name}...")
                    path = dirname(dataset_path)
                    over_path = join(path,OVERED_DATASET_FILE_NAME.get(over_name))
                    dataset_over = load_over_dataset(path=over_path, verbose=verbose)
                    dataset_dict["y_train"] = dataset_over[['ID',target]]
                    dataset_dict[set_name] = dataset_over.drop([target], axis=1)
                    continue
                
                file_name = dataset_path.replace(".csv", f"_split_{set_name}.csv")
                if dataset_dict.get(set_name, None) is None and exists(file_name):
                    if verbose > 0: info(short_name, f"set {set_name} Loading {set_name}...")
                    dataset_dict[set_name] = pd.read_csv(file_name, sep=',', low_memory=False)
                    if set_name.startswith("X_"):
                        dataset_dict[set_name] = add_amounts(dataset_dict[set_name], verbose=verbose)
                        dataset_dict[set_name] = reduce_data_by_typing(df=dataset_dict[set_name], verbose=verbose)
        
    # si les fichiers split n'existent pas :
    if force or len(dataset_dict) < 4:       
        
        if verbose > 0: info(short_name, f"Chargement des données train sources...")
        train_origin = pd.read_csv(dataset_path, sep=',', index_col="index" ,low_memory=False)
        dataset_dict['train_origin'] = train_origin
        if verbose > 0: info(short_name,f"Train d'originr {train_origin.shape} {surligne_text('LOAD')}")

        if verbose > 0: info(short_name, f"Séparation de X et y...")      
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
            dataset = reduce_data_by_typing(df=dataset, verbose=verbose)
            dataset_dict[dataset_name] = dataset
            if save_it and dataset is not None:
                dataset.to_csv(dataset_path.replace(".csv", f"_split_{dataset_name}.csv"))
                    
    if verbose > 0:
        info(short_name, f"{len(dataset_dict)} dataset {surligne_text('LOAD')}")
        if len(dataset_dict)>0: info(short_name, f"{dataset_dict.get(_DATASET_KEYS[0]).shape} / {dataset_dict.get(_DATASET_KEYS[1]).shape}")
    return dataset_dict

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

