
import numpy as np
import pandas as pd
from os.path import join, exists
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

from bnp_paribas_util import *
from bnp_paribas_preprocessing import reduce_data_by_typing

# ----------------------------------------------------------------------------------
#                        OVER DATAS FUNCTION
# ----------------------------------------------------------------------------------

start_ = "train_complete_encoded_hyper_light_2023_01_19_split_dataset_train_OVER_"

OVERED_DATASET_FILE_NAME = {
    "BorderlineSMOTE"            : start_+"BorderlineSMOTE.csv",
    "RandomOverSampler"          : start_+"RandomOverSampler.csv",
    "RandomOverSampler amount less" : start_+"RandomOverSampler1.csv",
    "SMOTE_minority"             : start_+"SMOTE_minority.csv",
    "SMOTE_minority amount less" : start_+"SMOTE_minority1.csv",
    "SVMSMOTE"                   : start_+"SVMSMOTE.csv",
}

def load_over_dataset(path, verbose=0):
    short_name = "load_over_dataset"
    dataset_over = None
    # Chargement des données
    save_path = path.replace(".csv", "_rounded.csv")
    if not exists(save_path):
        dataset_over = pd.read_csv(path, index_col="index")
        if verbose>0: info(short_name, f"Reduce datas {surligne_text('IN PROGRESS')}")
        dataset_over = reduce_data_by_typing(df=dataset_over, verbose=verbose)
        try:
            to_drop = "Unnamed: 0"
            if to_drop in list(dataset_over.columns):
                if verbose>0: info(short_name, f"{to_drop} column to drop.")
                dataset_over = dataset_over.drop(labels=[to_drop], axis=1)
            else:
                if verbose>1: debug(short_name, f"{to_drop} not in dataset.")
        except:
            pass
        dataset_over.to_csv(save_path, index_label="index")
        if verbose>0: info(short_name, f"{dataset_over.shape} datas keep {surligne_text('SAVE')}")
    else:
        dataset_over = pd.read_csv(save_path, index_col="index")
        # TODO voir s'il faut refaire le reduce
    if verbose>0: info(short_name, f"{dataset_over.shape} datas {surligne_text('LOAD')}")
    return dataset_over


from imblearn.over_sampling import RandomOverSampler
def over_dataset_with_RandomOverSampler(X_train,y_train,data_set_path, expected_val, random_state=42, target = 'fraud_flag', verbose=0):
    over_file_name = "train_complete_OVER_RandomOverSampler_"
        
    # Choix de la taille du nouveau dataset 
    distribution_of_samples = {0:expected_val, 1:expected_val}
    smote = RandomOverSampler(sampling_strategy = distribution_of_samples, random_state = random_state)
    return _over(over_model=smote, over_file_name=over_file_name, 
                X_train=X_train,y_train=y_train,data_set_path=data_set_path, 
                target =target, verbose=verbose)
    
from imblearn.over_sampling import SMOTE
def over_dataset_with_SMOTE(X_train,y_train,data_set_path, sampling_strategy='minority', random_state=42, target = 'fraud_flag', verbose=0):
    over_file_name = "train_complete_OVER_SMOTE_"+sampling_strategy+"_"
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5)
    return _over(over_model=smote, over_file_name=over_file_name, 
                X_train=X_train,y_train=y_train,data_set_path=data_set_path, 
                target =target, verbose=verbose)

from imblearn.over_sampling import BorderlineSMOTE
def over_dataset_with_BorderlineSMOTE(X_train,y_train,data_set_path, target = 'fraud_flag', verbose=0):
    over_file_name = "train_complete_OVER_BorderlineSMOTE_"    
    return _over(over_model=BorderlineSMOTE(), over_file_name=over_file_name, 
                X_train=X_train,y_train=y_train,data_set_path=data_set_path, 
                target =target, verbose=verbose)

from imblearn.over_sampling import SVMSMOTE
def over_dataset_with_SVMSMOTE(X_train,y_train,data_set_path, target = 'fraud_flag', verbose=0):
    over_file_name = "train_complete_OVER_SVMSMOTE_"
    return _over(over_model=SVMSMOTE(), over_file_name=over_file_name, 
                X_train=X_train,y_train=y_train,data_set_path=data_set_path, 
                target =target, verbose=verbose)

def _over(over_model, over_file_name, X_train,y_train,data_set_path, target = 'fraud_flag', verbose=0):
    short_name = over_file_name.replace("train_complete_", "")
    short_name = short_name[:-1]
    res_path = data_set_path.replace("train_complete_", over_file_name)
    dataset_over = None
    if exists(res_path):
        dataset_over = load_over_dataset(res_path, verbose=verbose)
    else:
        # Sur-Echantillonnage en utilisant la méthode SMOTE
        dataset_over, y_train_over = over_model.fit_resample(X_train,y_train[target])
        if verbose > 0: info(short_name, f"{dataset_over.shape}, {y_train_over.shape}")
        if verbose > 1: debug(short_name, f"{y_train_over.value_counts()}")

        dataset_over[target] = y_train_over
        dataset_over.to_csv(res_path, index_label="index")
        if verbose > 0: info(short_name, f"{res_path} {surligne_text('SAVE')}")
    return dataset_over


