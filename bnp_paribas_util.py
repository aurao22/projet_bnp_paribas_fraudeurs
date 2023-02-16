
import pandas as pd
from os.path import join, exists, getsize, basename
from tqdm import tqdm
from datetime import datetime


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

