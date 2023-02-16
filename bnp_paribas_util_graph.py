
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    ligne=len(Xs)
    figure, axes = color_graph_background(ligne=ligne, colonne=1)
    size = (size[0], ligne * 5 + 3)

    if ligne==1:
        axes = [axes]
        
    # scatter plot of examples by class label
    i = 0
    for X, y, title in zip(Xs, ys, titles):
        if y is None:
            axes[i].scatter(X['amount'], X['Nb_of_items'])
        else:
            for label in y[target].unique():
                axes[i].scatter(X.loc[y[target]==label, 'amount'], X.loc[y[target]==label, 'Nb_of_items'], label=labels[label])
            axes[i].legend()
        axes[i].set_title(title)
        i += 1
    
    figure.set_size_inches(size[0], size[1], forward=True)
    figure.set_dpi(100)
    figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
    figure.suptitle(f"{target} repartition", fontsize=16)


