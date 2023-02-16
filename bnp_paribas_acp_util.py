import numpy as np
from time import time
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from matplotlib.collections import LineCollection

nutriscore_colors = {"A":"green", "B":"#AACE73","C":"yellow","D":"orange","E":"red", 
                     "1":"green", "2":"#AACE73","3":"yellow","4":"orange","5":"red",
                     1:"green", 2:"#AACE73",3:"yellow",4:"orange",5:"red",
                     "1.0":"green", "2.0":"yellow","3.0":"orange","4.0":"red", 
                     1.0:"green", 2.0:"yellow",3.0:"orange",4.0:"red", np.nan:"blue"}

PLOT_FIGURE_BAGROUNG_COLOR = 'white'
PLOT_BAGROUNG_COLOR = PLOT_FIGURE_BAGROUNG_COLOR

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.figure(figsize=(18,7), facecolor=PLOT_FIGURE_BAGROUNG_COLOR)
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)


def plot_dendrogram(Z, names):
    plt.figure(figsize=(18,7), facecolor=PLOT_FIGURE_BAGROUNG_COLOR)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()

        

def display_factorial_planes_by_theme(X_projected,pca, n_comp, axis_ranks, alpha=0.5, illustrative_var=None, by_theme=False):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
            # affichage des points
            illustrative_var = np.array(illustrative_var)
            valil = np.unique(illustrative_var)

            figure, axes = plt.subplots(2,len(valil)//2)

            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            
            # On commence par traiter le NAN pour plus de lisibilité dans le graphe
            value = str(np.nan)
            i = 0
            j = 0
            if value in valil :
                _display_one_scatter(X_projected, pca, axes[i][j], value, d1, d2, alpha,boundary, illustrative_var)
                valil = valil[valil != value]
                j += 1
            
            for value in valil:
                _display_one_scatter(X_projected, pca, axes[i][j], value, d1, d2, alpha,boundary, illustrative_var)
                
                j += 1
                if j > (len(valil)//2):
                    i += 1
                    j = 0
            
            figure.set_size_inches(18.5, 7, forward=True)
            figure.set_dpi(100)
            figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
            figure.suptitle("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(10,10))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
                ax.set_facecolor(PLOT_BAGROUNG_COLOR)
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            fig.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)


def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=0.5, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            plt.figure(figsize=(18,15), facecolor=PLOT_FIGURE_BAGROUNG_COLOR)
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                valil = np.unique(illustrative_var)
                # On commence par traiter le NAN pour plus de lisibilité dans le graphe
                value = str(np.nan)
                if value in valil :
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, c=nutriscore_colors.get(value, "blue"), s=100)
                    valil = valil[valil != value]
                for value in valil:
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, c=nutriscore_colors.get(value, "blue"), s=100)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)



def _display_one_scatter(X_projected, pca, axe,value, d1, d2, alpha, boundary, illustrative_var):
    selected = np.where(illustrative_var == value)
    c=nutriscore_colors.get(value, "blue")
    axe.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, c=c, s=100)
    axe.legend()
    # nom des axes, avec le pourcentage d'inertie expliqué
    axe.set_xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
    axe.set_ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

    axe.set_xlim([-boundary,boundary])
    axe.set_ylim([-boundary,boundary])
    # affichage des lignes horizontales et verticales
    axe.plot([-100, 100], [0, 0], color='grey', ls='--')
    axe.plot([0, 0], [-100, 100], color='grey', ls='--')
    axe.set_facecolor(PLOT_BAGROUNG_COLOR)
    
