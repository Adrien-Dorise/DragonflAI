import numpy as np
import matplotlib.pyplot as plt 
from dragonflai.utils.utils_path import create_file_path

def create_set(n, list_center, seed=948401971):
    """Create a set

    Args:
        n (int): Number of samples in the set
        list_center (list): Center of the set
        seed (int): seed used for random creation of the blob sets. Default to 948401971.

    Returns:
        list[list]: Return the set with the coordinates of each samples
    """
  
    assert isinstance(list_center, list), 'Error when list_std is not a list'
    np.random.seed(seed)
    ret = np.random.rand(n, len(list_center))
    for i in range(len(list_center)):
        ret[:,i] += list_center[i] - 0.5

    return ret


def create_dataset(list_N, list_center, seed=948401971): 
    """Create a blob data set

    Args:
        list_N (list): list containing the number of samples in each blob
        list_center (list): Center of each blob
        seed (int): seed used for random creation of the blob sets. Default to 948401971. 

    Returns:
        np.array: List containing the features for each sample
        np.arrays: List containing the classes for each sample
    """
    features = []
    for c in range(len(list_N)):
        s = create_set(list_N[c], list_center[c], seed=seed)
        for data in s:
            features.append(data)
    targets = []
    for indice in range(len(list_N)):
        targets.extend([indice]*list_N[indice])
    targets = np.array(targets)
    return np.array(features), np.array(targets)


def plot_dataset(dataset,cluster,save_path):
    colors= ['b','r','g','k'] #classe 1 en blue, 2 en red...
    markers= ['o','^'] # pour différencier les point du centroide: rond et chapeur pour les centroides
    fig= plt.figure()
    plt.grid(True)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.title("Blob dataset visualisation")
    ax=fig.gca()
    for i in range(len(dataset)):#parcours données du dataset
        ax.scatter(dataset[i][0],dataset[i][1],color=colors[cluster[i]], marker=markers[0],alpha=0.5)
    
    create_file_path(save_path)
    fig.savefig('{}.png'.format(save_path))
    plt.close()