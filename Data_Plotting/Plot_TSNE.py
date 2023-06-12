from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


# Latent Feature Cluster for Training Data using T-SNE
def TSNE_reduction(latent_points: np.ndarray, perplexity=30, learning_rate=20):
    """
    :param latent_points: [ndarray] - an array of arrays that define the points of multiple objects in the latent space
    :param perplexity: [int] - default perplexity = 30 " Perplexity balances the attention t-SNE gives to local and global aspects of the data. It is roughly a guess of the number of close neighbors each point has... a denser dataset ... requires higher perplexity value" Recommended: Perplexity(5-50)
    :param learning_rate: [int] - default learning rate = 200 "If the learning rate is too high, the data may look like a ‘ball’ with any point approximately equidistant from its nearest neighbours. If the learning rate is too low, most points may look compressed in a dense cloud with few outliers." Recommended: learning_rate(10-1000)
    :return: [tuple] - the output is the x and y coordinates for the reduced latent space, a title, and a TSNE embedding
    """
    model = TSNE(n_components=2, random_state=0, perplexity=perplexity,
                 learning_rate=learning_rate)
    # the number of components = dimension of the embedded space

    embedding = model

    tsne_data = model.fit_transform(latent_points)
    # When there are more data points, only use a couple of hundred points so TSNE doesn't take too long
    x = tsne_data[:, 0]
    y = tsne_data[:, 1]
    title = ("T-SNE of Data")
    return x, y, title, embedding


def plot_dimensionality_reduction(x: list, y: list, label_set: list, title: str):
    """
    :param x: [list] - the first set of coordinates for each latent point
    :param y: [list] - the second set of coordinates for each latent point
    :param label_set: [list] - a set of values that define the color of each point based on an additional quantitative attribute.
    :return: matplotlib figure - the output is a matplotlib figure that displays all the points in a 2-dimensional latent space, based on the labels provided.
    """
    plt.title(title)
    # Color points based on a continuous label
    if label_set[0].dtype == float:
        plt.scatter(x, y, c=label_set)
        cbar = plt.colorbar()
        cbar.set_label('Average Density', fontsize=12)
        print("using scatter")

    # Color points based on a discrete label
    else:
        for label in set(label_set):
            cond = np.where(np.array(label_set) == str(label))
            plt.plot(x[cond], y[cond], marker='o', linestyle='none', label=label)

        plt.legend(numpoints=1)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
########################################################################################################################
"""
# Use for personal plotting

import pandas as pd
import json

df = pd.read_csv('2D_Lattice.csv')
# row = 0
# box = df.iloc[row,1]
# array = np.array(json.loads(box))

# Select a subset of the data to use
number_samples = 10000
perplexity = 300

random_samples = sorted(np.random.randint(0,len(df), number_samples))  # Generates ordered samples

df = df.iloc[random_samples]

print(df)
print(np.shape(df))


# For plotting CSV data
# define a function to flatten a box
def flatten_box(box_str):
    box = json.loads(box_str)
    return np.array(box).flatten()


# apply the flatten_box function to each row of the dataframe and create a list of flattened arrays
flattened_arrays = df['Array'].apply(flatten_box).tolist()
avg_density = np.sum(flattened_arrays, axis=1)/(len(flattened_arrays[0]))

x, y, title, embedding = TSNE_reduction(flattened_arrays, perplexity=perplexity)
plot_dimensionality_reduction(x, y, avg_density, title)
plt.title(title)
plt.savefig('TSNE_Partial_Factorial_Perplexity_' + str(perplexity) + "_Data_Samples_" + str(number_samples))

"""

