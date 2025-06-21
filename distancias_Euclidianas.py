import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

def Eu_dist_plot(X,Y):
    dist_matrix = distance_matrix(X, Y)
    plt.figure(figsize=(8, 8))
    plt.imshow(dist_matrix, cmap='gray', interpolation='nearest')
    plt.colorbar(label='Euclidean Distance')
    for i in range(len(X)):
        for j in range(len(Y)):
            plt.text(j, i, f'{dist_matrix[i, j]:.2f}', ha='center', va='center', color='black')
    
    labels = [chr(65 + i) for i in range(max(len(X), len(Y)))]
    
    plt.xticks(ticks=np.arange(len(Y)), labels=labels[:len(Y)])#labels=[f'C{i+1}' for i in range(len(Y))])
    plt.yticks(ticks=np.arange(len(X)), labels=labels[:len(X)])
    plt.title('Euclidean Distance Matrix')
    plt.xlabel('Points')
    plt.ylabel('Points')
    plt.show()

X = np.array([
    [2, 10],
    [2, 5],
    [8, 4],
    [5, 8],
    [7, 5],
    [6, 4],
    [1, 2],
    [4, 9]
])
Eu_dist_plot(X,X)
cl = int(input("clusters: "))

while True:
    new_cluster = np.empty((0, 2), float)
    for i in range(cl):
        var1 = input("Enter x coordinate or 'exit' to finish: ")
        if var1.lower() == "exit":
            break
        var2 = input("Enter y coordinate or 'exit' to finish: ")
        if var2.lower() == "exit":
            break
        new_cluster = np.vstack([new_cluster, [float(var1), float(var2)]])
    if var1.lower() == "exit" or var2.lower() == "exit":
        break
    Eu_dist_plot(X,new_cluster)
