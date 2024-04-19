import process_data
import matplotlib.pyplot as plt
import numpy as np

X = process_data.X
X = X.T

for i in range(X.shape[0]):
    plt.figure(figsize=(16, 16))
    if i == 0 or i == 2 or i == 3 or i == 4 or i == 10 or i == 15 or i == 16 or i == 17 or i == 18:
        continue
    unique_values, counts = np.unique(X[i], return_counts=True)
    plt.bar(unique_values, counts)
    plt.xlabel("Categories", fontsize=20)
    plt.ylabel("count", fontsize=20)
    plt.title(process_data.data_headers[i], fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if i == 21:
        plt.xticks(rotation=90)
    if i == 22:
        plt.xticks(rotation=50)
    plt.show()
