import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sns.set_style('whitegrid')



def visualize(graph):
    flows = pd.read_csv('all_flows.txt')
    packets = flows['bytes']

    for i in range(800000):
        if packets[i] >= 20000:
            packets[i] = 20000
    if graph == 1:
        sns.displot(packets)
    if graph == 2:
        sns.displot(packets, kind='kde')
    if graph == 3:
        sns.displot(packets, kind='ecdf')

    plt.show()


# # visualize(3)
# flows = pd.read_csv('all_flows.txt')
#
# packets = flows['packets']
# print(packets.describe())
#
# bytes = flows['bytes']
# print(bytes.describe())
#
# mean_bytes = flows['mean_bytes']
# print(mean_bytes.describe())
#
# duration = flows['duration']
# print(duration.describe())


def my_plotter(lr, T_max, eta_min=0):
    lr =lr
    x = np.linspace(0, 10, 500)  # Sample data.
    y = eta_min + 0.5 * (lr-eta_min) * (1 + np.cos((x / T_max) * np.pi))
    fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')
    ax.plot(x, y, label='CosineLine')
    ax.set_xlabel('epoch')
    ax.set_ylabel('learning rate')
    ax.set_title("CosineAnnealingLR")
    ax.legend()
    plt.show()

my_plotter(0.001,10)