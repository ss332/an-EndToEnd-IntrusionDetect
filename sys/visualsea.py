import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
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
    ax.plot(x, y,label="Cosine")
    ax.set_xlabel('epoch')
    ax.set_ylabel('learning rate')
    ax.set_title("CosineAnnealingLR")
    ax.legend()
    ax.grid(False)
    plt.show()

# Accuracy : 97.29839285714286 544871.0



def losses():
    loss=torch.load('losses3.pt')
    plt.xlabel('epoch',fontsize=14)
    plt.ylabel('loss',fontsize=14)
    plt.plot(loss)
    plt.grid(False)
    plt.show()

losses()
def plot_bar():
    df = pd.DataFrame( {"class":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                        "recall":[94.50780742063098, 99.99028324345333, 99.99218780516387, 100.0, 99.65341488277268,
                                  100.0, 100.0, 99.98847826713137, 89.55223880597015, 99.99143150919684, 86.66666666666667,
                                  81.48148148148148, 20.0, 99.89962440098432, 99.84631934839403],
                        "precision":[99.94212198665309, 100.0, 99.99218780516387, 99.97734224538348, 98.36989333869994,
                                     100.0, 99.98983739837398, 99.99423880167075, 92.3076923076923, 100.0, 87.5, 91.66666666666667,
                                     40.0, 99.70430932799043, 68.86938455341432],
                        "f1Score":[97.14902799620148, 99.9951413856768, 99.99218780516388, 99.98866983911171, 99.00749442981567,
                                   100.0, 99.99491844097768, 99.99135845143448, 90.90909090909089, 99.99571557104298,
                                   87.08133971291866, 86.27450980392157, 26.666666666666668, 99.80187130530418,
                                   81.51410217805882]})
    sns.catplot(x="class",y="f1Score", kind="bar", data=df)
    plt.show()




