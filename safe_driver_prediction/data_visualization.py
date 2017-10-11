from data_util import *
import matplotlib.pyplot as plt

train_data = load_train_data()

for i in range(1, len(train_data.columns)):
    plt.figure(i)
    plt.hist(train_data.iloc[:, i])
    plt.title(str(train_data.columns[i])+" Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.savefig("data_plot/"+str(train_data.columns[i]+".png"))

