
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Plot:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    # ploting the corrolation among features
    def correlation(self):
        corr = self.X.corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, len(self.X.columns), 1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.X.columns)
        ax.set_yticklabels(self.X.columns)
        plt.show()

    def pairplot(self , size):
        dataSize = self.X.shape[1]
        keys = self.X.keys()
        data = self.X
        data['class'] = self.Y

        i = 0
        while i < dataSize:
            j = 0
            while j < dataSize:
                sns.pairplot(self.X, x_vars=keys[i:i+size], y_vars=keys[j:j+size], hue='class')
                plt.savefig('./plot/pairPlot' + str(i)+str(j) + '.png')
                j += size
            i += size

