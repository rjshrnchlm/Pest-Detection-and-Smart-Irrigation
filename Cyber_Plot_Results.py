import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch

No_of_Dataset = 1


def Plot_Encryp():
    for n in range(No_of_Dataset):
        Graph_Time = np.load('Encryption_time.npy', allow_pickle=True)[n]
        values = Graph_Time[:, :5][:, 4]
        Classifier = ['RSA', 'DES', 'ECC', 'MK-FHE', 'GSSA-OMK-FHE']
        colors = ['#00BFFF', '#32CD32', '#ADFF2F', '#FFA500', '#FF1493']
        x = range(len(Classifier))
        bar_width = 0.5

        # Create figure and axis
        fig, ax = plt.subplots()

        # Plot bars with FancyBboxPatch
        for i, value in enumerate(values):
            bar = FancyBboxPatch((x[i], 0), bar_width, value, boxstyle="round,pad=0.0",
                                 linewidth=0, facecolor=colors[i], clip_on=False)
            ax.add_patch(bar)
            ax.bar(x[i] + 0.45, value, color='black', width=bar_width - 0.4, alpha=0.3)
            ax.text(x[i] + 0.25, value + (value / 100), f'{round(value, 2)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=colors[i])

        # Set labels and title
        ax.set_ylabel('Encryption time (S)')
        ax.set_xticks(np.arange(len(Classifier)) + 0.25)
        ax.set_xticklabels(Classifier)
        path = "./Results/encry_time_enc.png"
        plt.savefig(path)
        plt.show()


def Plot_Decryp():
    for n in range(No_of_Dataset):
        Graph_Time = np.load('Decryption_time.npy', allow_pickle=True)[n]
        values = Graph_Time[:, :5][:, 4]
        Classifier = ['RSA', 'DES', 'ECC', 'MK-FHE', 'GSSA-OMK-FHE']
        colors = ['#00BFFF', '#32CD32', '#ADFF2F', '#FFA500', '#FF1493']
        x = range(len(Classifier))
        bar_width = 0.5

        # Create figure and axis
        fig, ax = plt.subplots()

        # Plot bars with FancyBboxPatch
        for i, value in enumerate(values):
            bar = FancyBboxPatch((x[i], 0), bar_width, value, boxstyle="round,pad=0.0",
                                 linewidth=0, facecolor=colors[i], clip_on=False)
            ax.add_patch(bar)
            ax.bar(x[i] + 0.45, value, color='black', width=bar_width - 0.4, alpha=0.3)
            ax.text(x[i] + 0.25, value + (value / 100), f'{round(value, 2)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=colors[i])

        # Set labels and title
        ax.set_ylabel('Decryption time (S)')
        ax.set_xticks(np.arange(len(Classifier)) + 0.25)
        ax.set_xticklabels(Classifier)
        path = "./Results/decrypt_time_enc.png"
        plt.savefig(path)
        plt.show()


def plot_Total_comp_time():
    for n in range(No_of_Dataset):
        Graph_Time = np.load('Time.npy', allow_pickle=True)[n]
        values = Graph_Time[:, :5][:, 4]
        Classifier = ['RSA', 'DES', 'ECC', 'MK-FHE', 'GSSA-OMK-FHE']
        colors = ['#00BFFF', '#32CD32', '#ADFF2F', '#FFA500', '#FF1493']
        x = range(len(Classifier))
        bar_width = 0.5

        # Create figure and axis
        fig, ax = plt.subplots()

        # Plot bars with FancyBboxPatch
        for i, value in enumerate(values):
            bar = FancyBboxPatch((x[i], 0), bar_width, value, boxstyle="round,pad=0.0",
                                 linewidth=0, facecolor=colors[i], clip_on=False)
            ax.add_patch(bar)
            ax.bar(x[i] + 0.45, value, color='black', width=bar_width - 0.4, alpha=0.3)
            ax.text(x[i] + 0.25, value + (value / 100), f'{round(value, 2)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=colors[i])

        # Set labels and title
        ax.set_ylabel('Total Computational time (S)')
        ax.set_xticks(np.arange(len(Classifier)) + 0.25)
        ax.set_xticklabels(Classifier)
        path = "./Results/Total_Comput_time_enc.png"
        plt.savefig(path)
        plt.show()


def plot_key_sencitivity():
    for n in range(No_of_Dataset):
        sencitivity = np.load('key.npy', allow_pickle=True)[n]
        values = sencitivity[:, :5][:, 4]
        Classifier = ['RSA', 'DES', 'ECC', 'MK-FHE', 'GSSA-OMK-FHE']
        colors = ['#00BFFF', '#32CD32', '#ADFF2F', '#FFA500', '#FF1493']
        x = range(len(Classifier))
        bar_width = 0.5

        # Create figure and axis
        fig, ax = plt.subplots()

        # Plot bars with FancyBboxPatch
        for i, value in enumerate(values):
            bar = FancyBboxPatch((x[i], 0), bar_width, value, boxstyle="round,pad=0.0",
                                 linewidth=0, facecolor=colors[i], clip_on=False)
            ax.add_patch(bar)
            ax.bar(x[i] + 0.45, value, color='black', width=bar_width - 0.4, alpha=0.3)
            ax.text(x[i] + 0.25, value + (value / 100), f'{value:.1f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=colors[i])

        # Set labels and title
        ax.set_ylabel('Key Sensitivity Analysis')
        ax.set_xticks(np.arange(len(Classifier)) + 0.25)
        ax.set_xticklabels(Classifier)
        path = "./Results/Key_sen_enc.png"
        plt.savefig(path)
        plt.show()


def plot_Memory_size():
    for n in range(No_of_Dataset):
        Graph = np.load('Memory Size.npy', allow_pickle=True)[n]
        values = Graph[:, :5][:, 4]
        Classifier = ['RSA', 'DES', 'ECC', 'MK-FHE', 'GSSA-OMK-FHE']
        colors = ['#00BFFF', '#32CD32', '#ADFF2F', '#FFA500', '#FF1493']
        x = range(len(Classifier))
        bar_width = 0.5

        # Create figure and axis
        fig, ax = plt.subplots()

        # Plot bars with FancyBboxPatch
        for i, value in enumerate(values):
            bar = FancyBboxPatch((x[i], 0), bar_width, value, boxstyle="round,pad=0.0",
                                 linewidth=0, facecolor=colors[i], clip_on=False)
            ax.add_patch(bar)
            ax.bar(x[i] + 0.45, value, color='black', width=bar_width - 0.4, alpha=0.3)
            ax.text(x[i] + 0.25, value + (value / 100), f'{round(value, 2)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=colors[i])

        # Set labels and title
        ax.set_ylabel('Memory Size (KB)')
        ax.set_xticks(np.arange(len(Classifier)) + 0.25)
        ax.set_xticklabels(Classifier)
        path = "./Results/Mororysize_Algorithm_%s.png"
        plt.savefig(path)
        plt.show()


def Plot_encryption():
    plot_key_sencitivity()
    Plot_Encryp()
    Plot_Decryp()
    plot_Total_comp_time()
    plot_Memory_size()


if __name__ == '__main__':
    Plot_encryption()
