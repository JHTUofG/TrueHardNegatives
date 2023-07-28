import pickle
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def save_data(data, path, name):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(f'{path}/{name}.pkl', 'wb') as p:
        pickle.dump(data, p)
        
def load_data(path, name):
    with open(f'{path}/{name}.pkl', 'rb') as p:
        return pickle.load(p)
    
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def get_curves(json_paths : list, df, prefix, labels):
    """Plot the loss curves for both models
    """
    train_losses = []
    
    for path in json_paths:
        with open(os.path.join(path, 'logs.json')) as j:
            logs_json = json.load(j)
            train_losses.append(logs_json['loss'])
    
#     with open(json_path_0) as j:
#         logs_json_0 = json.load(j)
    
#     with open(json_path_1) as j:
#         logs_json_1 = json.load(j)
    
    # train_losses_0 = logs_json_0['loss']
    # train_losses_1 = logs_json_1['loss']
    
    yticks = []
    
    for train_loss in train_losses:
        if yticks:
            if min(train_loss) > yticks[0]:
                yticks.insert(0, max(train_loss))
        else:
            yticks.append(min(train_loss))
    
    # if min(train_losses_1) > min(train_losses_0):
    #     ytick_2 = min(train_losses_1)
    #     ytick_3 = min(train_losses_0)
    # else:
    #     ytick_2 = min(train_losses_0)
    #     ytick_3 = min(train_losses_1)

    # Plot the training losses for two models
    fig, (ax, ax_df) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 4]}, figsize=(15, 6))
    # plt.figure()
    #fig.tight_layout()
    for i, train_loss in enumerate(train_losses):
        ax.plot(train_loss, label=labels[i])
    # ax.plot(train_losses_0, label='Baseline Model Training Loss')
    # ax.plot(train_losses_1, label='True Negative Model Training Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.margins(x=0)
    ax.set_xticks([1, 100, 1000, len(train_losses[0])])
    
    yticks_pre = [train_losses[-1][0], 1]
    yticks_pre.extend(yticks)
    yticks = [max(yticks_pre)] + [1] + [min(yticks_pre)]
    ax.set_yticks(yticks)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()
    title = f'{prefix} Loss Curves'
    ax.set_title(title)
    cell_text = []
    for i, column in enumerate(df.columns):
        tmp = [column]
        
        for n in df[column]:
            if is_float(str(n)) and str(n) != 'nan':
                if float(n) >= 0.0001:
                    tmp.append(round(float(n), 4))
                else:
                    n = str(n).split('-')
                    n = f'{n[0][:4]}e-{n[1]}'
                    tmp.append(n)
            else:
                tmp.append(n)
        
        cell_text.append(tmp)
    df_table = ax_df.table(cellText=cell_text, loc='center')
    df_table.auto_set_column_width(col=list(range(len(df.columns))))
    ax_df.set_title('Evaluation Results', x=0.52, y=0.81)
    ax_df.axis('off')
    plt.subplots_adjust(wspace=0.1)
    plt.savefig('_'.join(title.split()))
    plt.show()
