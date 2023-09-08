import pickle
import os
import json
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


def save_data(data, path, name):
    """save any data into a pickle
    """
    if not os.path.exists(path):
        os.mkdir(path)
    with open(f'{path}/{name}.pkl', 'wb') as p:
        pickle.dump(data, p)


def load_data(path: str, name):
    """load data from a pickle
    """
    with open(f'{path}/{name}.pkl', 'rb') as p:
        return pickle.load(p)


def is_float(string: str) -> bool:
    """helper function to determine if a string is in float format
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def get_curves(json_paths: list,
               df: pd.DataFrame,
               prefix: str,
               labels: list,
               width_ratios: list[int] = [5, 4],
               height_ratios: list[int] = [4, 5],
               row=1,
               col=2):
    """Plot the loss curves for both models
    """
    train_losses = []
    
    for path in json_paths:
        with open(os.path.join(path, 'logs.json')) as j:
            logs_json = json.load(j)
            train_losses.append(logs_json['loss'])
    
    yticks = []
    
    for train_loss in train_losses:
        if yticks:
            if min(train_loss) > yticks[0]:
                yticks.insert(0, max(train_loss))
        else:
            yticks.append(min(train_loss))

    # Plot the training losses for two models
    _x = 0.5
    if width_ratios:
        fig, (ax, ax_df) = plt.subplots(row, col, gridspec_kw={'width_ratios': width_ratios}, figsize=(15, 6))
        _loc = 'center'
        _y = 0.81
    else:
        fig, (ax, ax_df) = plt.subplots(row, col, gridspec_kw={'height_ratios': height_ratios}, figsize=(10, 15))
        _loc = 'upper center'
        _y = 1

    for i, train_loss in enumerate(train_losses):
        ax.plot(train_loss, label=labels[i])

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
                if float(n) >= 0.0001 or n == 0:
                    tmp.append(round(float(n), 4))
                else:
                    n = str(n).split('-')
                    n = f'{n[0][:4]}e-{n[1]}'
                    tmp.append(n)
            else:
                tmp.append(n)
        
        cell_text.append(tmp)
    df_table = ax_df.table(cellText=cell_text, loc=_loc)
    df_table.auto_set_column_width(col=list(range(len(df.columns))))
    ax_df.set_title('Evaluation Results', x=_x, y=_y)
    ax_df.axis('off')
    plt.subplots_adjust(wspace=0.1)
    title = title.replace('.', '_')
    plt.savefig('_'.join(title.split()))
    plt.show()
