import pickle
import os


def save_data(data, path, name):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(f'{path}/{name}.pkl', 'wb') as p:
        pickle.dump(data, p)
        
def load_data(path, name):
    with open(f'{path}/{name}.pkl', 'rb') as p:
        return pickle.load(p)
    
