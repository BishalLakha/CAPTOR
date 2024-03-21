import argparse
from captor.captor import CAPTOR

import warnings
warnings.filterwarnings("ignore")

def get_bool(v):
    if isinstance(v, bool):
        return v
    bool_dict = {'yes': True, 'true': True, 't': True, 'y': True, '1': True,
                 'no': False, 'false': False, 'f': False, 'n': False, '0': False}
    try:
        return bool_dict[v.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
  
    args = argparse.ArgumentParser("captor")
    
    #Graph embedding parameters
    args.add_argument('-n1', "--clustersize", default = 3, type=int, 
                      help="Number of clusters")
    args.add_argument('-n2', '--dimensize', default = 4, type=int, 
                      help='Parameter determining dimension of embeddings')
    args.add_argument('-a', '--alpha', default = 5, type=int, 
                      help='modulates the effect of the transition matrix resulting from any two consecutive timestamps')
    args.add_argument('-d', '--delta', default = 250, type=int, 
                      help='modulates the effect of the transition matrices that result from any two timestamps in the window')
    args.add_argument('-w', '--trainwin', type=int, default = 1,
                      help='Snapshot size in minutes')
    
    # VAE parameters
    args.add_argument('-e', '--epoch', default= 250, type=int,
                      help='Number of epochs for VAE')
    args.add_argument('-es', '--earlystop', default= 25, type=int,
                      help='Minimum number of epochs without imporvement')
    args.add_argument('-l', '--lr', default=0.001, type=float,
                      help='Learning rate for VAE')
    args.add_argument('-ls', '--latentspace', default=4, type=float,
                      help='Latent space dimension')    
    
    # anomaly detection paprameter
    args.add_argument('-th', '--threshold', default=0.95, type=int,
                      help='Threshold percentile')
    
    # dataset
    args.add_argument('-dt', '--dataset', type=str, default='cicids',
                      help='Name of the dataset')
    
    # General settings
    args.add_argument('-t', '--train', nargs='?', const=True, default = True, type=get_bool,
                      help='Is training?')
    args.add_argument('-s', '--save', nargs='?', const=True, default = True, type=get_bool,
                      help='Save model and results')
    
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = CAPTOR(args,"./data","./models")
    
    model.train()