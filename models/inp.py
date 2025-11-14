import os
import torch
from typing import List

class TrainArgs:
    def __init__(self):
        self.depth = 3
        self.mpn_hidden = 300
        self.mpn_dropout = 0.1
        self.mpn_activation = 'ReLU'
        self.mpn_bias = True
        self.aggregation = 'sum'
        self.f_mol_size = 0
        self.num_features = 2
        self.num_targets = 2
        self.ffn_num_layers = 2
        self.ffn_hidden = 300
        self.ffn_dropout = 0.1
        self.ffn_activation = 'ReLU'
        self.ffn_bias = True
        self.max_num_mols = 2
        self.solute = True
        self.postprocess = False

        self.epochs = 30
        self.batch_size = 32
        self.lr_scheduler = 'plateau'
        self.learning_rates = [1e-4]
        self.loss_metric = 'mae'
        self.metric = 'mae'
        self.scale = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

        self.input_file = './data/data.csv'
        self.output_dir = './results/'
        self.model_path = os.path.join(self.output_dir, 'best_model.pt')
        self.save_predictions = True


        self.solute_headers: List[str] = ['solute_smiles']
        self.solvent_headers: List[str] = ['inchi_solvent1_smiles', 'inchi_solvent2_smiles']
        self.target_headers: List[str] = ['Gsolv (kcal/mol)', 'Hsolv (kcal/mol)']
        self.feature_headers: List[str] = ['temperature', 'pressure']  

        self.verbose = True
        self.save_model = True
        self.resume = False
        self.pretraining = False

    def print(self):
        for attr in vars(self):
            print(f'{attr}: {getattr(self, attr)}')

    def save(self, path='train_args.json'):
        import json
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, path='train_args.json'):
        import json
        with open(path, 'r') as f:
            self.__dict__.update(json.load(f))
