from tape import ProteinBertForMaskedLM, UniRepForLM, ProteinLSTMForLM, TAPETokenizer
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse

parser = argparse.ArgumentParser(description='TAPE model evaluations')
parser.add_argument('--csv_file', type=str, default='data/', help='location of the data ids')
parser.add_argument('--model', choices=['transformer', 'unirep'],
                    help='model: "transformer","unirep"')
parser.add_argument('--out_file', type=str, default='outputs.npy',
                    help='path + .npy')

args = parser.parse_args()

class DMSDataset(Dataset):
    
    def __init__(self, csv_file, tokenizer):
        data = pd.read_csv(csv_file)
        self.sequences = np.array(data['sequence'])
        self.fitness = np.array(data['fitness'])
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        
        selected_seq = self.sequences[idx]
        selected_fitness = self.fitness[idx]
        
        selected_seq = torch.tensor([self.tokenizer.encode(str(selected_seq))])
        
        return selected_seq, selected_fitness

def get_model_and_tokenizer(model_name):
    if model_name == 'transformer':
        model = ProteinBertForMaskedLM.from_pretrained('bert-base')
        vocab = 'iupac'
    if model_name == 'unirep':
        model = UniRepForLM.from_pretrained('babbler-1900')
        vocab = 'unirep'
#     if model_name == 'lstm':
#         model = ProteinLSTMForLM.from_pretrained()
#         vocab = 'iupac'
    return model, TAPETokenizer(vocab=vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer = get_model_and_tokenizer(args.model)
model.to(device)
model.eval()

dms = DMSDataset(args.csv_file, tokenizer)
eval_loader = DataLoader(dms, batch_size=1, shuffle=False)

with open(args.out_file, 'wb') as f:
    for i, sample in tqdm(enumerate(eval_loader), total=len(eval_loader)):
        with torch.no_grad():
            sequences = sample[0][0,:,:].to(device)
            fitness = sample[1].to(device)
            outputs = model(sequences)[0]
            np.save(f, np.array(outputs.cpu))