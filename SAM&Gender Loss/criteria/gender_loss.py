import torch
from torch import nn
import torch.nn.functional as F

from configs.paths_config import model_paths
from models.gender import gender_predictor


class GenderLoss(nn.Module):

    def __init__(self, opts):
        super(GenderLoss, self).__init__()
        self.gender_net = gender_predictor(opts)
        ckpt = torch.load(model_paths['gender_predictor'], map_location="cpu")['model_state_dict']
        ckpt = {'model.'+str(k.replace('-', '_')): v for k, v in ckpt.items()}
        self.gender_net.load_state_dict(ckpt)
        self.gender_net.cuda()
        self.gender_net.eval()
        self.opts = opts


    def extract_gender(self, x):
        for inputs in x: 
            outputs = self.gender_net(inputs)
            _, predicted_gender = torch.max(outputs, 1)

        return predicted_gender

    def forward(self, y_hat, y, target_gender, id_logs, label=None):
        n_samples = y.shape[0]

        if id_logs is None:
            id_logs = []

        input_gender = self.extract_gender(y)
        output_gender = self.extract_gender(y_hat)

        


        for i in range(n_samples):
            # if id logs for the same exists, update the dictionary
            if len(id_logs) > i:
                id_logs[i].update({f'input_age_{label}': float(input_gender[i]) * 100,
                                   f'output_age_{label}': float(output_gender[i]) * 100,
                                   f'target_age_{label}': float(target_gender[i]) * 100})
            # otherwise, create a new entry for the sample
            else:
                id_logs.append({f'input_age_{label}': float(input_gender[i]) * 100,
                                f'output_age_{label}': float(output_gender[i]) * 100,
                                f'target_age_{label}': float(target_gender[i]) * 100})

        loss = F.binary_cross_entropy(output_gender, target_gender)
        return loss, id_logs










