import torch
from torch import nn
import torch.nn.functional as F

from configs.paths_config import model_paths
from models.gender import gender_predictior


class GenderLoss(nn.Module):

    def __init__(self, opts):
        super(GenderLoss, self).__init__()
        self.gender_net = gender_predictior()
        ckpt = torch.load(model_paths['gender_predictior'], map_location="cpu")['state_dict']
        ckpt = {k.replace('-', '_'): v for k, v in ckpt.items()}
        self.gender_net.load_state_dict(ckpt)
        self.gender_net.cuda()
        self.gender_net.eval()
        self.opts = opts

    def __get_predicted_age(self, age_pb):
        predict_age_pb = F.softmax(age_pb)
        predict_age = torch.zeros(age_pb.size(0)).type_as(predict_age_pb)
        for i in range(age_pb.size(0)):
            for j in range(age_pb.size(1)):
                predict_age[i] += j * predict_age_pb[i][j]
        return predict_age

    def extract_ages(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        predict_age_pb = self.age_net(x)['fc8']
        predicted_age = self.__get_predicted_age(predict_age_pb)
        return predicted_age

    def forward(self, y_hat, y, target_gender, id_logs, label=None):
        n_samples = y.shape[0]

        if id_logs is None:
            id_logs = []

        # input_ages = self.extract_ages(y) / 100.
        # output_ages = self.extract_ages(y_hat) / 100.

        for inputs, label in tqdm(zip(X_test,y_test)):
            inputs = inputs.to(device)
            label = label.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, label)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == label.data)




        for i in range(n_samples):
            # if id logs for the same exists, update the dictionary
            if len(id_logs) > i:
                id_logs[i].update({f'input_age_{label}': float(input_ages[i]) * 100,
                                   f'output_age_{label}': float(output_ages[i]) * 100,
                                   f'target_age_{label}': float(target_ages[i]) * 100})
            # otherwise, create a new entry for the sample
            else:
                id_logs.append({f'input_age_{label}': float(input_ages[i]) * 100,
                                f'output_age_{label}': float(output_ages[i]) * 100,
                                f'target_age_{label}': float(target_ages[i]) * 100})

        loss = F.mse_loss(output_ages, target_ages)
        return loss, id_logs










