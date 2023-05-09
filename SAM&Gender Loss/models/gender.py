import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
from configs.paths_config import model_paths


from torchvision import models

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object


# # RESNET18







# model.eval()

#     with torch.no_grad():
#         running_loss = 0.
#         running_corrects = 0

#         for inputs, label in tqdm(zip(X_test,y_test)):
#             inputs = inputs.to(device)
#             label = label.to(device)

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             loss = criterion(outputs, label)

#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == label.data)

#         epoch_loss = running_loss / len(y_test.dataset)
#         epoch_acc = running_corrects / len(y_test.dataset) * 100.
#         print('[Validation #{}] Loss: {:.4f} Acc: {:.4f}%'.format(epoch, epoch_loss, epoch_acc))

class gender_predictor(nn.Module):
	def __init__(self, opts):
		
		super(gender_predictor, self).__init__()
		# self.set_opts(opts)
		self.model = None 
		# self.optimizer = None 
		# self.criterion = None
		self.load_weights()
    

	def load_weights(self):

		# self.criterion = nn.CrossEntropyLoss()
		# self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
		self.model = models.resnet18(pretrained=True)
		num_features = self.model.fc.in_features
		self.model.fc = nn.Linear(num_features, 2) # binary classification (num_of_class == 2)
		checkpoint = torch.load(model_paths['gender_predictor'])
		self.model.load_state_dict(checkpoint['model_state_dict'])
		# self.criterion = nn.CrossEntropyLoss()
		# self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
		# self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
		# self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    # self.model.eval()

