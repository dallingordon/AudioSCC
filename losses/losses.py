import torch
import torch.nn as nn

class ConsecutiveDifferenceHigherOrderLossBatch(nn.Module):
    def __init__(self, consecutive_size, order=1):
        super(ConsecutiveDifferenceHigherOrderLossBatch, self).__init__()
        self.consecutive_size = consecutive_size
        self.order = order

    def forward(self, prediction, target):
        pred_reshape = prediction.view(-1, self.consecutive_size)
        target_reshape = target.view(-1, self.consecutive_size)
        result = torch.zeros(1, device=prediction.device)  # To ensure it uses the same device as the input


        pred_a = pred_reshape[1:, :]
        pred_b = pred_reshape[:-1, :]
        target_a = target_reshape[1:, :]
        target_b = target_reshape[:-1, :]
        for i in range(self.order):
            pred_dif = pred_a - pred_b
            target_dif = target_a - target_b
            pred_a = pred_dif[1:, :]
            pred_b = pred_dif[:-1, :]
            target_a = target_dif[1:, :]
            target_b = target_dif[:-1, :]

            result += torch.mean((pred_dif - target_dif) ** 2) / self.order
        return result


class ConsecutiveDifferenceHigherOrderLoss(nn.Module):
    def __init__(self, consecutive_size, order=1):
        super(ConsecutiveDifferenceHigherOrderLoss, self).__init__()
        self.consecutive_size = consecutive_size
        self.order = order

    def forward(self, prediction, target):
        pred_reshape = prediction.view(-1, self.consecutive_size)
        target_reshape = target.view(-1, self.consecutive_size)
        result = torch.zeros(1, device=prediction.device)  # To ensure it uses the same device as the input


        pred_a = pred_reshape[:, 1:]
        pred_b = pred_reshape[:, :-1]
        target_a = target_reshape[:, 1:]
        target_b = target_reshape[:, :-1]
        for i in range(self.order):
            pred_dif = pred_a - pred_b
            target_dif = target_a - target_b
            pred_a = pred_dif[:, 1:]
            pred_b = pred_dif[:, :-1]
            target_a = target_dif[:, 1:]
            target_b = target_dif[:, :-1]

            result += torch.mean((pred_dif - target_dif) ** 2) / self.order
        return result

class PairwiseDifferenceLoss(nn.Module):
    def __init__(self, consecutive_size, device, scale_up=100_000):
        super(PairwiseDifferenceLoss, self).__init__()
        self.consecutive_size = consecutive_size
        self.device = device
        self.scale_up  = scale_up

    def forward(self, data, labels):
        
        ####YOU HAVE TO MAKE THIS batch by consec!!
        data = data.view(-1, self.consecutive_size)
        labels = labels.view(-1, self.consecutive_size)
        batch_size = data.shape[0]
        #print(data.shape, labels.shape)
        # Expand the tensor to compute pairwise differences
        data_expanded1 = data.unsqueeze(2).expand(batch_size, self.consecutive_size, self.consecutive_size)
        data_expanded2 = data.unsqueeze(1).expand(batch_size, self.consecutive_size, self.consecutive_size)
        
        # Compute differences between data and labels
        data_differences = (data_expanded1 - data_expanded2)
        #print(data_expanded1 == data_expanded2)
        #print(data_expanded1,data_expanded2, data_differences)
        # Compute differences with the labels
        labels_expanded1 = labels.unsqueeze(2).expand(batch_size, self.consecutive_size, self.consecutive_size)
        labels_expanded2 = labels.unsqueeze(1).expand(batch_size, self.consecutive_size, self.consecutive_size)
        label_differences = (labels_expanded1 - labels_expanded2)
        #print(labels_expanded1,labels_expanded1, label_differences)
        
        #mask = torch.triu(torch.ones(batch_size, self.consecutive_size, self.consecutive_size), diagonal=1).to(self.device)
        
        differences = (( data_differences*self.scale_up - label_differences*self.scale_up)**2)
        #print(differences.shape)
        # Apply a mask to ignore upper triangle
        
       
        #print(differences)
        # Compute the mean to return a scalar loss
        loss = differences.mean()
        
        return loss
