import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from sklearn.metrics import roc_auc_score
import math

# computues cosine similarity
def cosine_sim(x1, x2, dim=1, eps=1e-8):
#  torch.mm does matrix mult
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    # torch.ger => Outer product of input and vec2. If input is a vector 
    # of size nn and vec2 is a vector of size mm , 
    # then out must be a matrix of size (n \times m)(n×m) .

    return ip / torch.ger(w1,w2).clamp(min=eps)

def _get_loss_func(function_name: str):
    if function_name == 'cross_entropy':
        return nn.CrossEntropyLoss(reduction='mean')

    # elif function_name == 'mse':
    #     return nn.MSELoss()
    
    elif function_name == 'nll':
        # print()
        return nn.NLLLoss()
    
    # Cant use triplet margin loss as we dont provide an extra negative input
    elif function_name == 'TripletMarginLoss':
        return nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False)
    else:
        raise RuntimeError('Unkown loss function: {}'.format(function_name))
    


def _get_activation_func(function_name: str):

    if function_name == 'relu':
        return nn.ReLU()

    elif function_name == 'lrelu':
            return nn.LeakyReLU()
        
    elif function_name == 'prelu':
        # return nn.PReLU(num_parameters=1, init=0.25)  --- original values
        return nn.PReLU(num_parameters=1, init=0.3)
    
    elif function_name == 'lrelu':
        return nn.LeakyReLU()

    elif function_name == 'elu':
            return nn.ELU(alpha=1.0, inplace=False)

    elif function_name == 'softplus':
            return nn.Softplus(beta=1, threshold=20)
    else:
        raise RuntimeError(f'Unknown activation func = {function_name}')

def _get_optimiser(optimiser_name: str):
    if optimiser_name == 'sgd':
        optimizer = optim.SGD
    elif optimiser_name == 'adam':
        optimizer = optim.Adam
    # elif optimiser_name == 'Adadelta':
    #     optimizer = optim.Adadelta( lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    
    else:
        raise RuntimeError(f'Unknown optimiser = {optimiser_name}')
    
    return optimizer
class BaseNetwork(nn.Module):
    def __init__(self, network_settings: dict):
        super().__init__()
        self.input_size = network_settings['input_size']
        self.output_size = network_settings['output_size']
        # TODO: Choice of activation function
        # TODO: choice of loss function
        # TODO: Choice of optimiser
        # TODO: batch norm, dropout, use biases?
    
    def _device(self):
        return next(self.parameters()).device
    
    def _is_on_cuda(self):
        next_ = next(self.parameters())
        return next_.is_cuda

    def loss(self, y_hat, y):
        raise NotImplementedError

    def build_network(self):
        raise NotImplementedError
        
    def forward(self):
        raise NotImplementedError

    def train_a_batch(self, x, y):
        self.train()
        self.optimizer.zero_grad()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        loss.backward()
        self.optimizer.step()

        # stats for trainer
        pred_targets = torch.max(y_hat, dim=1)[1]
        auc = roc_auc_score(y.cpu().numpy(), pred_targets.cpu().numpy())

        return {'loss': loss,
                'auc': auc}



class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, output_size, activation_func = None, use_batch_norm = False, use_bias=True,  dropout = 0.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        if activation_func is not None:
            self.activation_func = activation_func
        self.use_batch_norm = use_batch_norm
        self.dropout_val = dropout
        self.use_bias = use_bias

        self.build_network()
    
    def build_network(self):
        if self.dropout_val > 0.0:
            self.dropout = nn.Dropout(self.dropout_val)
        
        self.linear = nn.Linear(in_features=self.input_size,
                                out_features=self.output_size,
                                bias=self.use_bias)
        
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(self.output_size)
        
      
    def forward(self, x):
        out = self.dropout(x) if hasattr(self, 'dropout') else x
        out = self.linear(out)
        out = self.bn(out) if hasattr(self, 'bn') else out
            
        return out


class MLP(BaseNetwork):

    def __init__(self, network_settings):
        super().__init__(network_settings)
        # self.layer_size = network_settings['layer_sizes']
        self.layer_config = network_settings['layer_config'] if 'layer_config' in network_settings else []
        self.use_batch_norm = network_settings["use_batch_norm"]
        self.activation_func = _get_activation_func(network_settings["activation_func"])
        self.loss_function = _get_loss_func(network_settings["loss_func"])
        self.use_bias = network_settings["use_bias"]
        self.dropout_value = network_settings["dropout_val"]
        
        self.layers = nn.ModuleList()
        self.build_network()
        
        optimizer_type = network_settings['optimiser_type']
        
        learning_rate = network_settings['learning_rate']
        # print("got LR",learning_rate )
        weight_decay = network_settings['weight_decay']
        # print("got wd", weight_decay)
        # self.optimizer = _get_optimiser(optimizer_type)(self.parameters(), learning_rate)

        #  Adding weight_decay to check if it improves
        self.optimizer = _get_optimiser(optimizer_type)(self.parameters(), learning_rate,weight_decay=0.2)
    
    
    def build_network(self):
        if self.layer_config == []:
            # TODO: add hidden layers
            raise NotImplementedError
        else:
            self.layer_config = [self.input_size] + self.layer_config
        
        for i in range(len(self.layer_config) - 1):
            input_size = self.layer_config[i]
            output_size = self.layer_config[i + 1]
            layer = FullyConnectedLayer(input_size, output_size, self.activation_func, self.use_bias, self.use_batch_norm, self.dropout_value)
            self.layers.append(layer)

        # Add final output layer
        final_layer = FullyConnectedLayer(self.layer_config[-1], self.output_size, None, self.use_bias, self.use_batch_norm, self.dropout_value)
        self.layers.append(final_layer)

    def loss(self, y_hat, y):
        loss = self.loss_function(y_hat, y)
        return loss
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return F.softmax(out, dim=1)

# class MarginCosineProduct(BaseNetwork):
#     r"""Implement of large margin cosine distance: :
#     Args:
#         in_features: size of each input sample 
#         out_features: size of each output sample
#         s: norm of input feature
#         m: margin -> this is same as the m val used in arcface and sphereFace
#     """

#     def __init__(self, in_features, out_features, s=30.0, m=0.40):
#         super(MarginCosineProduct, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
#         #stdv = 1. / math.sqrt(self.weight.size(1))
#         #self.weight.data.uniform_(-stdv, stdv)

#     def loss(self):
#         # --------------------------------loss function and optimizer-----------------------------
#         lossFunc = torch.nn.CrossEntropyLoss()
    

#     def forward(self, input, label):
#         cosine = cosine_sim(input, self.weight)

#         one_hot.scatter_(1, label.view(-1, 1), 1.0)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = self.s * (cosine - one_hot * self.m)

#         return output

#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#                + 'in_features=' + str(self.in_features) \
#                + ', out_features=' + str(self.out_features) \
#                + ', s=' + str(self.s) \
#                + ', m=' + str(self.m) + ')'



# class AngleLinear(nn.Module):
#     def __init__(self, in_features, out_features, m=4):
#         super(AngleLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.m = m
#         self.base = 1000.0
#         self.gamma = 0.12
#         self.power = 1
#         self.LambdaMin = 5.0
#         self.iter = 0
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#         # duplication formula
#         self.mlambda = [
#             lambda x: x ** 0,
#             lambda x: x ** 1,
#             lambda x: 2 * x ** 2 - 1,
#             lambda x: 4 * x ** 3 - 3 * x,
#             lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
#             lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
#         ]

#     def forward(self, input, label):
#         # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
#         self.iter += 1
#         self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
#         cos_theta = cos_theta.clamp(-1, 1)
#         cos_m_theta = self.mlambda[self.m](cos_theta)
#         theta = cos_theta.data.acos()
#         k = (self.m * theta / 3.14159265).floor()
#         phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
#         NormOfFeature = torch.norm(input, 2, 1)

#         # --------------------------- convert label to one-hot ---------------------------
#         one_hot = torch.zeros_like(cos_theta)
#         one_hot.scatter_(1, label.view(-1, 1), 1)

#         # --------------------------- Calculate output ---------------------------
#         output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
#         output *= NormOfFeature.view(-1, 1)

#         return output

#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#                + 'in_features=' + str(self.in_features) \
#                + ', out_features=' + str(self.out_features) \
#                + ', m=' + str(self.m) + ')'