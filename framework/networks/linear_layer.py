import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
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


class BaseNetwork(nn.Module):
    def __init__(self, input_size: tuple, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        # TODO: Choice of activation function
        # TODO: choice of loss function
        # TODO: Choice of optimiser
        # TODO: batch norm, dropout, use biases?
        self.build_network()
    
    def _device(self):
        return next(self.parameters()).device
    
    def _is_on_cuda(self):
        next_ = next(self.parameters())
        return next_.is_cuda

    def loss(self):
        raise NotImplementedError

    def build_network(self):
        raise NotImplementedError
        
    def forward(self):
        raise NotImplementedError

class MarginCosineProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample 
        out_features: size of each output sample
        s: norm of input feature
        m: margin -> this is same as the m val used in arcface and sphereFace
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def loss(self):
        # --------------------------------loss function and optimizer-----------------------------
        lossFunc = torch.nn.CrossEntropyLoss()
    

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)

        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'



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