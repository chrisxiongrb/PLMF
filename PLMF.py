import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import MSELoss


class PersonalizedLSTM(nn.Module):
    '''
    Personalized LSTM Cell
    '''
    def __init__(self, num, K):
        self.K = K
        #Input Gate
        self.input_W = nn.Embedding(num, K)
        self.input_U = nn.Linear(K, K)
        #Forget Gate
        self.forget_W = nn.Embedding(num, K)
        self.forget_U = nn.Linear(K, K)
        #Output Gate
        self.output_W = nn.Embedding(num, K)
        self.output_U = nn.Linear(K, K)
        #Cell State
        self.state_W = nn.Embedding(num, K)
        self.state_U = nn.Linear(K, K)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, X, h, C):
        forget_gate = F.sigmoid(self.forget_W(X) + self.forget_U(h))
        input_gate = F.sigmoid(self.input_W(X) + self.input_U(h))
        input_gate = self.dropout(input_gate)
        new_info = F.tanh(self.state_W(X) + self.state_U(h))
        new_C = forget_gate * C + input_gate * new_info
        output_gate = F.sigmoid(self.output_W(X) + self.output_U(h))
        output_gate = self.dropout(output_gate)
        new_h = output_gate * F.tanh(new_C)
        new_h = self.dropout(new_h)
        return new_h, new_C


class PLMF:
    '''
    ## Personalized LSTM Based Matrix Factorization
    '''
    def __init__(self, user_num, service_num, K, user_h, user_C, service_h, service_C, length):
        '''
        - user_num: number of users
        - service_num: number of services
        - K: size of latent features
        - user_h: user side P-LSTM start hidden vector
        - user_C: user side P-LSTM start cell state
        - service_h: service side P-LSTM start hidden vector
        - service_C: service side P-LSTM start cell state
        - length: size of sliding window (aka T in paper)
        '''
        self.length = length
        self.user_num = user_num
        self.service_num = service_num
        #User-Side and Service-Side P-LSTM
        self.user_cell = PersonalizedLSTM(user_num, K)
        self.service_cell = PersonalizedLSTM(service_num, K)
        #Holding State
        self.user_h = Variable(torch.from_numpy(user_h).type(torch.FloatTensor))
        self.user_C = Variable(torch.from_numpy(user_C).type(torch.FloatTensor))
        self.service_h = Variable(torch.from_numpy(service_h).type(torch.FloatTensor))
        self.service_C = Variable(torch.from_numpy(service_C).type(torch.FloatTensor))
        if torch.cuda.is_available():
            self.user_h = self.user_h.cuda()
            self.user_C = self.user_C.cuda()
            self.service_h = self.service_h.cuda()
            self.service_C = self.service_C.cuda()
        #Loss
        self.mse = MSELoss(size_average=False, reduce=True)
        #Used for update Holding State
        self.all_services = Variable(torch.arange(service_num).type(dtypei), volatile=True)
        self.all_users = Variable(torch.arange(user_num).type(dtypei), volatile=True)
        
    def forward(self, users, services, mask):
        '''
        For training
        - users: input users, 2-d tensor, batch first
        - services: input services, 2-d tensor, batch first
        - mask: indicates real value and padding. Byte Tensor
        '''
        users_h, users_C = self.user_h[users], self.user_C[users]
        services_h, services_C = self.service_h[services], self.service_C[services]
        length = self.length
        users_hs, services_hs = [None for i in range(length)], [None for i in range(length)]        
        
        for i in range(length):
            users_h, users_C = self.user_cell(users, users_h, users_C)
            services_h, services_C = self.service_cell(services, services_h, services_C)
            users_hs[i] = users_h            
            services_hs[i] = services_h
        
        users_embeddings, services_embeddings = torch.stack(users_hs, dim = 1), torch.stack(services_hs, dim = 1)
        score = torch.sum(users_embeddings * services_embeddings, dim = 2)
        score = score[mask]
        return score

    
    def test(self, users, services):
        '''
        For testing
        '''
        users_h, users_C = self.user_h[users], self.user_C[users]
        services_h, services_C = self.service_h[services], self.service_C[services]
        length = self.length
        for i in range(length):
            users_h, users_C = self.user_cell(users, users_h, users_C)
            services_h, services_C = self.service_cell(services, services_h, services_C)
        score = torch.sum(users_h * services_h, dim = 1)
        return score
    
    def loss(self, pred, y, mask, lamda): 
        '''
        Loss Function
        - pred: model prediction
        - y: true value
        '''
        y = y[mask]
        l = self.mse(pred, y)
        for param in self.parameters():
            if param.requires_grad:
                l += lamda * torch.sum(param ** 2)
        return l 

    def initParam(self):
        '''
        Init Cell State
        '''
        nn.init.orthogonal(self.user_cell.input_U.weight)
        nn.init.orthogonal(self.user_cell.input_W.weight)
        nn.init.orthogonal(self.user_cell.forget_U.weight)
        nn.init.orthogonal(self.user_cell.forget_W.weight)
        nn.init.orthogonal(self.user_cell.output_U.weight)
        nn.init.orthogonal(self.user_cell.output_W.weight)
        nn.init.orthogonal(self.service_cell.input_U.weight)
        nn.init.orthogonal(self.service_cell.input_W.weight)
        nn.init.orthogonal(self.service_cell.forget_U.weight)
        nn.init.orthogonal(self.service_cell.forget_W.weight)
        nn.init.orthogonal(self.service_cell.output_U.weight)
        nn.init.orthogonal(self.service_cell.output_W.weight)

        nn.init.constant(self.user_cell.forget_U.bias, 1.0)
        nn.init.constant(self.service_cell.forget_U.bias, 1.0)
    
    def refreshState(self):
        '''
        When training done, refresh model state to train new time interval data.
        '''
        last_user_h, last_user_C = self.user_h[self.all_users, :], self.user_C[self.all_users, :]
        last_service_h, last_service_C = self.service_h[self.all_services, :], self.service_C[self.all_services, :]

        user_h, user_C = self.user_cell(self.all_users, last_user_h, last_user_C)
        service_h, service_C = self.service_cell(self.all_services, last_service_h, last_service_C)

        return (user_h, user_C), (service_h, service_C)

