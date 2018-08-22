import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T


class AutoEncoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out, H1=200, da=20):
        """
        Initialize the model configurations and parameters.
        In the model, the network structure is like [D_in, H1, H, H1, D_out], and D_in equals to D_out.

        :param D_in: the dimension of the input
        :param H: the dimension of the bottleneck layer
        :param D_out: the dimension of the output
        :param H1: the dimension of the first hidden layer
        :param da: the dimension of the attention model
        """
        super(AutoEncoder, self).__init__()
        if torch.cuda.is_available():
            self.linear1 = torch.nn.Linear(D_in, H1, bias=False).cuda()
            self.linear2 = torch.nn.Linear(H1, H).cuda()
            self.linear3 = torch.nn.Linear(H, H1).cuda()
            self.linear4 = torch.nn.Linear(H1, D_out).cuda()
            self.attention_matrix1 = Variable(torch.zeros(da, H1).type(T.FloatTensor), requires_grad=True)
            # self.attention_matrix2 = Variable(torch.zeros(20, 30).type(T.FloatTensor), requires_grad=True)
            self.attention_matrix1 = torch.nn.init.xavier_uniform_(self.attention_matrix1)
            # self.attention_matrix2 = torch.nn.init.xavier_uniform(self.attention_matrix2)
            self.self_attention = torch.nn.Linear(da, 1).cuda()
        else:
            self.linear1 = torch.nn.Linear(D_in, H1, bias=False)
            self.linear2 = torch.nn.Linear(H1, H)
            self.linear3 = torch.nn.Linear(H, H1)
            self.linear4 = torch.nn.Linear(H1, D_out)
            self.attention_matrix1 = Variable(torch.zeros(da, H1).type(T.FloatTensor), requires_grad=True)
            # self.attention_matrix2 = Variable(torch.zeros(20, 30).type(T.FloatTensor), requires_grad=True)
            self.attention_matrix1 = torch.nn.init.xavier_uniform_(self.attention_matrix1)
            # self.attention_matrix2 = torch.nn.init.xavier_uniform(self.attention_matrix2)
            self.self_attention = torch.nn.Linear(da, 1)

    def forward(self, batch_item_index, place_correlation):
        """
        The forward pass of the autoencoder.
        :param batch_item_index: a list of arrays that each array stores the place id a user has been to
        :param place_correlation: the pairwise poi relation matrix
        :return: the predicted ratings
        """
        item_vector = self.linear1.weight[:, T.LongTensor(batch_item_index[0].astype(np.int32))]
        # Compute the neighbor inner products
        inner_product = item_vector.t().mm(self.linear4.weight.t())
        item_corr = Variable(
            torch.from_numpy(place_correlation[batch_item_index[0]].toarray()).type(T.FloatTensor))
        inner_product = inner_product * item_corr
        neighbor_product = inner_product.sum(dim=0).unsqueeze(0)

        # Compute the self attention score
        score = F.tanh(self.attention_matrix1.mm(item_vector))
        score = F.softmax(score, dim=1)
        embedding_matrix = score.mm(item_vector.t())
        linear_z = self.self_attention(embedding_matrix.t()).t()

        # print score
        for i in range(1, len(batch_item_index)):
            item_vector = self.linear1.weight[:, T.LongTensor(batch_item_index[i].astype(np.int32))]
            # Compute the neighbor inner products
            inner_product = item_vector.t().mm(self.linear4.weight.t())
            item_corr = Variable(
                torch.from_numpy(place_correlation[batch_item_index[i]].toarray()).type(T.FloatTensor))
            inner_product = inner_product * item_corr
            inner_product = inner_product.sum(dim=0).unsqueeze(0)
            neighbor_product = torch.cat((neighbor_product, inner_product), 0)

            # Compute the self attention score
            score = F.tanh(self.attention_matrix1.mm(item_vector))
            score = F.softmax(score, dim=1)
            embedding_matrix = score.mm(item_vector.t())
            tmp_z = self.self_attention(embedding_matrix.t()).t()
            linear_z = torch.cat((linear_z, tmp_z), 0)

        z = F.tanh(linear_z)
        z = F.dropout(z, training=self.training)
        z = F.tanh(self.linear2(z))
        z = F.dropout(z, training=self.training)
        d_z = F.tanh(self.linear3(z))
        d_z = F.dropout(d_z, training=self.training)
        y_pred = F.sigmoid(self.linear4(d_z) + neighbor_product)

        return y_pred