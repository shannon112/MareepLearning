import torch
from torch import nn

# Text Sentiment Classification DNN based
class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # parameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # embedding layer
        self.embedding_matrix = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding_matrix.weight = torch.nn.Parameter(embedding)
        # fix embedding_matrix or not，if False，it will update during the training
        self.embedding_matrix.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)

        # fc layer
        self.classifier = nn.Sequential( 
            nn.Linear(embedding_dim*32, embedding_dim),
            nn.Sigmoid(),
            nn.Linear(embedding_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        # inputs is (batch128, sen_len20, vectordim250])
        inputs = self.embedding_matrix(inputs)
        # x is (batch128, sen_len20 * vectordim250])
        x = inputs.view(inputs.shape[0],-1)
        x = self.classifier(x)
        return x