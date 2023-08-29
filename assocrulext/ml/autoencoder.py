import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.base import TransformerMixin


class AutoEncoderDimensionReduction(TransformerMixin):
    def __init__(self,encoding_dim, epochs, batch_size, lr=1e-3,novelty_score=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.encoding_dim = encoding_dim
        self.novelty_score=novelty_score

    #def loss_function(self, X, Y):
        #reconstruction_loss = nn.functional.mse_loss(X, Y)
        #novelty_score_tensor = torch.FloatTensor(self.novelty_score) 
        #average_novelty_score = torch.mean(novelty_score_tensor)
        #total_loss = average_novelty_score * reconstruction_loss
       # return total_loss
    
    class Encoder(nn.Module):
        def __init__(self, input_dim, encoding_dim):
            super(AutoEncoderDimensionReduction.Encoder, self).__init__()
            self.encoder_layers = nn.ModuleList([
                GCNConv(input_dim, 256), 
                nn.Tanh(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256,128),
                nn.Tanh(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128, encoding_dim),
                nn.Tanh()
            ])

        def forward(self, x, edge_index):
            for layer in self.encoder_layers:
                if isinstance(layer, GCNConv):
                    x = layer(x, edge_index)
                else:
                    x = layer(x)
            return x
    class Decoder(nn.Module):
        def __init__(self, encoding_dim, input_dim):
            super(AutoEncoderDimensionReduction.Decoder, self).__init__()
            self.decoder_layers = nn.ModuleList([
                nn.Linear(encoding_dim, 128),
                nn.Tanh(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128,256),
                nn.Tanh(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                GCNConv(256, input_dim),
                nn.Sigmoid()  
            ])


        def forward(self, x ,edge_index):
                for layer in self.decoder_layers:
                    if isinstance(layer, GCNConv):
                        x = layer(x, edge_index)  
                    else:
                        x = layer(x)
                return x  
        
    def fit(self, adj_matrix, y=None):
        input_dim = adj_matrix.shape[1]
        encoding_dim = self.encoding_dim
        encoder_model = self.Encoder(input_dim, encoding_dim)
        decoder_model = self.Decoder(encoding_dim, input_dim)
        autoencoder = nn.ModuleList([encoder_model, decoder_model])

        optimizer = optim.Adam(autoencoder.parameters(), lr=self.lr)
        #criterion = self.loss_function
        criterion = nn.MSELoss()
        adj_matrix_tensor = torch.FloatTensor(adj_matrix)
        edge_index = adj_matrix_tensor.nonzero().t()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            inputs = adj_matrix_tensor
            encoded = encoder_model(inputs, edge_index)
            decoded = decoder_model(encoded, edge_index)
            loss = criterion(decoded, inputs)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

        self.encoder_model = encoder_model
        with torch.no_grad():
            return self.encoder_model(torch.tensor(adj_matrix, dtype=torch.float32), edge_index).numpy()


    def fit_transform(self, adj_matrix):
        embeddings=self.fit(adj_matrix)
        return embeddings