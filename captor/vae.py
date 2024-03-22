import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm 

# Clear up the memory
torch.cuda.empty_cache()

# Explicitly run garbage collector
gc.collect()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VariationalEncoder(nn.Module):
    def __init__(self,n, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(n, int(n/latent_dims))
        self.linear2 = nn.Linear(int(n/latent_dims), latent_dims)
        self.linear3 = nn.Linear(int(n/latent_dims), latent_dims)

        self.N = torch.distributions.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        
        if device == 'cuda':
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.sigmoid(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape).to(device)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):
    def __init__(self,n, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, int(n/latent_dims))
        self.linear2 = nn.Linear(int(n/latent_dims), int(n))

    def forward(self, z):
        z = torch.sigmoid(self.linear1(z))
        z = torch.tanh(self.linear2(z))
        return z
    
class VariationalAutoencoder(nn.Module):
    def __init__(self,args,n,model_path):
        super(VariationalAutoencoder, self).__init__()
        
        self.args = args
        self.model_path = model_path
        # n is embedding size
        self.encoder = VariationalEncoder(n,self.args.latentspace).to(device)
        self.decoder = Decoder(n,self.args.latentspace).to(device)
        
        # Validation using MSE Loss function
        self.loss_function = torch.nn.MSELoss().to(device)
        

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def train_vae(self,X_train0):
        self.scaler = MinMaxScaler()
        X_train=self.scaler.fit_transform(X_train0)
                
        # Using an Adam Optimizer with lr = 0.1
        optimizer = torch.optim.Adam(self.parameters(),
                                    lr = self.args.lr,
                                    weight_decay = 1e-6)

        epochs = self.args.epoch
        batch = self.args.batch
        vec = np.array([i for i in range(X_train.shape[0])])
        XD = X_train
        total_data = int(XD.shape[0] / batch)
        best_loss = np.inf
        count_epoch = 0
        # Create the tqdm progress bar for the epochs
        for epoch in range(epochs):
            losses = []

            with tqdm(total=total_data, desc=f"Epoch {epoch}") as pbar:

                for i in range(total_data):
                    XB = XD[i * batch:(i + 1) * batch]
                    XBT = torch.FloatTensor(XB).to(device)

                    # Output of Autoencoder
                    reconstructed = self(XBT)

                    # Calculating the loss function
                    loss = self.loss_function(reconstructed, XBT)

                    # The gradients are set to zero,
                    # the gradient is computed and stored.
                    # .step() performs parameter update
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Storing the losses in a list for plotting
                    losses.append(loss.cpu().detach().numpy())

                    # Update the progress bar with the average loss for this epoch
                    average_loss = sum(losses[-(i+1):]) / (i+1)
                    pbar.set_postfix(loss="{:.8f}".format(average_loss))
                    pbar.update(1)
            
            if average_loss < best_loss:
                print(f"Best model with loss {average_loss} being saved at epoch {epoch}")
                
                # Saving best model
                torch.save(self.state_dict(), f'{self.model_path}/{self.args.dataset}/vae-model.pth')  #
                best_loss = average_loss
                count_epoch = 0
            
            # Early Stopping
            if count_epoch > self.args.earlystop:
                break

            count_epoch += 1
        
        self.load_state_dict(torch.load(f'{self.model_path}/{self.args.dataset}/vae-model.pth'))
        
    def test(self,data):
        """ Used trained model and return reconstruction loss for given dataset 

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.eval()
        self.loss_function = torch.nn.MSELoss(reduction='none').to(device)

        ypred=[]
        test=self.scaler.transform(data)
        for i in range(math.ceil(data.shape[0]/self.args.batch)):
            XB=test[i*self.args.batch:(i+1)*self.args.batch]
            XBT=torch.FloatTensor(XB).to(device)
            reconstructed = self(XBT)
            res=torch.mean(self.loss_function(reconstructed, XBT),1)
            #print(res.shape)
            ypred.extend(res.cpu().detach().numpy().flatten().tolist())
        scoresTest=np.array(ypred)
        
        return scoresTest
        
        
