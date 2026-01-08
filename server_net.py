import nest_asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax
import torch.nn.functional as F

def squash(x, dim=-1):
    # x: (batch, n_caps_out, caps_out_dim)
    s_norm_sq = torch.sum(x ** 2, dim=dim, keepdim=True)
    s_norm = torch.sqrt(s_norm_sq + 1e-9)
    scale = s_norm_sq / (1.0 + s_norm_sq) / s_norm
    v = scale * x
    return v

class primary_capsules(nn.Module) : 
    def __init__(self , number_of_capsules ,  input_dim) -> None:
        super().__init__()
        self.number_of_capsulse = number_of_capsules
        self.input_features = input_dim
    def forward(self , x ) :
        #  x : (batch , input_features)
        b , input_dim = x.shape
        if input_dim % self.number_of_capsulse ==0 : 
            k = input_dim//self.number_of_capsulse
        else : 
            print('the capsule numbers or the input feature number is not valid')
        x = x.reshape(b , self.number_of_capsulse , k ) 
        return x 


class secoundary_capsules(nn.Module):
    def __init__(self, n_caps_in, n_caps_out, caps_input_dim, caps_out_dim, n_routing) -> None:
        super().__init__()
        # Transformation matrices
        self.W = nn.Parameter(torch.randn(n_caps_in, caps_input_dim, caps_out_dim))
        self.n_caps_out = n_caps_out
        self.n_routing = n_routing

    def forward(self, x):
        # x: (batch, n_caps_in, caps_input_dim)
        batch_size, n_caps_in, caps_input_dim = x.shape

        # apply transformation
        x = x.unsqueeze(-1)  # (batch, n_caps_in, caps_input_dim, 1)
        W = self.W.unsqueeze(0)  # (1, n_caps_in, caps_input_dim, caps_out_dim)
        u_hat = torch.matmul(W.transpose(2, 3), x).squeeze(-1)  # (batch, n_caps_in, caps_out_dim)

        # initialize logits (bij)
        b = torch.zeros(batch_size, n_caps_in, self.n_caps_out, device=x.device)

        for r in range(self.n_routing):
            c = softmax(b, dim=-1)  # (batch, n_caps_in, n_caps_out)
            # weighted sum s_j
            s =  (c.unsqueeze(-1) * u_hat.unsqueeze(2)).sum(dim=1)  # (batch, n_caps_out, caps_out_dim)
            v = squash(s)
            if r < self.n_routing - 1:
                # agreement (dot product)
                # u_hat: (batch, n_caps_in, caps_out_dim)
                # v:     (batch, n_caps_out, caps_out_dim)
                v_expanded = v.unsqueeze(1)  # (batch, 1, n_caps_out, caps_out_dim)
                u_expanded = u_hat.unsqueeze(2)  # (batch, n_caps_in, 1, caps_out_dim)
                a = torch.sum(u_expanded * v_expanded, dim=-1)  # (batch, n_caps_in, n_caps_out)
                b = b + a

        return v

# مدل
class prediction_net(nn.Module):
    def __init__(self, d_in , n_input_caps , n_output_caps , in_caps_dim , out_caps_dim , n_routing = 3, lr=0.01):
        super().__init__()
        self.n_input_caps , self.n_output_caps , self.in_caps_dim , self.out_caps_dim = n_input_caps , n_output_caps , in_caps_dim , out_caps_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Improved feature extraction layers with batch norm and dropout
        self.layer1 = nn.Linear(d_in , 64 )
        self.layer2 = nn.Linear(64 , 32 )
        self.layer3 = nn.Linear(32 , n_input_caps*in_caps_dim )
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(64)
        self.layer_norm2 = nn.LayerNorm(32)
        
        self.primary_caps = primary_capsules(n_input_caps ,n_input_caps*in_caps_dim )
        self.secoundary_caps = secoundary_capsules(n_input_caps , n_output_caps , in_caps_dim , out_caps_dim , n_routing)
        
        # Improved final layer with residual-like connection
        self.final_layer = nn.Sequential(
            nn.Linear(n_output_caps*out_caps_dim , 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        
    def prediction(self , x ) : 
        #x : (batch , d_in)
        batch_size  , _= x.shape
        x = self.layer1(x)
        x = self.layer_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.relu(x)
        x = self.primary_caps(x)
        x = self.secoundary_caps(x)
        x = x.reshape(batch_size , -1)
        x = self.final_layer(x)
        return x 
    def forward(self, combined_embedded, label=None, status='test'):
        combined_embedded = torch.tensor(combined_embedded, dtype=torch.float, device=self.device)
        combined_embedded.requires_grad_(True)

        if status == 'train':
            label = torch.tensor(label, dtype=torch.float, device=self.device)
            self.optimizer.zero_grad()
            output = self.prediction(combined_embedded)
            loss = self.loss_fn(output, label)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            input_grad = combined_embedded.grad.detach().cpu().tolist()
            self.optimizer.step()
            result = {'grad' : input_grad}
            return result
        else:  # test
            output = self.prediction(combined_embedded)
            result = output.detach().cpu().tolist()
            result = {'prediction': result}
            return result
