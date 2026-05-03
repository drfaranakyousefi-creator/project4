import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax


def squash(x, dim=-1):
    s_norm_sq = torch.sum(x ** 2, dim=dim, keepdim=True)
    s_norm = torch.sqrt(s_norm_sq + 1e-9)
    scale = s_norm_sq / (1.0 + s_norm_sq) / s_norm
    return scale * x


class primary_capsules(nn.Module):
    def __init__(self, number_of_capsules, input_dim) -> None:
        super().__init__()
        self.number_of_capsulse = number_of_capsules

    def forward(self, x):
        b, input_dim = x.shape
        if input_dim % self.number_of_capsulse == 0:
            k = input_dim // self.number_of_capsulse
        else:
            print('the capsule numbers or the input feature number is not valid')
        return x.reshape(b, self.number_of_capsulse, k)


class secoundary_capsules(nn.Module):
    def __init__(self, n_caps_in, n_caps_out, caps_input_dim, caps_out_dim, n_routing) -> None:
        super().__init__()
        self.n_caps_out = n_caps_out
        self.n_routing = n_routing

        # درست: هر capsule خروجی W جداگانه داره
        # شکل: (n_caps_out, n_caps_in, caps_input_dim, caps_out_dim)
        self.W = nn.Parameter(
            torch.randn(n_caps_out, n_caps_in, caps_input_dim, caps_out_dim) * 0.1
        )

    def forward(self, x):
        # x: (batch, n_caps_in, caps_input_dim)
        batch_size, n_caps_in, caps_input_dim = x.shape

        # x: (batch, 1, n_caps_in, caps_input_dim, 1)
        x = x.unsqueeze(1).unsqueeze(-1)

        # W: (1, n_caps_out, n_caps_in, caps_input_dim, caps_out_dim)
        W = self.W.unsqueeze(0)

        # u_hat: (batch, n_caps_out, n_caps_in, caps_out_dim)
        u_hat = torch.matmul(W.transpose(-2, -1), x).squeeze(-1)

        # routing
        b = torch.zeros(batch_size, self.n_caps_out, n_caps_in, device=x.device)

        for r in range(self.n_routing):
            # c: (batch, n_caps_out, n_caps_in)
            c = softmax(b, dim=1)

            # s: (batch, n_caps_out, caps_out_dim)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=2)

            # v: (batch, n_caps_out, caps_out_dim)
            v = squash(s)

            if r < self.n_routing - 1:
                # agreement: (batch, n_caps_out, n_caps_in)
                a = torch.sum(
                    v.unsqueeze(2) * u_hat,
                    dim=-1
                )
                b = b + a

        return v


class prediction_net(nn.Module):
    def __init__(self, d_in, n_input_caps, n_output_caps, in_caps_dim, out_caps_dim, n_routing=3, lr=0.01):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.layer1 = nn.Linear(d_in, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, n_input_caps * in_caps_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(64)
        self.layer_norm2 = nn.LayerNorm(32)

        self.primary_caps = primary_capsules(n_input_caps, n_input_caps * in_caps_dim)
        self.secoundary_caps = secoundary_capsules(
            n_input_caps, n_output_caps, in_caps_dim, out_caps_dim, n_routing
        )

        self.final_layer = nn.Sequential(
            nn.Linear(n_output_caps * out_caps_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        # دو پارامتر قابل آموزش برای scale و shift خروجی
        # y_pred_final = y_pred * output_scale + output_shift
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_shift = nn.Parameter(torch.tensor(0.0))

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def prediction(self, x):
        batch_size, _ = x.shape

        x = self.layer1(x)
        x = self.layer_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        # relu قبل از capsule حذف شده
        x = self.primary_caps(x)
        x = self.secoundary_caps(x)
        x = x.reshape(batch_size, -1)
        x = self.final_layer(x)

        # y_pred * a + b
        x = x * self.output_scale + self.output_shift

        return x

    def forward(self, combined_embedded, label=None, status='test'):
        if not isinstance(combined_embedded, torch.Tensor):
            combined_embedded = torch.tensor(combined_embedded, dtype=torch.float, device=self.device)
        else:
            combined_embedded = combined_embedded.to(self.device).float()
        combined_embedded.requires_grad_(True)

        if status == 'train':
            label = torch.tensor(label, dtype=torch.float, device=self.device)
            self.optimizer.zero_grad()
            output = self.prediction(combined_embedded)
            loss = self.loss_fn(output.squeeze(-1), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            input_grad = combined_embedded.grad.detach().cpu().tolist()
            self.optimizer.step()
            return {'grad': input_grad}
        else:
            output = self.prediction(combined_embedded)
            return {'prediction': output.detach().cpu().tolist()}
