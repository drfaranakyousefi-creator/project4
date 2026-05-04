import torch 
from server_net import prediction_net
import json


class Transmitter:
    def __init__(self, d_in, device, lr, label_mean=0.0, label_std=1.0):
        self.model = prediction_net(
            d_in,
            n_input_caps=4,
            n_output_caps=3,
            in_caps_dim=6,
            out_caps_dim=8,
            lr=lr,
            label_mean=float(label_mean),
            label_std=float(label_std),
        )
        self.device = device

    def update_learning_rate(self, new_lr):
        for param_group in self.model.optimizer.param_groups:
            param_group['lr'] = new_lr

    def data_to_json(self, x, label, status):
        x_copy = x.detach().cpu().tolist()
        label_copy = label.detach().cpu().tolist()
        if status == 'train':
            data = {
                'prediction_iput': x_copy,
                'label': label_copy,
                'status': status
            }
        elif status == 'test':
            data = {
                'prediction_iput': x_copy,
                'label': [],
                'status': status
            }
        return json.dumps(data)

    def send_data(self, x, label, status):
        data_transfer_to_server = self.data_to_json(x, label, status)
        server_recieves_data = json.loads(data_transfer_to_server)
        combined_embedded = server_recieves_data['prediction_iput']
        label = server_recieves_data['label']
        status = server_recieves_data['status']
        result = self.model(combined_embedded, label, status)
        data_back_to_client = json.dumps(result)
        data_recive_in_client = json.loads(data_back_to_client)
        if status == 'train':
            grad = data_recive_in_client['grad']
            return torch.tensor(grad).to(self.device)
        elif status == 'test':
            prediction = torch.tensor(data_recive_in_client['prediction']).to(self.device)
            return prediction
