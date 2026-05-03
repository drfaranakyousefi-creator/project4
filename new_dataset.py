import pandas as pd
import torch
import math
from torch.utils.data import DataLoader, Dataset


def filter_noisy_data(x, dataset_name):
    item_id = {
        'metavision': [220045, 220210, 220179, 220180, 220052, 220277],
        'carevue':    [211, 618, 52, 456, 676, 678, 646]
    }
    return x[x['itemid'].isin(item_id[dataset_name])].copy()


def extract_data_from_person(dataframe, W, dataset_name, target):
    N = 4 if dataset_name == 'metavision' else 5
    data, label, mask = [], [], []
    e = torch.zeros(N)
    x = torch.zeros(W, N)
    m = torch.zeros(W)
    s = 0

    for _, row in dataframe.iterrows():
        item_id = row['itemid']
        value = row['value_normalized']
        try:
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                continue
        except (ValueError, TypeError):
            continue

        if s >= W:
            s = 0
            m = torch.zeros(W)
            x = torch.zeros(W, N)

        if (item_id == 646) | (item_id == 220277):
            if target == 'spO2':
                if s > 0:
                    data.append(x.clone()); label.append(row['label_normalized']); mask.append(m.clone())
                    m = torch.zeros(W); x = torch.zeros(W, N); s = 0
            elif target == 'BP':
                e[0] = value; x[s, :] = e.clone(); m[s] = 1; s += 1
            elif target == 'RR':
                e[1] = value; x[s, :] = e.clone(); m[s] = 1; s += 1

        elif (item_id == 52) | (item_id == 220052):
            if target == 'BP':
                if s > 0:
                    data.append(x.clone()); label.append(row['label_normalized']); mask.append(m.clone())
                    m = torch.zeros(W); x = torch.zeros(W, N); s = 0
            else:
                e[0] = value; x[s, :] = e.clone(); m[s] = 1; s += 1

        elif (item_id == 618) | (item_id == 220210):
            if target == 'RR':
                if s > 0:
                    data.append(x.clone()); label.append(row['label_normalized']); mask.append(m.clone())
                    m = torch.zeros(W); x = torch.zeros(W, N); s = 0
            else:
                e[1] = value; x[s, :] = e.clone(); m[s] = 1; s += 1

        elif (item_id == 211) | (item_id == 220045):
            e[2] = value; x[s, :] = e.clone(); m[s] = 1; s += 1
        elif (item_id == 220179) | (item_id == 220180):
            e[3] = value; x[s, :] = e.clone(); m[s] = 1; s += 1
        elif item_id == 456:
            e[3] = value; x[s, :] = e.clone(); m[s] = 1; s += 1
        elif (item_id == 678) | (item_id == 676):
            e[4] = value; x[s, :] = e.clone(); m[s] = 1; s += 1

    if len(data) > 0:
        data  = torch.stack(data, dim=0)
        label = torch.tensor(label, dtype=torch.float)
        mask  = torch.stack(mask, dim=0)
    else:
        data, label, mask = None, None, None
    return data, label, mask


class data(Dataset):
    def __init__(self, data_in, label, mask):
        super().__init__()
        self.data, self.label, self.mask = data_in, label, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, :, :], self.label[index], self.mask[index, :]


class data_preparing:
    def __init__(self, data_frame, dataset_name, w, test_size, target, batch_size):

        N = 4 if dataset_name == 'metavision' else 5

        target_itemids = {
            'spO2': {'metavision': 220277, 'carevue': 646},
            'BP':   {'metavision': 220052, 'carevue': 52},
            'RR':   {'metavision': 220210, 'carevue': 618},
        }
        label_itemid = target_itemids[target][dataset_name]

        if dataset_name == 'metavision':
            col_map = {220052: 0, 220210: 1, 220045: 2, 220179: 3, 220180: 3}
        else:
            col_map = {52: 0, 618: 1, 211: 2, 456: 3, 676: 4, 678: 4}

        filtered = filter_noisy_data(data_frame, dataset_name).copy()

        # جمع‌آوری مقادیر feature ها برای normalize
        col_values = {i: [] for i in range(N)}
        for _, row in filtered.iterrows():
            item_id = row['itemid']
            value = row['value']
            try:
                value = float(value)
                if math.isnan(value) or math.isinf(value):
                    continue
            except (ValueError, TypeError):
                continue
            if item_id in col_map:
                col_values[col_map[item_id]].append(value)

        # محاسبه mean و std هر feature
        self.feature_mean = torch.zeros(N)
        self.feature_std  = torch.ones(N)
        for i in range(N):
            if len(col_values[i]) > 0:
                vals = torch.tensor(col_values[i], dtype=torch.float)
                self.feature_mean[i] = vals.mean()
                self.feature_std[i]  = vals.std() + 1e-4

        # normalize فقط feature ها - label خام میمونه
        def normalize_value(item_id, value):
            if item_id in col_map:
                c = col_map[item_id]
                return (value - self.feature_mean[c].item()) / self.feature_std[c].item()
            return value

        filtered['value_normalized'] = filtered.apply(
            lambda row: normalize_value(row['itemid'], self._safe_float(row['value'])), axis=1
        )
        # label_normalized = مقدار خام بدون هیچ تغییری
        filtered['label_normalized'] = filtered.apply(
            lambda row: self._safe_float(row['value']) if row['itemid'] == label_itemid else 0.0, axis=1
        )

        # ساختاربندی فرد به فرد
        all_data, all_labels, all_masks = [], [], []
        for subject_id in filtered['subject_id'].unique():
            subject_df = filtered[filtered['subject_id'] == subject_id]
            d, l, m = extract_data_from_person(subject_df, w, dataset_name, target)
            if d is not None:
                all_data.append(d); all_labels.append(l); all_masks.append(m)

        x     = torch.concat(all_data,   dim=0)
        y     = torch.concat(all_labels, dim=0)
        masks = torch.concat(all_masks,  dim=0)

        # shuffle قبل از split
        perm = torch.randperm(x.shape[0])
        x, y, masks = x[perm], y[perm], masks[perm]

        train_number  = int((1 - test_size) * x.shape[0])
        train_dataset = data(x[:train_number], y[:train_number], masks[:train_number])
        test_dataset  = data(x[train_number:], y[train_number:], masks[train_number:])

        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.test_loader  = DataLoader(test_dataset,  batch_size, shuffle=True)

        # فقط feature stats - label normalize نشده
        self.stats = {
            'feature_mean': self.feature_mean,
            'feature_std':  self.feature_std,
        }

    @staticmethod
    def _safe_float(value):
        try:
            v = float(value)
            return 0.0 if (math.isnan(v) or math.isinf(v)) else v
        except (ValueError, TypeError):
            return 0.0
