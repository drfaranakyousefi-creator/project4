import pandas as pd
import torch
import math
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────
# فیلتر کردن داده های نویزی
# ─────────────────────────────────────────────
def filter_noisy_data(x, dataset_name):
    item_id = {
        'metavision': [
            220045,          # Heart Rate
            220210,          # Respiratory Rate
            220179, 220180,  # Non-invasive BP Mean
            220052,          # Arterial BP Mean
            220277           # SpO2
        ],
        'carevue': [
            211,             # ضربان قلب
            618,             # نرخ تنفس
            52,              # متوسط فشار خون شریانی
            456,             # متوسط فشار خون NBP
            676, 678,        # دما
            646              # SpO2
        ]
    }
    return x[x['itemid'].isin(item_id[dataset_name])].copy()


# ─────────────────────────────────────────────
# استخراج خام داده های یک فرد (بدون نرمال‌سازی)
# خروجی: list of (value, item_id, col_index)
# ─────────────────────────────────────────────
def extract_raw_values(dataframe, dataset_name, target):
    """
    همه مقادیر خام یک فرد رو برمیگردونه به صورت:
      feature_values: dict  col_index -> list of float
      label_values:  list of float
    """
    if dataset_name == 'metavision':
        N = 4
    else:
        N = 5

    # mapping itemid -> col index  (target col = -1 یعنی label)
    col_map = {}
    if dataset_name == 'metavision':
        # heart rate=2, RR=1, NIBP=3, ArtBP=0, SpO2=label
        col_map = {
            220045: 2,   # Heart Rate
            220210: 1,   # Respiratory Rate
            220179: 3,   # NIBP
            220180: 3,
            220052: 0,   # Arterial BP
            220277: -1,  # SpO2 label
        }
    else:
        col_map = {
            211:  2,   # Heart Rate
            618:  1,   # Respiratory Rate
            52:   0,   # Arterial BP
            456:  3,   # NBP
            676:  4,   # Temperature
            678:  4,
            646:  -1,  # SpO2 label
        }

    # اگه target چیز دیگه‌ایه، col آن feature label میشه
    target_itemid = {
        'spO2':   {'metavision': 220277, 'carevue': 646},
        'BP':     {'metavision': 220052, 'carevue': 52},
        'RR':     {'metavision': 220210, 'carevue': 618},
    }
    label_id = target_itemid[target][dataset_name]

    # برای هر col یه لیست از مقادیر معتبر
    feature_values = {i: [] for i in range(N)}
    label_values = []

    for _, row in dataframe.iterrows():
        item_id = row['itemid']
        value = row['value']
        try:
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                continue
        except (ValueError, TypeError):
            continue

        if item_id == label_id:
            label_values.append(value)
        elif item_id in col_map and col_map[item_id] >= 0:
            feature_values[col_map[item_id]].append(value)

    return feature_values, label_values, N


# ─────────────────────────────────────────────
# ساختاربندی داده‌های یک فرد (بعد از نرمال‌سازی)
# ─────────────────────────────────────────────
def extract_data_from_person(dataframe, W, dataset_name, target):
    """
    همون منطق اصلی - داده‌های یک فرد رو ساختاربندی میکنه
    ورودی dataframe باید داده‌های normalize شده داشته باشه
    """
    if dataset_name == 'metavision':
        N = 4
    else:
        N = 5

    data = []
    label = []
    e = torch.zeros(N)
    x = torch.zeros(W, N)
    mask = []
    s = 0
    m = torch.zeros(W)

    for index, row in dataframe.iterrows():
        item_id = row['itemid']
        value = row['value_normalized']  # از ستون normalize شده استفاده میکنیم
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

        if (item_id == 646) | (item_id == 220277):  # spo2
            if target == 'spO2':
                if s > 0:
                    data.append(x.clone())
                    label.append(row['label_normalized'])
                    mask.append(m.clone())
                    m = torch.zeros(W)
                    x = torch.zeros(W, N)
                    s = 0
            elif target == 'BP':
                idx = 0
                e[idx] = value
                x[s, :] = e.clone()
                m[s] = 1
                s += 1
            elif target == 'RR':
                idx = 1
                e[idx] = value
                x[s, :] = e.clone()
                m[s] = 1
                s += 1

        elif (item_id == 52) | (item_id == 220052):
            if target == 'BP':
                if s > 0:
                    data.append(x.clone())
                    label.append(row['label_normalized'])
                    mask.append(m.clone())
                    m = torch.zeros(W)
                    x = torch.zeros(W, N)
                    s = 0
            else:
                idx = 0
                e[idx] = value
                x[s, :] = e.clone()
                m[s] = 1
                s += 1

        elif (item_id == 618) | (item_id == 220210):  # RR
            if target == 'RR':
                if s > 0:
                    data.append(x.clone())
                    label.append(row['label_normalized'])
                    mask.append(m.clone())
                    m = torch.zeros(W)
                    x = torch.zeros(W, N)
                    s = 0
            else:
                idx = 1
                e[idx] = value
                x[s, :] = e.clone()
                m[s] = 1
                s += 1

        elif (item_id == 211) | (item_id == 220045):
            idx = 2
            e[idx] = value
            x[s, :] = e.clone()
            m[s] = 1
            s += 1

        elif (item_id == 220179) | (item_id == 220180):  # metavision
            idx = 3
            e[idx] = value
            x[s, :] = e.clone()
            m[s] = 1
            s += 1

        elif item_id == 456:  # carevue
            idx = 3
            e[idx] = value
            x[s, :] = e.clone()
            m[s] = 1
            s += 1

        elif (item_id == 678) | (item_id == 676):
            idx = 4
            e[idx] = value
            x[s, :] = e.clone()
            m[s] = 1
            s += 1

    if len(data) > 0:
        data = torch.stack(data, dim=0)
        label = torch.tensor(label, dtype=torch.float)
        mask = torch.stack(mask, dim=0)
    else:
        data, label, mask = None, None, None

    return data, label, mask


# ─────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────
class data(Dataset):
    def __init__(self, data_in, label, mask):
        super().__init__()
        self.data, self.label, self.mask = data_in, label, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index, :, :]
        y = self.label[index]
        mask = self.mask[index, :]
        return x, y, mask


# ─────────────────────────────────────────────
# کلاس اصلی آماده‌سازی داده
# ─────────────────────────────────────────────
class data_preparing:
    """
    مرحله 1: نرمال‌سازی هر ستون feature و label به صورت مجزا
             روی کل داده‌های همه افراد با هم
             مقادیر mean و std به عنوان attribute نگه داشته میشن

    مرحله 2: ساختاربندی داده‌ها فرد به فرد با حلقه جداگانه
             تا داده‌های افراد با هم قاطی نشن
    """

    def __init__(self, data_frame, dataset_name, w, test_size, target, batch_size):

        if dataset_name == 'metavision':
            N = 4
        else:
            N = 5

        # ── مرحله 1: نرمال‌سازی ──────────────────────────────
        # itemid های feature و label برای هر dataset
        target_itemids = {
            'spO2':  {'metavision': 220277, 'carevue': 646},
            'BP':    {'metavision': 220052, 'carevue': 52},
            'RR':    {'metavision': 220210, 'carevue': 618},
        }
        label_itemid = target_itemids[target][dataset_name]

        # col mapping: itemid -> col index
        if dataset_name == 'metavision':
            col_map = {220052: 0, 220210: 1, 220045: 2, 220179: 3, 220180: 3}
        else:
            col_map = {52: 0, 618: 1, 211: 2, 456: 3, 676: 4, 678: 4}

        filtered = filter_noisy_data(data_frame, dataset_name).copy()

        # جمع‌آوری مقادیر هر ستون از همه افراد
        col_values = {i: [] for i in range(N)}
        label_values_raw = []

        for _, row in filtered.iterrows():
            item_id = row['itemid']
            value = row['value']
            try:
                value = float(value)
                if math.isnan(value) or math.isinf(value):
                    continue
            except (ValueError, TypeError):
                continue

            if item_id == label_itemid:
                label_values_raw.append(value)
            elif item_id in col_map:
                col_values[col_map[item_id]].append(value)

        # محاسبه mean و std هر ستون
        self.feature_mean = torch.zeros(N)
        self.feature_std = torch.ones(N)
        for i in range(N):
            if len(col_values[i]) > 0:
                vals = torch.tensor(col_values[i], dtype=torch.float)
                self.feature_mean[i] = vals.mean()
                self.feature_std[i] = vals.std() + 1e-4

        # محاسبه mean و std label
        label_tensor = torch.tensor(label_values_raw, dtype=torch.float)
        self.label_mean = label_tensor.mean()
        self.label_std = label_tensor.std() + 1e-4

        # اعمال نرمال‌سازی روی dataframe
        def normalize_value(item_id, value):
            if item_id == label_itemid:
                return (value - self.label_mean.item()) / self.label_std.item()
            elif item_id in col_map:
                c = col_map[item_id]
                return (value - self.feature_mean[c].item()) / self.feature_std[c].item()
            return value

        filtered['value_normalized'] = filtered.apply(
            lambda row: normalize_value(row['itemid'], self._safe_float(row['value'])),
            axis=1
        )
        filtered['label_normalized'] = filtered.apply(
            lambda row: (self._safe_float(row['value']) - self.label_mean.item()) / self.label_std.item()
            if row['itemid'] == label_itemid else 0.0,
            axis=1
        )

        # ── مرحله 2: ساختاربندی فرد به فرد ──────────────────
        all_data = []
        all_labels = []
        all_masks = []

        for subject_id in filtered['subject_id'].unique():
            subject_df = filtered[filtered['subject_id'] == subject_id]
            d, l, m = extract_data_from_person(subject_df, w, dataset_name, target)
            if d is not None:
                all_data.append(d)
                all_labels.append(l)
                all_masks.append(m)

        x = torch.concat(all_data, dim=0)
        y = torch.concat(all_labels, dim=0)
        masks = torch.concat(all_masks, dim=0)
        perm = torch.randperm(x.shape[0])
        x = x[perm]
        y = y[perm]
        masks = masks[perm]
        # تقسیم train/test
        train_number = int((1 - test_size) * x.shape[0])
        train_dataset = data(x[:train_number], y[:train_number], masks[:train_number])
        test_dataset = data(x[train_number:], y[train_number:], masks[train_number:])

        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

        # stats برای denormalize کردن پیش‌بینی‌ها در صورت نیاز
        self.stats = {
            'feature_mean': self.feature_mean,
            'feature_std':  self.feature_std,
            'label_mean':   self.label_mean,
            'label_std':    self.label_std,
        }

    @staticmethod
    def _safe_float(value):
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return 0.0
            return v
        except (ValueError, TypeError):
            return 0.0
