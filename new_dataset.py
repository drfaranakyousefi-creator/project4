import pandas as pd
import torch
import random
from torch.utils.data import  TensorDataset , DataLoader , Dataset
import math
# مسیر فایل خود را اینجا وارد کنید


def filter_noisy_data(x , dataset_name): 
    item_id = {
        'metavision' : [  
                220045,  # Heart Rate
                220210,  # Respiratory Rate
                220179, 220180,  # Non-invasive BP Mean
                220052,  # Arterial BP Mean
                220277   # SpO2 (هدف پیش‌بینی) 
        ],
        'carevue' : [
                211,   # ضربان قلب
                618,   # نرخ تنفس
                52,    # متوسط فشار خون شریانی
                456,          # متوسط فشار خون NBP
                676, 678,     # دما
                646   # SpO2 (هدف پیش‌بینی)
        ]
    }

    filtered_df = x[x['itemid'].isin(item_id[dataset_name])].copy()
    return filtered_df

def extract_data_from_person(dataframe , W , dataset_name , target) : 
    if dataset_name == 'metavision'  : 
        N = 4 
    else : 
        N = 5
    data = []
    label = []
    e = torch.zeros(N)
    x = torch.zeros(W , N)
    mask = []
    s=0
    m = torch.zeros(W)
    for index, row in dataframe.iterrows():
        item_id = row['itemid']
        value = row['value']
        try:
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                continue   # رد کردن مقادیر نامعتبر
        except (ValueError, TypeError):
            continue

    #data _order for metavision : (heart rate , respiratory rate , Non-invasive BP Mean , Arterial BP Mean)
    #data order for 'carevue' is (heart rate, respiratory rate, arterial BP mean, NBP mean, temperature)
        if s >= W : 
            s = 0
            m = torch.zeros(W)
            x = torch.zeros(W , N)
            
        if (item_id == 646) |  (item_id==220277) :  # spo2
            if target == 'spO2' :  
                if  s > 0 :
                    data.append(x)
                    label.append(value)
                    mask.append(m)
                    m = torch.zeros(W)
                    x = torch.zeros(W , N)
                    s=0
            elif target == 'BP'  : 
                idx =  0
                e[idx] = value
                x[s , :] = e
                m[s]=1
                s+=1

            elif target == 'RR' : 
                idx=1
                e[idx] = value
                x[s , :] = e
                m[s]=1
                s+=1

        elif (item_id==52) | (item_id == 220052)  :
            if target == 'BP' : 
                if  s > 0 :
                    data.append(x)
                    label.append(value)
                    mask.append(m)
                    m = torch.zeros(W)
                    x = torch.zeros(W , N)
                    s=0
            else : 
                idx=0
                e[idx] = value
                x[s , :] = e
                m[s]=1
                s+=1


        elif(item_id==618) | (item_id == 220210) :#RR
            if target == 'RR' : 
                if  s > 0 :
                    data.append(x)
                    label.append(value)
                    mask.append(m)
                    m = torch.zeros(W)
                    x = torch.zeros(W , N)
                    s=0
            else : 
                idx = 1
                e[idx] = value
                x[s , :] = e
                m[s]=1
                s+=1  

        elif (item_id == 211) |  (item_id==220045) :   
            idx = 2
            e[idx] = value
            x[s , :] = e
            m[s]=1
            s+=1

        elif ((item_id == 220179 ) |  (item_id==220180)): #metavision
            idx = 3
            e[idx] = value
            x[s , :] = e
            m[s]=1
            s+=1

        elif(item_id==456)  :  #carevue
            idx = 3
            e[idx] = value
            x[s , :] = e
            m[s]=1
            s+=1

        elif ((item_id==678 ) | (item_id == 676))  : 
            idx = 4
            e[idx] = value
            x[s , :] = e
            m[s]=1
            s+=1           


    if len(data) > 0: 
        data = torch.stack(data , dim=0)
        label = torch.tensor(label)
        mask = torch.stack(mask , dim=0)

    else : 
        data , label , mask = None  , None , None 

    return data , label , mask


  

def extract_data(dataset_name , df_chartevents , w ,  target , normalize=True) : 
    totol_subject_ids  = df_chartevents['subject_id'].unique()
    all_user_data  = [] 
    all_labels = [] 
    all_mask = []
    for subject_id in totol_subject_ids : 
        subject_data = df_chartevents[df_chartevents['subject_id'] == subject_id]
        filtered_df = filter_noisy_data( subject_data, dataset_name)
        data , label , mask= extract_data_from_person(filtered_df  , w , dataset_name , target )
        if label != None : 
            all_labels.append(label)
            all_user_data.append(data)
            all_mask.append(mask)
        #the function will return data of torch type tensor and shape is : (sample of this user , 2 , w , N)
        # N is 5 for 'carevue' and 4 for 'metavision'
    data = torch.concat(all_user_data , dim=0)
    if normalize : 
        mean = torch.mean(data)
        std = torch.std(data)
        data = (data - mean)/(std + 1e-4)
    return data , torch.concat(all_labels , dim=0),torch.concat(all_mask , dim = 0)


class data(Dataset) : 
    def __init__(self , data_in , label , mask)  :
        super().__init__()
        self.data , self.label, self.mask  = data_in , label , mask
    def __len__(self  ) :
        return len(self.data)
    def __getitem__(self, index ) :
        x = self.data[index , : , :]
        y = self.label[index]
        mask = self.mask[index , : ]
        return x , y , mask



        


class data_preparing : 
    def __init__(self ,data_frame , dataset_name , w , test_size , target ,batch_size  ) :  #target can be spo2 or BP or RR:raspiratory rate  
        x , y  , mask = extract_data(dataset_name , data_frame , w , target)
        train_number = int((1-test_size)*x.shape[0])
        train_dataset = data(x[:train_number , : , : ] , y[:train_number], mask[:train_number , : ])
        test_dataset = data(x[train_number: , : , : ] , y[train_number:] , mask[train_number: , :])
        self.train_loader = DataLoader(train_dataset , batch_size , shuffle=True)
        self.test_loader = DataLoader(test_dataset ,  batch_size , shuffle=True)

