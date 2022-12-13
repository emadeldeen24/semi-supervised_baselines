import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from torchvision import transforms

import os
import numpy as np
from .ts_augmentations import apply_transformation, DataTransform, DataTransform_SemiTime, cutPF


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, train_mode, ssl_method, augmentation):
        super(Load_Dataset, self).__init__()
        self.train_mode = train_mode
        self.ssl_method = ssl_method
        self.augmentation = augmentation

        X_train = dataset["samples"]
        y_train = dataset["labels"]


        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train).long()

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        self.x_data = X_train
        self.y_data = y_train

        self.len = X_train.shape[0]

        self.num_transformations = len(self.augmentation.split("_"))

    def __getitem__(self, index):
        if self.train_mode == "ssl" and self.ssl_method in ["simclr", "ts_tcc", "cpc"]:
            if self.ssl_method == "ts_tcc":  # TS-TCC has its own augmentations
                self.augmentation = "tsTcc_aug"
            elif self.ssl_method == "clsTran":
                self.augmentation = "permute_timeShift_scale_noise"
            transformed_samples = apply_transformation(self.x_data[index], self.augmentation)
            sample = {
                'transformed_samples': transformed_samples,
                'sample_ori': self.x_data[index].squeeze(-1)
            }

        elif self.train_mode == "ssl" and self.ssl_method in ["FixMatch", "DivideMix"]:
            w_aug, s_aug = DataTransform(self.x_data[index])
            sample = {
                'sample_ori': self.x_data[index].squeeze(-1),
                'class_labels': int(self.y_data[index]),
                'sample_w_aug': w_aug,
                'sample_s_aug': s_aug
            }

        elif self.train_mode == "ssl" and self.ssl_method == "SemiTime":
            x = self.x_data[index]

            t1, t2, t3, t4 = DataTransform_SemiTime(x)
            (t1_P, t1_F) = cutPF(t1)
            (t2_P, t2_F) = cutPF(t2)
            (t3_P, t3_F) = cutPF(t3)
            (t4_P, t4_F) = cutPF(t4)
            x_past_list = [t1_P, t2_P, t3_P, t4_P]
            # x_past = torch.cat(x_list_past, 0)
            x_future_list = [t1_F, t2_F, t3_F, t4_F]
            # x_future = torch.cat(x_list_future, 0)

            sample = {
                'sample_ori': self.x_data[index].squeeze(-1),
                'class_labels': int(self.y_data[index]),
                'x_past_list': x_past_list,
                'x_future_list': x_future_list
            }

            return sample


        elif self.train_mode == "ssl" and self.ssl_method == "clsTran":
            transformed_samples = apply_transformation(self.x_data[index], self.augmentation)
            order = np.random.randint(self.num_transformations)
            transformed_sample = transformed_samples[order]
            sample = {
                'transformed_samples': transformed_sample,
                'aux_labels': int(order),
                'sample_ori': self.x_data[index].squeeze(-1)
            }
        else:
            sample = {
                'sample_ori': self.x_data[index].squeeze(-1),
                'class_labels': int(self.y_data[index])
            }

        return sample

    def __len__(self):
        return self.len


def data_generator(data_path, data_percentage, dataset_configs, hparams, train_mode, ssl_method,
                   augmentation, oversample):
    batch_size = hparams["batch_size"]
    # original
    if data_percentage != "100":
        train_dataset = torch.load(os.path.join(data_path, f"train_{data_percentage}perc.pt"))
    else:
        train_dataset = torch.load(os.path.join(data_path, "train.pt"))

    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset, train_mode, ssl_method, augmentation)
    val_dataset = Load_Dataset(valid_dataset, train_mode, ssl_method, augmentation)
    test_dataset = Load_Dataset(test_dataset, train_mode, ssl_method, augmentation)

    if train_dataset.__len__() < batch_size:
        batch_size = 16
    if train_dataset.__len__() < 16:
        batch_size = train_dataset.__len__()
        # print(train_dataset.__len__())
    print(batch_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=False, num_workers=0)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                             shuffle=False, drop_last=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                             shuffle=False, drop_last=False, num_workers=0)

    num_classes = len(np.unique(train_dataset.y_data.numpy()))

    labeled_few_shot_dataset = torch.load(os.path.join(data_path, "train_5perc.pt"))
    labeled_few_shot_dataset = Load_Dataset(labeled_few_shot_dataset, train_mode, ssl_method, augmentation)
    labeled_few_shot_loader = torch.utils.data.DataLoader(dataset=labeled_few_shot_dataset, batch_size=batch_size,
                                                          shuffle=True, drop_last=False, num_workers=0)

    if train_mode == "ssl" and ssl_method in ["FixMatch", "SemiTime", "MeanTeacher", "DivideMix"]:
        return train_loader, val_loader, test_loader, num_classes, labeled_few_shot_loader
    else:
        return train_loader, val_loader, test_loader, num_classes
