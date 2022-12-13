import torch
import torch.nn as nn
import numpy as np

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from models.loss import NTXentLoss, softmax_mse_loss , SemiLoss
import torch.nn.functional as F
from models.TC import TC
from models.helpers import proj_head


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class simclr(Algorithm):
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams, device):
        super(simclr, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.temporal_encoder = backbone_temporal(hparams)
        self.proj_head = proj_head(configs, hparams)
        self.network = nn.Sequential(self.feature_extractor, self.proj_head)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )
        self.hparams = hparams
        self.contrastive_loss = NTXentLoss(device, hparams["batch_size"], 0.2, True)

    def update(self, samples):
        # ====== Data =====================
        aug1 = samples["transformed_samples"][0]
        aug2 = samples["transformed_samples"][1]

        self.optimizer.zero_grad()

        features1 = self.feature_extractor(aug1)
        z1 = self.proj_head(features1)

        features2 = self.feature_extractor(aug2)
        z2 = self.proj_head(features2)

        # normalize projection feature vectors
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Cross-Entropy loss
        loss = self.contrastive_loss(z1, z2)

        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.proj_head]


class cpc(Algorithm):
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams, device):
        super(cpc, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.temporal_encoder = backbone_temporal(hparams)
        self.classifier = classifier(configs, hparams)
        self.network = nn.Sequential(self.feature_extractor, self.temporal_encoder, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )

        self.hparams = hparams

        self.num_channels = hparams["num_channels"]
        self.hid_dim = hparams["hid_dim"]
        self.timestep = hparams["timesteps"]
        self.Wk = nn.ModuleList([nn.Linear(self.hid_dim, self.num_channels) for _ in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device

        self.lstm = nn.LSTM(self.num_channels, self.hid_dim, bidirectional=False, batch_first=True)

    def update(self, samples):
        # ====== Data =====================
        data = samples['sample_ori'].float()

        self.optimizer.zero_grad()

        # Src original features
        features = self.feature_extractor(data)
        seq_len = features.shape[2]
        features = features.transpose(1, 2)

        batch = self.hparams["batch_size"]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick timesteps

        loss = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = features[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = features[:, :t_samples + 1, :]

        output1, _ = self.lstm(forward_seq)
        c_t = output1[:, t_samples, :].view(batch, self.hid_dim)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            loss += torch.sum(torch.diag(self.lsoftmax(total)))
        loss /= -1. * batch * self.timestep

        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.classifier]


class ts_tcc(Algorithm):
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams, device):
        super(ts_tcc, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.temporal_encoder = backbone_temporal(hparams)
        self.classifier = classifier(configs, hparams)
        self.temporal_contr_model = TC(hparams, device)

        self.network = nn.Sequential(self.feature_extractor, self.temporal_encoder,
                                     self.classifier, self.temporal_contr_model)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )
        self.hparams = hparams
        self.contrastive_loss = NTXentLoss(device, hparams["batch_size"], 0.2, True)

    def update(self, samples):
        # ====== Data =====================
        aug1 = samples["transformed_samples"][0]
        aug2 = samples["transformed_samples"][1]

        self.optimizer.zero_grad()

        features1 = self.feature_extractor(aug1)
        features2 = self.feature_extractor(aug2)

        # normalize projection feature vectors
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        temp_cont_loss1, temp_cont_lstm_feat1 = self.temporal_contr_model(features1, features2)
        temp_cont_loss2, temp_cont_lstm_feat2 = self.temporal_contr_model(features2, features1)

        # Cross-Entropy loss
        loss = temp_cont_loss1 + temp_cont_loss2 + \
               0.7 * self.contrastive_loss(temp_cont_lstm_feat1, temp_cont_lstm_feat2)

        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.classifier]


class clsTran(Algorithm):
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams, device):
        super(clsTran, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.temporal_encoder = backbone_temporal(hparams)
        self.classifier = nn.Linear(hparams["clf"], configs.num_clsTran_tasks)
        self.network = nn.Sequential(self.feature_extractor, self.temporal_encoder, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )

        self.hparams = hparams

    def update(self, samples):
        # ====== Data =====================
        data = samples["transformed_samples"].float()
        labels = samples["aux_labels"].long()

        self.optimizer.zero_grad()

        features = self.feature_extractor(data)
        features = features.flatten(1, 2)

        logits = self.classifier(features)

        # Cross-Entropy loss
        loss = self.cross_entropy(logits, labels)

        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.classifier]


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


class FixMatch(Algorithm):
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams, device):
        super(FixMatch, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.temporal_encoder = backbone_temporal(hparams)
        self.classifier = classifier(configs, hparams)
        self.network = nn.Sequential(self.feature_extractor, self.temporal_encoder, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )

        self.hparams = hparams
        self.device = device

        self.mu = 1
        self.threshold = 0.95
        self.T = 1
        self.lambda_u = 1

    def update(self, samples, fl_samples):
        # ====== Data =====================
        data = samples["sample_ori"].float()
        data_w_aug = samples["sample_w_aug"].float()
        data_s_aug = samples["sample_s_aug"].float()

        fl_data = fl_samples["sample_ori"].float()  # fl for few labeled
        fl_labels = fl_samples["class_labels"].long()

        self.optimizer.zero_grad()

        fl_batch_size = fl_data.shape[0]
        inputs = interleave(torch.cat((fl_data, data_w_aug, data_s_aug)), 1).to(self.device)

        features = self.feature_extractor(inputs)
        features = features.flatten(1, 2)
        logits = self.classifier(features)

        logits = de_interleave(logits, 1)

        logits_fl = logits[:fl_batch_size]
        logits_u_w, logits_u_s = logits[fl_batch_size:].chunk(2)
        del logits

        Lx = F.cross_entropy(logits_fl, fl_labels, reduction='mean')

        pseudo_label = torch.softmax(logits_u_w.detach() / self.T, dim=-1)

        max_probs, ps_data_u = torch.max(pseudo_label, dim=-1)

        mask = max_probs.ge(self.threshold).float()

        Lu = (F.cross_entropy(logits_u_s, ps_data_u, reduction='none') * mask).mean()

        loss = Lx + self.lambda_u * Lu

        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.classifier]


class SemiTime(Algorithm):
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams, device):
        super(SemiTime, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.temporal_encoder = backbone_temporal(hparams)
        self.classifier = classifier(configs, hparams)
        self.relation_head = torch.nn.Sequential(
            torch.nn.Linear(configs.features_len * 2 * configs.final_out_channels, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 2))

        self.network = nn.Sequential(self.feature_extractor, self.temporal_encoder, self.classifier, self.relation_head)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )

        self.hparams = hparams
        self.device = device

        # self.CE = nn.CrossEntropyLoss()

    def update(self, samples, fl_samples):
        # ====== Data =====================
        data = samples["sample_ori"].float()
        data_past = samples["x_past_list"] #.float()
        data_future = samples["x_future_list"] #.float()

        fl_data = fl_samples["sample_ori"].float()  # fl for few labeled
        fl_labels = fl_samples["class_labels"].long()

        self.optimizer.zero_grad()

        fl_features = self.feature_extractor(fl_data)
        fl_features = fl_features.flatten(1, 2)
        fl_logits = self.classifier(fl_features)
        Lx = F.cross_entropy(fl_logits, fl_labels, reduction='mean')

        ######################################################
        K = 4  # number of augmentations
        data_past = torch.cat(data_past, 0).float()
        data_future = torch.cat(data_future, 0).float()

        features_P = self.feature_extractor(data_past.transpose(1,2)).flatten(1, 2)
        features_F = self.feature_extractor(data_future.transpose(1,2)).flatten(1, 2)

        # aggregation function
        relation_pairs, targets = self.aggregate(features_P, features_F, K)
        targets = targets.long()

        # forward pass (relation head)
        score = self.relation_head(relation_pairs).squeeze()
        # cross-entropy loss and backward

        # output = torch.nn.Sigmoid()(score)
        semi_loss = self.cross_entropy(score, targets)

        loss = Lx + semi_loss

        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.classifier]

    def aggregate(self, features_P, features_F, K):
        relation_pairs_list = list()
        targets_list = list()
        size = int(features_P.shape[0] / K)
        shifts_counter = 1
        for index_1 in range(0, size * K, size):
            for index_2 in range(index_1 + size, size * K, size):
                # Using the 'cat' aggregation function by default
                pos1 = features_P[index_1:index_1 + size]
                pos2 = features_F[index_2:index_2 + size]
                pos_pair = torch.cat([pos1, pos2], 1)  # (batch_size, fz*2)

                # Shuffle without collisions by rolling the mini-batch (negatives)
                neg1 = torch.roll(features_F[index_2:index_2 + size], shifts=shifts_counter, dims=0)
                neg_pair1 = torch.cat([pos1, neg1], 1)  # (batch_size, fz*2)

                relation_pairs_list.append(pos_pair)
                relation_pairs_list.append(neg_pair1)

                targets_list.append(torch.ones(size, dtype=torch.float32).cuda())
                targets_list.append(torch.zeros(size, dtype=torch.float32).cuda())

                shifts_counter += 1
                if (shifts_counter >= size):
                    shifts_counter = 1  # avoid identity pairs
        relation_pairs = torch.cat(relation_pairs_list, 0).cuda()  # K(K-1) * (batch_size, fz*2)
        targets = torch.cat(targets_list, 0).cuda()
        return relation_pairs, targets


class MeanTeacher(Algorithm):
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams, device):
        super(MeanTeacher, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.temporal_encoder = backbone_temporal(hparams)
        self.classifier = classifier(configs, hparams)
        self.network = nn.Sequential(self.feature_extractor, self.temporal_encoder, self.classifier)

        self.ema_feature_extractor = backbone_fe(configs)
        self.ema_classifier = classifier(configs, hparams)

        for param in self.ema_feature_extractor.parameters():
            param.detach_()
        for param in self.ema_classifier.parameters():
            param.detach_()

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )

        self.hparams = hparams
        self.device = device

        self.ema_decay = 0.999
        self.cons_wt = 0.1 # 1.0

    def update(self, samples, fl_samples):
        # ====== Data =====================
        data = samples["sample_ori"].float()
        fl_data = fl_samples["sample_ori"].float()
        fl_labels = fl_samples["class_labels"].long()

        self.optimizer.zero_grad()

        # the labeled data part
        fl_features = self.feature_extractor(fl_data)
        fl_features = fl_features.flatten(1, 2)
        fl_logits = self.classifier(fl_features)
        Lx = F.cross_entropy(fl_logits, fl_labels, reduction='mean')

        ema_fl_features = self.ema_feature_extractor(fl_data)
        ema_fl_features = ema_fl_features.flatten(1, 2)
        ema_fl_logits = self.ema_classifier(ema_fl_features)
        ema_Lx = F.cross_entropy(ema_fl_logits, fl_labels, reduction='mean')


        # Unlabeled data part:
        features = self.feature_extractor(data)
        features = features.flatten(1, 2)
        logits = self.classifier(features)

        ema_features = self.ema_feature_extractor(data) # requires_grad=False !!
        ema_features = ema_features.flatten(1, 2)
        ema_logits = self.ema_classifier(ema_features)

        Lconsistency = softmax_mse_loss(logits, ema_logits)


        loss = Lx + ema_Lx + Lconsistency * self.cons_wt

        loss.backward()
        self.optimizer.step()
        self.update_ema_variables(self.feature_extractor, self.ema_feature_extractor, self.ema_decay)
        self.update_ema_variables(self.classifier, self.ema_classifier, self.ema_decay)

        return {'Total_loss': loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.classifier]

    def update_ema_variables(self, model, ema_model, alpha):
        # Use the true average until the exponential average is more correct
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

class DivideMix(Algorithm):
    # https://github.com/LiJunnan1992/DivideMix
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams, device):
        super(DivideMix, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.temporal_encoder = backbone_temporal(hparams)
        self.classifier = classifier(configs, hparams)

        self.feature_extractor2 = backbone_fe(configs)
        self.classifier2 = classifier(configs, hparams)

        self.network = nn.Sequential(self.feature_extractor, self.classifier, self.feature_extractor2, self.classifier2)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.configs = configs

        self.T = 0.5
        self.alpha = 0.1
        self.semiLoss = SemiLoss()
        self.lamb = 0.1


    def update(self, samples, fl_samples):
        self.feature_extractor.train()
        self.classifier.train()

        self.feature_extractor2.eval()
        self.classifier2.eval()

        # ====== Data =====================
        u_data = samples["sample_w_aug"].float()
        u_data2 = samples["sample_s_aug"].float()


        fl_data = fl_samples["sample_w_aug"].float()  # fl for few labeled
        batch_size = fl_data.size(0)
        fl_data2 = fl_samples["sample_s_aug"].float()  # fl for few labeled
        fl_labels = fl_samples["class_labels"].long()
        fl_labels = torch.zeros(batch_size, self.configs.num_classes).to(self.device).scatter_(1, fl_labels.view(-1, 1), 1)
        w_x =torch.from_numpy(np.array([0.4]*batch_size)).type(torch.FloatTensor).to(self.device)
        w_x = w_x.view(-1, 1)


        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = self.classifier(self.feature_extractor(u_data).flatten(1, 2))
            outputs_u12 = self.classifier(self.feature_extractor(u_data2).flatten(1, 2))
            outputs_u21 = self.classifier2(self.feature_extractor2(u_data).flatten(1, 2))
            outputs_u22 = self.classifier2(self.feature_extractor2(u_data2).flatten(1, 2))

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
                  torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4

            ptu = pu ** (1 / self.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = self.classifier(self.feature_extractor(fl_data).flatten(1, 2))
            outputs_x2 = self.classifier(self.feature_extractor(fl_data2).flatten(1, 2))

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * fl_labels + (1 - w_x) * px
            ptx = px ** (1 / self.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

            # mixmatch
        l = np.random.beta(self.alpha, self.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([fl_data, fl_data2, u_data, u_data2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = self.classifier(self.feature_extractor(mixed_input).flatten(1, 2))
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        Lx, Lu = self.semiLoss(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:])


        # regularization
        prior = torch.ones(self.configs.num_classes) / self.configs.num_classes
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + self.lamb * Lu + penalty


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.classifier]

class supervised(Algorithm):
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams):
        super(supervised, self).__init__(configs)

        self.feature_extractor = backbone_fe
        self.temporal_encoder = backbone_temporal
        self.classifier = classifier(configs, hparams)
        self.network = nn.Sequential(self.feature_extractor, self.temporal_encoder, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )

        self.hparams = hparams

    def update(self, samples):
        # ====== Data =====================
        data = samples['sample_ori'].float()
        labels = samples['class_labels'].long()

        # ====== Source =====================
        self.optimizer.zero_grad()

        # Src original features
        features = self.feature_extractor(data)
        features = self.temporal_encoder(features)
        logits = self.classifier(features)

        # Cross-Entropy loss
        x_ent_loss = self.cross_entropy(logits, labels)

        x_ent_loss.backward()
        self.optimizer.step()

        return {'Total_loss': x_ent_loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.classifier]
