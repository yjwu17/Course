import os
import tqdm
import torch
import torch.utils.model_zoo
import torch.nn.functional as F
import torchvision.models as models
from src.model.snn import *
from src.model.cann import CANN
from src.tools.utils import *
from src.model.btsp import *

# gpu environment 
#os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class MHNNBTSP(nn.Module):
    def __init__(self, **kwargs):
        super(MHNNBTSP, self).__init__()
        
        # experimental parameters
        self.spike_train_sample_len = kwargs.get('spike_train_sample_len')
        self.temporal_dim_handling = kwargs.get('temporal_dim_handling')
        self.binary_btsp = kwargs.get('binary_btsp')
        self.btsp_hash_ratio = kwargs.get('btsp_hash_ratio')
        self.btsp_fw = kwargs.get('btsp_fw')
        self.btsp_wta_ratio = kwargs.get('btsp_wta_ratio')
        self.btsp_fq_constant = kwargs.get('btsp_fq_constant')
        self.fast_btsp = kwargs.get('fast_btsp')
        
        post_preprocessing_norm = kwargs.get('post_preprocessing_norm')
        self.aps_scale = post_preprocessing_norm.modality_scale_factor[0]
        self.dvs_scale = post_preprocessing_norm.modality_scale_factor[1]
        self.gps_scale = post_preprocessing_norm.modality_scale_factor[2]
        self.head_scale = post_preprocessing_norm.modality_scale_factor[3]
        self.time_scale = post_preprocessing_norm.modality_scale_factor[4]

        # static parameters
        self.cnn_arch = kwargs.get('cnn_arch')
        self.num_class = kwargs.get('num_class')
        
        self.enable_cnn_wta = kwargs.get('enable_cnn_wta')
        self.cnn_wta_ratio = kwargs.get('cnn_wta_ratio')
        
        self.cann_num = kwargs.get('cann_num')
        self.rnn_num = kwargs.get('rnn_num')
        self.lr = kwargs.get('lr')
        self.sparse_lambdas = kwargs.get('sparse_lambdas')
        self.r = kwargs.get('r')

        self.reservoir_num = kwargs.get('reservoir_num')
        self.threshold = kwargs.get('spiking_threshold')

        self.num_epoch = kwargs.get('num_epoch')
        self.num_iter = kwargs.get('num_iter')
        self.w_fps = kwargs.get('w_fps')
        self.w_gps = kwargs.get('w_gps')
        self.w_dvs = kwargs.get('w_dvs')
        self.w_head = kwargs.get('w_head')
        self.w_time = kwargs.get('w_time')

        self.seq_len_aps = kwargs.get('seq_len_aps')
        self.seq_len_gps = kwargs.get('seq_len_gps')
        self.seq_len_dvs = kwargs.get('seq_len_dvs')
        self.seq_len_head = kwargs.get('seq_len_head')
        self.seq_len_time = kwargs.get('seq_len_time')
        self.dvs_expand = kwargs.get('dvs_expand')

        self.ann_pre_load = kwargs.get('ann_pre_load')
        self.snn_pre_load = kwargs.get('snn_pre_load')
        self.re_trained = kwargs.get('re_trained')

        self.train_exp_idx = kwargs.get('train_exp_idx')
        self.test_exp_idx = kwargs.get('test_exp_idx')

        self.data_path = kwargs.get('data_path')
        self.snn_path = kwargs.get('snn_path')

        self.device = kwargs.get('device')
        #self.device = device

        if self.ann_pre_load:
            print("=> Loading pre-trained model '{}'".format(self.cnn_arch))
            self.cnn = models.__dict__[self.cnn_arch](pretrained=self.ann_pre_load)
        else:
            print("=> Using randomly inizialized model '{}'".format(self.cnn_arch))
            self.cnn = models.__dict__[self.cnn_arch](pretrained=self.ann_pre_load)

        if self.cnn_arch == "mobilenet_v2":
            """ MobileNet """
            self.feature_dim = self.cnn.classifier[1].in_features
            self.cnn.classifier[1] = nn.Identity()

        elif self.cnn_arch == "resnet50":
            """ Resnet50 """

            self.feature_dim = 512

            # self.cnn.layer1 = nn.Identity()
            self.cnn.layer2 = nn.Identity()
            self.cnn.layer3 = nn.Identity()
            self.cnn.layer4 = nn.Identity()
            self.cnn.fc = nn.Identity()

            self.cnn.layer1[1] = nn.Identity()
            self.cnn.layer1[2] = nn.Identity()

            # self.cnn.layer2[0] = nn.Identity()
            # self.cnn.layer2[0].conv2 = nn.Identity()
            # self.cnn.layer2[0].bn2 = nn.Identity()

            fc_inputs = 256
            self.cnn.fc = nn.Linear(fc_inputs,self.feature_dim)

        else:
            print("=> Please check model name or configure architecture for feature extraction only, exiting...")
            exit()

        for param in self.cnn.parameters():
            param.requires_grad = self.re_trained

        #############
        # SNN module
        #############
        self.snn = SNN(device = self.device).to(self.device)
        self.snn_out_dim = self.snn.fc2.weight.size()[1]
        self.ann_out_dim = self.feature_dim
        self.cann_out_dim = 4 * self.cann_num
        self.reservior_inp_num = self.ann_out_dim + self.snn_out_dim + self.cann_out_dim
        self.LN = nn.LayerNorm(self.reservior_inp_num)
        if self.snn_pre_load:
            self.snn.load_state_dict(torch.load(self.snn_path)['snn'])

        #############
        # CANN module
        #############
        self.cann_num = self.cann_num
        self.cann = None
        self.num_class = self.num_class
        
        #############
        # BTSP module
        #############
        
        self.btsp_input_dim = self.feature_dim + self.snn_out_dim + self.cann_out_dim
        if self.temporal_dim_handling == 'flatten':
            self.btsp_input_dim *= self.spike_train_sample_len
        self.btsp_input_dim = int(self.btsp_input_dim)
        self.btsp_memory_neurons = int(self.btsp_input_dim * self.btsp_hash_ratio) # empirical value
        self.btsp_wta_num = int(self.btsp_memory_neurons * self.btsp_wta_ratio)
        # self.btsp_fw = 0.6 # connection ratio between neurons, empirical value
        # self.btsp_wta_num = 100 # arbitrary value, subject to change
        self.btsp_fq = self.btsp_fq_constant * self.btsp_wta_num / self.btsp_memory_neurons # plateau potential possibility
        print(f"BTSP fq: {self.btsp_fq}")
        if self.binary_btsp:
            self.btsp = BinaryBTSPLayer(BinaryBTSPLayerParams(input_dim=self.btsp_input_dim,
                                              memory_neurons=self.btsp_memory_neurons,
                                              fq=self.btsp_fq,
                                              fw=self.btsp_fw,
                                              device=self.device,
                                              fast_btsp=self.fast_btsp))
        else:
            self.btsp = ContinuousBTSPLayer(ContinuousBTSPLayerParams(input_dim=self.btsp_input_dim,
                                              memory_neurons=self.btsp_memory_neurons,
                                              fq=self.btsp_fq,
                                              fw=self.btsp_fw,
                                              device=self.device))

    def cann_init(self, data):
        self.cann = CANN(data, num=self.cann_num)

    def lr_initial_schedule(self, lrs=1e-3):
        hyper_param_list = ['decay',
                            'thr_beta1', 'thr_decay1',
                            'ref_beta1', 'ref_decay1',
                            'cur_beta1', 'cur_decay1']
        hyper_params = list(filter(lambda x: x[0] in hyper_param_list, self.named_parameters()))
        base_params = list(filter(lambda x: x[0] not in hyper_param_list, self.named_parameters()))
        hyper_params = [x[1] for x in hyper_params]
        base_params = [x[1] for x in base_params]
        optimizer = torch.optim.SGD(
            [
                {'params': base_params, 'lr': lrs},
                {'params': hyper_params, 'lr': lrs / 2},
            ], lr=lrs, momentum=0.9, weight_decay=1e-7
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.num_epoch))
        return optimizer, scheduler
    
    def temporal_dim_processing(self, data: list, sample_seq_len: int):
        """Given a list of data at different time scales, sample them to the same length
        
        For each data tensor in the data list, the first dimension should be the spike train dimension.
        Every input tensor will be sampled to the same spike train length defined by sample_seq_len.
        Then, all tensors will be concatenated along the first dimension, flattened into a single tensor with larger neuron dimension, or averaged into a single tensor with same shape.
        """
        resampled_modalities = []

        for modality in data:
            spike_train_len, sample_idx, channels = modality.shape
            
            # if has the same length, no need to resample
            if spike_train_len == sample_seq_len:
                resampled_modalities.append(modality)
                continue

            # Calculate the resampling scale factor
            scale_factor = float(sample_seq_len) / float(spike_train_len)

            # Perform uniform sampling
            resampled_spike_train = F.interpolate(modality.permute(2, 0, 1).unsqueeze(0),
                                                  size=(sample_seq_len, sample_idx),
                                                  mode='nearest')
            resampled_modalities.append(resampled_spike_train.squeeze(0).permute(1, 2, 0))

        # Concatenate all modalities' channels along the last dimension
        final_output = torch.cat(resampled_modalities, dim=-1)
        # now the final_output has shape (seq_length, batch_size, feature_dim)
        # if flatten_temporal_input is True, then merge seq_length into feature_dim, as now the temporal dimension is regard as part of the feature
        # desired shape: (1, batch_size, feature_dim * seq_length)
        if self.temporal_dim_handling == 'flatten':
            final_output = final_output.permute(1, 0, 2)
            # now the final_output has shape (batch_size, seq_length, feature_dim)
            final_output = final_output.reshape(final_output.size(0), 1, -1)
            # now the final_output has shape (batch_size, 1, feature_dim * seq_length)
            final_output = final_output.permute(1, 0, 2)
            # now the final_output has shape (1, batch_size, feature_dim * seq_length)
        elif self.temporal_dim_handling == 'average':
            # average on temporal(seq_length) dimension
            final_output = torch.mean(final_output, dim=0)
            # now the final_output has shape (batch_size, feature_dim)
            final_output = final_output.unsqueeze(0)
            # now the final_output has shape (1, batch_size, feature_dim)
        elif self.temporal_dim_handling == 'as_pattern':
            # do nothing
            pass
        else:
            raise ValueError(f"Unknown temporal_dim_handling: {self.temporal_dim_handling}")

        return final_output

    def forward(self, inp, epoch=100):
        output = None
        
        aps_inp = inp[0].to(self.device)
        gps_inp = inp[1]
        dvs_inp = inp[2].to(self.device)
        head_inp = inp[3].to(self.device)

        batch_size, seq_len, channel, w, h = aps_inp.size()

        aps_inp = aps_inp.view(batch_size * seq_len, channel, w, h)
        aps_out = self.cnn(aps_inp)
        out1 = aps_out.reshape(batch_size, self.seq_len_aps, -1).permute([1, 0, 2])
        if not self.w_fps > 0:
            out1 = torch.zeros(out1.shape, device=self.device).to(torch.float32)

        assert out1.size()[2] == self.ann_out_dim
        
        if self.enable_cnn_wta:
            topk, _ = torch.topk(out1, int(self.cnn_wta_ratio * self.ann_out_dim), dim=2)
            out1[out1 < topk[:, :, -1].unsqueeze(2)] = 0

        if self.w_dvs > 0:
            out2 = self.snn(dvs_inp, out_mode='time')
        else:
            out2 = torch.zeros(self.seq_len_dvs * 3, batch_size, self.snn_out_dim, device=self.device).to(torch.float32)

        ### CANN module

        
        if self.w_gps + self.w_head + self.w_time > 0:
            gps_record = []
            for idx in range(batch_size):
                buf = self.cann.update(torch.cat((gps_inp[idx],head_inp[idx].cpu()),axis=1), trajactory_mode=True)
                gps_record.append(buf[None, :, :, :])
            gps_out = torch.from_numpy(np.concatenate(gps_record)).cuda()
            gps_out = gps_out.permute([1, 0, 2, 3]).reshape(self.seq_len_gps, batch_size, -1)
        else:
            gps_out = torch.zeros((self.seq_len_gps, batch_size, self.cann_out_dim), device=self.device)

        # A generic CANN module was used for rapid testing; CANN1D/2D are provided in cann.py
        out3 = gps_out[:, :, self.cann_num:self.cann_num * 3].to(self.device).to(torch.float32)  # position
        out4 = gps_out[:, :, : self.cann_num].to(self.device).to(torch.float32)  # time
        out5 = gps_out[:, :, - self.cann_num:].to(self.device).to(torch.float32)  # direction

        out3 *= self.w_gps
        out4 *= self.w_time
        out5 *= self.w_head
        
        #print(f"out1 shape: {out1.shape}")
        #print(f"out2 shape: {out2.shape}")
        #print(f"out3 shape: {out3.shape}")
        #print(f"out4 shape: {out4.shape}")
        #print(f"out5 shape: {out5.shape}")
        
        ### post-preprocessing normalization
        
        # scale out1 - out5 by the corresponding scale factor
        out1 = out1 * self.aps_scale
        out2 = out2 * self.dvs_scale
        out3 = out3 * self.gps_scale
        out4 = out4 * self.time_scale
        out5 = out5 * self.head_scale
        
        
        ### BTSP module
        
        ## step 1: process multimodal information

        # the BTSP neurons take in input one by one at the scale of action potential
        # for each batch sample, ideally, BTSP should take in all spikes from all modalities
        # however, different modalities have different time scales, causing misalignment
        multimodal_inputs = [out1, out2, out3, out4, out5]
        combined_input = self.temporal_dim_processing(multimodal_inputs, self.spike_train_sample_len)
        #print(f"Combined input shape: {combined_input.shape}")
        
        # now that we have a uniform spike train length, combined input has shape (seq_length, batch_size, feature_dim)
        # BTSP neuron does not take a sample(batch) dimension, so we need to reshape the input
        # merge the batch info and seq info and get (seq_length * batch_size, feature_dim)
        # make sure the order is batch-first
        # i.e after reshape, comb_input[0:seq_length] should be the input of the first batch sample
        combined_input = combined_input.permute(1, 0, 2).reshape(-1, self.btsp_input_dim)
        #print(f"BTSP input shape: {combined_input.shape}")
        
        assert combined_input.size()[-1] == self.btsp_input_dim
        
        ## step 2: BTSP neuron simulation
        
        if self.training:
            if self.binary_btsp:
                btsp_output = self.btsp.learn_and_forward(combined_input)
            else:
                # learn each sample one by one
                btsp_output = torch.zeros(combined_input.size()[0], self.btsp_memory_neurons, device=self.device)
                for i in range(combined_input.size()[0]):
                    btsp_output[i] = self.btsp.learn_single_pattern(combined_input[i].unsqueeze(0))
        else:
            btsp_output = self.btsp.forward(combined_input)
            
        # WTA mechanism: only the top K neurons are allowed to spike
        # the rest are set to zero
        topk, _ = torch.topk(btsp_output, self.btsp_wta_num, dim=1)
        btsp_output[btsp_output < topk[:, -1].unsqueeze(1)] = 0
        
        # btsp_output has shape (seq_length * batch_size, btsp_memory_neurons)
        # reshape to (batch_size, seq_length, btsp_memory_neurons)
        #print(f"BTSP output shape: {btsp_output.shape}")
        btsp_output = btsp_output.reshape(batch_size, -1, self.btsp_memory_neurons)
        output = btsp_output
        return output, None