import torch
import datasets as cdata
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import models as models
import math
import random
import numpy as np
import os


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self._create_model()

        self.optimizer = torch.optim.Adam(self.model.module.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        def linear_warmup_lr(epoch):
            max_lr = self.args.lr  
            min_lr = max_lr / 10
            if epoch < self.args.warmup_epochs:
                return min_lr + (max_lr - min_lr) * (epoch / self.args.warmup_epochs) 
            else:
                return 1.0  
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_warmup_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.gamma)
        if dist.get_rank() == 0:
            self.writer = SummaryWriter(os.path.join(self.args.dump_dir, 'log'))
        self.pbar = tqdm(range(self.current_epoch, self.args.num_epochs), unit="iters")

        os.makedirs(self.args.dump_dir, exist_ok=True)
        for subdir in ['ckpts', 'log', 'test_RMSE']:
            dir = os.path.join(self.args.dump_dir, subdir)
            os.makedirs(dir, exist_ok=True)

    def _create_model(self):
        if self.args.case == 'cylinder':
            self.model_class = models.Cylinder
            self.dataset_class = cdata.MeshCylinderDataset
        elif self.args.case == 'flag':
            self.model_class = models.Flag
            self.dataset_class = cdata.MeshFlagDataset
        else:
            raise NotImplementedError("A Case not wrapped yet")
        self.model = self.model_class(
            pos_dim=self.args.space_dim, 
            ld=self.args.hidden_dim, 
            layer_num=self.args.multi_mesh_layer, 
            pre_layer_num=self.args.pre_layer_num, 
            bottom_layer_num=self.args.bottom_layer_num,
            mlp_hidden_layer=self.args.hidden_depth, 
            MP_times=self.args.mp_time,
            enhance=self.args.enhance, 
            agg_conv_pos=self.args.agg_conv_pos
        )

        if self.args.restart_epoch < 0 and not self.args.scratch:
            try:
                self.model.load_state_dict(
                    torch.load(self.args.path),
                    strict=False
                )
            except:
                checkpoint = torch.load(self.args.path)
                model_dict = self.model.state_dict()
                # Filter out unnecessary keys in checkpoint
                checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
                keys = set(checkpoint.keys())
                # Check if shapes match
                for k in keys:
                    if model_dict[k].shape != checkpoint[k].shape:
                        print(f"\033[31mIgnoring parameter {k} due to shape mismatch:\033[0m {model_dict[k].shape} != {checkpoint[k].shape}")
                        del checkpoint[k]
                self.model.load_state_dict(checkpoint, strict=False)
            print(f"\033[31mLoad pretrained from\033[0m {self.args.path}")

        POST_FIX_1 = '_layernum_' + str(self.args.multi_mesh_layer)
        POST_FIX_2 = POST_FIX_1 + '_MPHIDDENLAYER_' + str(self.args.hidden_depth) + '_MPHIDDENTDIM_' + str(self.args.hidden_dim) + '_MPtime_' + str(self.args.mp_time) + '_NoiseLevel_' + str(
            self.args.noise_level)
        self.checkpt_name = self.args.case + POST_FIX_2 + '.pt'
        if self.args.restart_epoch >= 0:
            print('hi, restarting')
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(torch.load(os.path.join(self.args.dump_dxir, 'ckpts', str(self.args.restart_epoch) + "_" + self.checkpt_name)))
            else:
                self.model.load_state_dict(torch.load(os.path.join(self.args.dump_dir, 'ckpts', str(self.args.restart_epoch) + "_" + self.checkpt_name)))
            
            self.current_epoch = self.args.restart_epoch + 1
            if self.current_epoch < self.args.warmup_epochs:
                min_lr = self.args.lr / 10
                max_lr = self.args.lr
                self.args.lr = min_lr + (max_lr - min_lr) * (self.current_epoch / self.args.warmup_epochs)
            else:
                decay_epochs = self.current_epoch - self.args.warmup_epochs
                self.args.lr *= self.args.gamma**(decay_epochs)
                self.args.lr = max(self.args.lr, 1e-6)
            print('restarted lr is: ', self.args.lr)
        else:
            self.current_epoch = 0

        self.model = nn.parallel.DistributedDataParallel(
            self.model.cuda(self.args.local_rank),
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
        )

    def _preproc_multi_infos(self, mdata, b_data):
        # process the multi-level mesh for batched data here
        # 1. if there is contact, merge multiple graphs into a big one by adding offsets (each layer)
        _, n, _ = mdata.in_feature.shape
        # no contact, then share the graph between batches
        # only need to reshape input tensor
        m_ids = mdata.m_idx
        m_gs_list = mdata.m_g
        m_gs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in m_gs_list]
        # reshape
        b = b_data.y.shape[0] // n
        b_data.x = b_data.x.reshape(b, n, -1).to(self.device)
        b_data.y = b_data.y.reshape(b, n, -1).to(self.device)
        return m_ids, m_gs, b_data

    def _create_datset_offline(self, id, mode='train', stride=1):
        if mode == 'train':
            prob = np.random.rand()
            add_noise = (prob < 0.667)
            mdata = self.dataset_class(self.args.data_dir,
                                       instance_id=id,
                                       layer_num=self.args.multi_mesh_layer,
                                       stride=stride,
                                       noise_shuffle=add_noise,
                                       noise_level=self.args.noise_level,
                                       noise_gamma=self.args.noise_gamma,
                                       recal_mesh=self.args.recal_mesh,
                                       consist_mesh=self.args.consist_mesh,
                                       args=self.args)
        else:
            mdata = self.dataset_class(self.args.data_dir,
                                       instance_id=id,
                                       layer_num=self.args.multi_mesh_layer,
                                       stride=stride,
                                       noise_shuffle=False,
                                       recal_mesh=self.args.recal_mesh,
                                       consist_mesh=self.args.consist_mesh,
                                       mode=mode,
                                       args=self.args)
        return mdata

    def run_epoch(self, epoch, mode='train'):
        if mode != 'train':
            self.model.eval()
        mean_loss_insts = 0
        count_insts = 0
        instance_len = self.args.n_train if mode == 'train' else (self.args.n_valid if mode == 'valid' else self.args.n_test)
        instance_list = list(range(instance_len))
        if mode == 'train':
            random.shuffle(instance_list)
        # to avoid messy writer, store RMSE 1st in an array
        rmse_array = np.zeros(len(instance_list))
        for i in range(len(instance_list)):
            self.pbar.set_postfix(instance_id=i)
            id = instance_list[i]
            mdata = self._create_datset_offline(id, mode=mode)
            mean_loss = 0
            count = 0
            pen_coeff = mdata.suggested_pen_coef().to(self.device)
            sampler = torch.utils.data.distributed.DistributedSampler(mdata, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
            if mode == 'test':
                loader = DataLoader(mdata, batch_size=1, shuffle=False)
            else:
                bs = self.args.batch
                loader = DataLoader(
                    mdata, batch_size=bs, shuffle=False,
                    sampler=sampler,
                )
            sampler.set_epoch(epoch)
            for id_batch, b_data in enumerate(loader):
                m_ids, m_gs, b_data = self._preproc_multi_infos(mdata, b_data)
                # optimization
                self.optimizer.zero_grad()
                loss, _, non_zero_elements = self.model(m_ids, m_gs, b_data.x, b_data.y, pen_coeff)
                
                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()
                # stats
                mean_loss += loss.item() * non_zero_elements
                count += non_zero_elements

            # stats
            with torch.autograd.no_grad():
                mean_loss_insts += mean_loss
                count_insts += count
                mean_loss /= count
                rmse_array[i] = math.sqrt(mean_loss)
            # safety clean
            del mdata
        
        if dist.get_rank() == 0:
            for i in range(len(instance_list)):
                self.writer.add_scalar(f'Inst RMSE/{mode}/Epoch: {epoch}', rmse_array[i], i)

        # stats
        mean_loss_insts /= count_insts
        if mode == 'train':
            # opt scheduler
            if epoch < self.args.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                if self.optimizer.param_groups[0]['lr'] > 1e-6:
                    self.scheduler.step()
                    if self.optimizer.param_groups[0]['lr'] < 1e-6:
                        self.optimizer.param_groups[0]['lr'] = 1e-6
        else:
            self.model.train()
        return mean_loss_insts

    
    def train(self):
        for epoch in self.pbar:
            mean_loss_train = self.run_epoch(epoch)

            with torch.autograd.no_grad():
                # train loss record
                if dist.get_rank() == 0:
                    self.pbar.set_description(f"Epoch: {epoch}, Training RMSE: {math.sqrt(mean_loss_train)}")
                    self.writer.add_scalar('RMSE/train', math.sqrt(mean_loss_train), epoch)
                    self.writer.add_scalar('lr/train', self.optimizer.param_groups[0]['lr'], epoch)
                    # dump ckpt
                    ckpt_path = os.path.join(self.args.dump_dir, 'ckpts', str(epoch) + "_" + self.checkpt_name)
                    torch.save(self.model.module.state_dict(), ckpt_path)

                # valid loss record
                if self.args.n_valid > 0 and (epoch+1) % 5 == 0:
                    mean_loss_valid = self.run_epoch(epoch, mode='test')
                    if dist.get_rank() == 0:
                        self.writer.add_scalar('RMSE/test', math.sqrt(mean_loss_valid), epoch)
                dist.barrier()

    
    def test(self, epoch=None):
        self.model.eval()
        epoch = self.args.restart_epoch if epoch==None else epoch
        rmse = []
        with torch.autograd.no_grad():
            # reload the model
            print('test on epoch, ', epoch)
            if self.args.n_test > 0:
                mean_loss_test = self.run_epoch(epoch, mode='test')
                print('epoch', epoch, 'RMSE/test', math.sqrt(mean_loss_test))
                if dist.get_rank() == 0:
                    self.writer.add_scalar('RMSE/test', math.sqrt(mean_loss_test), epoch)

            rmse.append(math.sqrt(mean_loss_test))

        if dist.get_rank() == 0:
            RMSE_cell = np.array([np.mean(rmse)])
            dump_path = os.path.join(self.args.dump_dir, 'test_RMSE', 'epoch_' + str(epoch) + '.csv')
            np.savetxt(dump_path, RMSE_cell, delimiter=',')
        
    
    def rollout(self, epoch=None, time_stps=None):
        epoch = self.args.restart_epoch if epoch == None else epoch
        instance_list = list(range(self.args.n_test))
        errors = []
        self.model.eval()
        with torch.autograd.no_grad():
            for i in tqdm(range(len(instance_list))):
                id = instance_list[i]
                # rollout for this instance, then record into a file
                mdata = self._create_datset_offline(id, mode='test')
                if time_stps is None:
                    L = mdata.in_feature.shape[0]
                    instance_rollout_error = np.zeros(L)
                    time_stps = L
                else:
                    L = time_stps
                    instance_rollout_error = np.zeros(L)
                for id_batch, b_data in enumerate(DataLoader(mdata, batch_size=1, shuffle=False)):
                    # print('id_batch', id_batch)
                    _, n, _ = mdata.in_feature.shape
                    m_ids = mdata.m_idx
                    m_gs_list = mdata.m_g
                    m_gs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in m_gs_list]
                    if mdata.has_contact:
                        m_cgs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in mdata.m_cgs[id_batch]]
                        m_g_cg = [[g, cg] for g, cg in zip(m_gs, m_cgs)]
                        m_gs = m_g_cg
                    pen_coeff = mdata.suggested_pen_coef().to(self.device)
                    if id_batch == 0:
                        current_stat = mdata[0].x.reshape(1, n, -1).to(self.device)
                    
                    b_data.y = b_data.y.reshape(1, n, -1).to(self.device)
                    loss, out, _ = self.model(m_ids, m_gs, current_stat, b_data.y, pen_coeff)
                    # record global error
                    instance_rollout_error[id_batch] = math.sqrt(loss.item())
                    # push forward state
                    current_stat = mdata._push_forward(out, current_stat)
                    current_stat = current_stat.detach()

                dir = os.path.join(self.args.dump_dir, f'rollout_RMSE_epoch_' + str(epoch))
                os.makedirs(dir, exist_ok=True)
                dump_path = os.path.join(dir, str(id) + '.csv')
                np.savetxt(dump_path, instance_rollout_error, delimiter=',')
                errors.append(np.mean(instance_rollout_error))