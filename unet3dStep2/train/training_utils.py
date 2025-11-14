"""
Modified from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import shutil
import time
import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
#from torchsummary import summary

from monai.transforms import RandGaussianNoise


try:
    from torch.utils.data._utils.collate import default_collate
except ModuleNotFoundError:
    # import from older versions of pytorch
    from torch.utils.data.dataloader import default_collate

global_step = 0

def epoch_training(training_labeled_loader, training_unlabeled_loader,
                   model_one, model_two, criterion, 
                   optimizer_one, optimizer_two,
                   epoch, n_gpus=None, print_frequency=1,
                   print_gpu_memory=False, scaler=None, samples_per_epoch=None, 
                   iteration=1):
    
    # 记录当前epoch运行了多少次反向传播
    i = 0
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    # losses_one = AverageMeter('total Loss (model1)', ':.4e')
    # losses_two = AverageMeter('total Loss (model2)', ':.4e')
    seg_losses_one = AverageMeter('seg Loss (model1)', ':.4e')
    seg_losses_two = AverageMeter('seg Loss (model2)', ':.4e')
    cps_losses_one = AverageMeter('cps loss (model1)', ':.4e')
    cps_losses_two = AverageMeter('cps loss (mode2)', ':.4e')
    
    if iteration > 1:
        prefix = "Epoch: [{}({})]".format(epoch, iteration)
    else:
        prefix = "Epoch: [{}]".format(epoch)
    # 只跑两个labeled和unlabeled loss
    batch_size_cyb = 4
    progress = ProgressMeter(
        batch_size_cyb,  #len(training_labeled_loader)+len(training_unlabeled_loader),
        [batch_time, data_time, 
         seg_losses_one, seg_losses_two,
         cps_losses_one, cps_losses_two],
        prefix=prefix)

    use_amp = scaler is not None

    # switch to train mode
    model_one.train()
    model_two.train()

    end = time.time()
    
    # 分别对labeled和unlabele image进行数据扩增，得到两个网络的对应输入
    # 这里的概率prob=1.0，不然只有部分数据会被随机做扩增
    trans_gau_noise = RandGaussianNoise(prob=1.0, mean=0.2, std=0.2)
    cps_loss_weight = 0.1
    
    for i_labeled, item_labeled in enumerate(training_labeled_loader):
        
        images_labeled = item_labeled["image"]
        target = item_labeled["label"]
        images_labeled = trans_gau_noise(images_labeled, randomize=True)
        
        batch_size_labeled = images_labeled.shape[0]
        # measure data loading time
        #data_time.update(time.time() - end)
        # if n_gpus:
        #     torch.cuda.empty_cache()
        #     if print_gpu_memory:
        #         for i_gpu in range(n_gpus):
        #             print("Memory allocated (device {}):".format(i_gpu),
        #                   human_readable_size(torch.cuda.memory_allocated(i_gpu)))
        #             print("Max memory allocated (device {}):".format(i_gpu),
        #                   human_readable_size(torch.cuda.max_memory_allocated(i_gpu)))
        #             print("Memory cached (device {}):".format(i_gpu),
        #                   human_readable_size(torch.cuda.memory_cached(i_gpu)))
        #             print("Max memory cached (device {}):".format(i_gpu),
        #                   human_readable_size(torch.cuda.max_memory_cached(i_gpu)))
        seg_loss_n1, seg_loss_n2 = batch_loss_train_labeled(
                                      model_one, 
                                      model_two,
                                      images_labeled,
                                      target, 
                                      criterion,
                                      epoch,
                                      n_gpus=n_gpus, 
                                      use_amp=use_amp) 
        seg_losses_one.update(seg_loss_n1.item(), batch_size_labeled)
        seg_losses_two.update(seg_loss_n2.item(), batch_size_labeled)
        
        # compute gradient and do optimizing step        
        optimizer_one.zero_grad()   
        optimizer_two.zero_grad()
        if scaler:
            scaler.scale(seg_loss_n1).backward()
            scaler.step(optimizer_one)
            scaler.update()
            scaler.scale(seg_loss_n2).backward()
            scaler.step(optimizer_two)
            scaler.update()
        else:
            # compute gradient and do step
            seg_loss_n1.backward()
            optimizer_one.step()
            seg_loss_n2.backward()
            optimizer_two.step()
            
        del seg_loss_n1, seg_loss_n2
        
        i += 1
        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_frequency == 0:
            progress.display(i)
        # 只跑两个labeled和unlabeled loss，和上面的batch_size_cyb呼应
        if i == 2:
            break
        # if samples_per_epoch and (i) * batch_size >= samples_per_epoch:
        #     break
        
    for i_unlabeled, item_unlabeled in enumerate(training_unlabeled_loader):
        images_unlabeled = item_unlabeled["image"]
        images_unlabeled = trans_gau_noise(images_unlabeled, randomize=True)       
        
        cps_loss_one, cps_loss_two = batch_loss_train_unlabeled(
                                      model_one, 
                                      model_two,
                                      images_unlabeled,
                                      criterion,
                                      epoch,
                                      n_gpus=n_gpus, 
                                      use_amp=use_amp)   
        batch_size_unlabeled = images_unlabeled.shape[0]
        # measure accuracy and record loss
        cps_loss_one = cps_loss_weight * cps_loss_one
        cps_loss_two = cps_loss_weight * cps_loss_two
        
        cps_losses_one.update(cps_loss_one.item(), batch_size_unlabeled)
        cps_losses_two.update(cps_loss_two.item(), batch_size_unlabeled)
        
        # compute gradient and do optimizing step        
        optimizer_one.zero_grad()   
        optimizer_two.zero_grad()
        if scaler:
            scaler.scale(cps_loss_one).backward()
            scaler.step(optimizer_one)
            scaler.update()
            scaler.scale(cps_loss_two).backward()
            scaler.step(optimizer_two)
            scaler.update()
        else:
            # compute gradient and do step
            cps_loss_one.backward()
            optimizer_one.step()
            cps_loss_two.backward()
            optimizer_two.step()
    
            
         
        i += 1
        del cps_loss_one, cps_loss_two
    
            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
        if i % print_frequency == 0:
            progress.display(i)
            
        # 只跑两个labeled和unlabeled loss，和上面的batch_size_cyb呼应
        if i == 4:
            break
    
            # if samples_per_epoch and (i) * batch_size >= samples_per_epoch:
            #     break
    return seg_losses_one.avg, seg_losses_two.avg, cps_losses_one.avg, cps_losses_two.avg


def batch_loss_train_labeled(model_one, model_two, 
                         images_labeled,
                         target, criterion,
                         epoch, n_gpus=0, use_amp=None,
                         inferer=None):
    
    if n_gpus is not None:
        images_labeled_var = torch.autograd.Variable(images_labeled.cuda())  
        target_var = torch.autograd.Variable(target.cuda())

    # compute output
    if use_amp:
        from torch.cuda.amp import autocast
        with autocast():
            seg_loss_one = _batch_loss_labeled(
                                       model_one,
                                       images_labeled_var,
                                       target_var, 
                                       criterion, 
                                       epoch, inferer=inferer)
            seg_loss_two = _batch_loss_labeled(
                                       model_two,
                                       images_labeled_var,
                                       target_var, 
                                       criterion, 
                                       epoch, inferer=inferer)
          
            return seg_loss_one, seg_loss_two
    else:
        seg_loss_one = _batch_loss_labeled(
                                   model_one,
                                   images_labeled_var,
                                   target_var, 
                                   criterion, 
                                   epoch, inferer=inferer)
        seg_loss_two = _batch_loss_labeled(
                                   model_two,
                                   images_labeled_var,
                                   target_var, 
                                   criterion, 
                                   epoch, inferer=inferer)
       
        return seg_loss_one, seg_loss_two

def batch_loss_train_unlabeled(model_one, model_two, 
                         images_unlabeled,
                         criterion,
                         epoch, n_gpus=0, use_amp=None,
                         inferer=None):
    
    if n_gpus is not None:
        images_unlabeled_var = torch.autograd.Variable(images_unlabeled.cuda())    

    # compute output
    if use_amp:
        from torch.cuda.amp import autocast
        with autocast():            
            cps_loss_one = _batch_loss_unlabeled(
                                       model_one,
                                       model_two,
                                       images_unlabeled_var,
                                       criterion, 
                                       epoch, inferer=inferer)
            cps_loss_two = _batch_loss_unlabeled(
                                       model_two,
                                       model_one,
                                       images_unlabeled_var,
                                       criterion, 
                                       epoch, inferer=inferer)

            return cps_loss_one, cps_loss_two
    else:        
        cps_loss_one = _batch_loss_unlabeled(
                                   model_one,
                                   model_two,
                                   images_unlabeled_var,
                                   criterion, 
                                   epoch, inferer=inferer)
        cps_loss_two = _batch_loss_unlabeled(
                                   model_two,
                                   model_one,
                                   images_unlabeled_var,
                                   criterion, 
                                   epoch, inferer=inferer)
        return cps_loss_one, cps_loss_two

def _batch_loss_labeled(model, images_labeled_var, 
                        target_var, criterion,
                        epoch, inferer=None):
    """
    inferer: should take in the inputs and the model and output the prediction. This is based on the MONAI Inferer
    classes.
    """
    if inferer is not None:
        output = inferer(images_labeled_var, model).to(images_labeled_var.device)
    else:
        output = model(images_labeled_var)
         
    seg_loss = criterion(output, target_var)
    # 教师模型的seg loss没有用到，仅仅是打印了一下
    
    return seg_loss

def _batch_loss_unlabeled(model, psudo_labeled_model, images_unlabeled_var, 
                        criterion,
                        epoch, inferer=None):
    """
    inferer: should take in the inputs and the model and output the prediction. This is based on the MONAI Inferer
    classes.
    """
    if inferer is not None:
        output = inferer(images_unlabeled_var, model).to(images_unlabeled_var.device)
        output_psudo = inferer(images_unlabeled_var, 
                               psudo_labeled_model).to(images_unlabeled_var.device)
    else:
        output = model(images_unlabeled_var)
        output_psudo = psudo_labeled_model(images_unlabeled_var)
    
    # confidence = 0.95
    # output_pred = torch.autograd.Variable(output.detach().data, requires_grad=False)
    # output_pred_prob = torch.sigmoid(output_pred)
    # psudo_pred = torch.autograd.Variable(output_psudo.detach().data, requires_grad=False)
    # psudo_pred_prob = torch.sigmoid(psudo_pred)
    # confidence_mask = (output_pred_prob > confidence) & (psudo_pred_prob > confidence)
    # 另一个模型的输出转为二值     
    output_psudo_pred = torch.autograd.Variable(output_psudo.detach().data, requires_grad=False)
    pred_psudo = torch.sigmoid(output_psudo_pred)
    pred_psudo_bina = pred_psudo >= 0.5
    pred_psudo_bina = pred_psudo_bina.long()
    
    cps_loss = criterion(output, pred_psudo_bina)
    
    return cps_loss


def _batch_loss(model, images, target, criterion, inferer=None):
    """
    inferer: should take in the inputs and the model and output the prediction. This is based on the MONAI Inferer
    classes.
    """
    if inferer is not None:
        output = inferer(images, model).to(images.device)
    else:
        output = model(images)
    batch_size = images.size(0)
    loss = criterion(output, target)
    return loss, batch_size

def batch_loss(model, images, target, criterion, n_gpus=0, use_amp=None, inferer=None):
    if n_gpus is not None:
        images = images.cuda()
        target = target.cuda()
    # compute output
    if use_amp:
        from torch.cuda.amp import autocast
        with autocast():
            return _batch_loss(model, images, target, criterion, inferer=inferer)
    else:
        return _batch_loss(model, images, target, criterion, inferer=inferer)

def epoch_validation(val_loader, model, criterion, n_gpus, print_freq=1, use_amp=False, inferer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Validation: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, item in enumerate(val_loader):
            images = item["image"]
            target = item["label"]
            loss, batch_size = batch_loss(model, images, target, criterion, n_gpus=n_gpus,  use_amp=use_amp,
                                          inferer=inferer)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i+1)

            if n_gpus:
                torch.cuda.empty_cache()

    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def update_ema_variables(model, ema_model, alpha_para, global_step):
    # Use the true average until the exponential average is more correct
    alpha_para = min(1 - 1 / (global_step + 1), alpha_para)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        alpha = 1-alpha_para
        ema_param.data.mul_(alpha_para).add_(param.data, alpha=alpha)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    # consistency是最终值, consistency_rampup是指定过多少个epoch达到最终值
    # 最开始是个小值 
    # 原始的ST代码中给定的consistency是100，consistency_rampup是5，CIFAI数据集上
    consistency = 0.1
    consistency_rampup = 10
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def human_readable_size(size, decimal_places=1):
    for unit in ['', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


def collate_flatten(batch, x_dim_flatten=5, y_dim_flatten=2):
    x, y = default_collate(batch)
    if len(x.shape) > x_dim_flatten:
        x = x.flatten(start_dim=0, end_dim=len(x.shape) - x_dim_flatten)
    if len(y.shape) > y_dim_flatten:
        y = y.flatten(start_dim=0, end_dim=len(y.shape) - y_dim_flatten)
    return [x, y]


def collate_5d_flatten(batch, dim_flatten=5):
    return collate_flatten(batch, x_dim_flatten=dim_flatten, y_dim_flatten=dim_flatten)
