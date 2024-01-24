from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta

import wandb

from opts import opts
from models.model import create_model, load_model, save_model, freeze_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from models.classifier import Processor

os.environ["WANDB_API_KEY"] = 'c2047857ed4262cd4a3984cc221bf41d234f8f71'
os.environ["WANDB_MODE"] = "offline"


def main(opt):
	########################## 	DDP change 1  ##########################
  local_rank = int(os.environ['LOCAL_RANK']) # local_rank是当前进程的id
  world_size = int(os.environ['WORLD_SIZE']) # world_size是进程总数
  # DDP需要初始化Process Group
  torch.distributed.init_process_group("nccl", init_method='env://', rank=local_rank, world_size=world_size)
  
  torch.autograd.set_detect_anomaly(True)
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  # if opt.kvobj:
  #     Dataset.num_classes = 3
  # else:
  #     Dataset.num_classes = 2
  Dataset.num_classes = opt.class_num + 1
  
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  opt.device=torch.device("cuda", local_rank)
  torch.cuda.set_device(local_rank)
  
  # wandb可视化设置
  if not os.path.exists(os.path.join(opt.save_dir, 'wandb')):
    os.makedirs(os.path.join(opt.save_dir, 'wandb'))
  wandb_config = dict(exp_id = opt.exp_id,
                      dataset=opt.dataset,
                      lr=opt.lr,
                      batch_size=opt.batch_size,
                      epoch=opt.num_epochs, 
                      arch=opt.arch,
                      tsfm_layers=opt.tsfm_layers,
                      stacking_layers=opt.stacking_layers,
                      class_num=opt.class_num,
                      )
  wandb.init(project='LORE', 
             name='LORE_ic19_1224', 
             config=wandb_config,
             dir=os.path.join(opt.save_dir, 'wandb'))

  if local_rank == 0:
    logger = Logger(opt)

  ########################## 	DDP change 2  ##########################
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  if opt.load_model != '':
    #model, optimizer, start_epoch = load_model(model, opt.load_model)
      #model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    model = load_model(model, opt.load_model, opt.device)
  # freeze the detector
  if opt.only_processor:
    model = freeze_model(model)
  model = model.to(opt.device)
  torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # change for BN in DDP
  
  processor = Processor(opt)
  if opt.load_processor != '':
    processor = load_model(processor, opt.load_processor, opt.device)
  processor = processor.to(opt.device)
  
  if not opt.only_processor:
    optimizer = torch.optim.Adam([  \
                {'params': model.parameters()}, \
                {'params': processor.parameters()}],  lr =opt.lr, betas= (0.9, 0.98), eps=1e-9)
  else:
    optimizer = torch.optim.Adam(processor.parameters(), lr =opt.lr, betas= (0.9, 0.98), eps=1e-9)
  # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1)
  # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_step, gamma=0.1, last_epoch=-1)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.num_epochs, eta_min=0, last_epoch=-1)


  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer, processor)
  trainer.model_with_loss = DDP(trainer.model_with_loss, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
  # trainer.model_with_loss.eval()
  start_epoch = 0
    
  ########################## 	DDP change 3  ##########################
  #处理Dataloader DDP sampler修改
  train_dataset = Dataset(opt, 'train')
  if not os.path.exists(os.path.join(opt.anno_path, 'test.json')):
    val_dataset = Dataset(opt, 'val')
  else:
    val_dataset = Dataset(opt, 'test')
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
  val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
  #torch.utils.data.DataLoader中的shuffle应该设置为False（默认），因为打乱的任务交给了sampler
  
  val_loader = torch.utils.data.DataLoader(
      val_dataset, 
      batch_size=1, 
      shuffle=False,
      sampler=val_sampler,
      num_workers=0, # 1
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      train_dataset, 
      batch_size=opt.batch_size, 
      shuffle=False,
      sampler=train_sampler,
      num_workers=0,
      pin_memory=True,
      drop_last=True
  )
  
  if local_rank == 0:
    print('Starting training...')
  best = 1e10

  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    ########################## 	DDP change 4  ##########################
    train_loader.sampler.set_epoch(epoch) 
    log_dict_train, _ = trainer.train(epoch, train_loader)
    scheduler.step()
    wandb.log({'loss': log_dict_train['loss'], 
               'ax_l': log_dict_train['ax_l'],
               'sax_l': log_dict_train['sax_l'],
               'sp_l': log_dict_train['sp_l'],
               'sm_l': log_dict_train['sm_l'],
               'lr': optimizer.state_dict()['param_groups'][0]['lr']
              })

    ########################## 	DDP change 5  ##########################
    if local_rank == 0:
      logger.write('epoch: {} |'.format(epoch))    
      for k, v in log_dict_train.items():
        logger.scalar_summary('train_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      
      if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
        save_model(os.path.join(opt.save_dir, 'processor_{}.pth'.format(mark)), 
                  epoch, processor, optimizer)
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                  epoch, model, optimizer)
        with torch.no_grad():
          log_dict_val, preds = trainer.val(epoch, val_loader)
        for k, v in log_dict_val.items():
          logger.scalar_summary('val_{}'.format(k), v, epoch)
          logger.write('{} {:8f} | '.format(k, v))

        if log_dict_val[opt.metric] < best:
          best = log_dict_val[opt.metric]
          save_model(os.path.join(opt.save_dir, 'processor_best.pth'), 
                    epoch, processor)
          save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                    epoch, model)
      else:
        save_model(os.path.join(opt.save_dir, 'processor_last.pth'), 
                  epoch, processor, optimizer)
        save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                  epoch, model, optimizer)
      logger.write('\n')
      # if epoch in opt.lr_step:
      #   save_model(os.path.join(opt.save_dir, 'processor_{}.pth'.format(epoch)), 
      #             epoch, processor, optimizer)
      #   save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
      #             epoch, model, optimizer)
      #   lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      #   print('Drop LR to', lr)
      #   for param_group in optimizer.param_groups:
      #       param_group['lr'] = lr
  wandb.finish()
  if local_rank == 0:
    logger.close()
  torch.distributed.destroy_process_group()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
