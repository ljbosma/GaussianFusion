from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import shutil

import torch
import torch.utils.data
from lib.opts import opts
from lib.model.model import create_model, load_model, save_model
from lib.model.data_parallel import DataParallel
from lib.logger import Logger
from lib.dataset.dataset_factory import get_dataset
from lib.trainer import Trainer
from test import prefetch_test
import json

def get_optimizer(opt, model):
  if opt.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  elif opt.optim == 'sgd':
    print('Using SGD')
    optimizer = torch.optim.SGD(
      model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
  else:
    assert 0, opt.optim
  return optimizer

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.eval
  Dataset = get_dataset(opt.dataset)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  logger = Logger(opt)
  loss_keys_2d = ['hm', 'wh', 'reg']
  loss_keys_3d = ['dep', 'dep_sec', 'dim', 'rot', 'rot_sec', 'amodel_offset', 'nuscenes_att', 'velocity']

  history = {}

  print('Creating model...')
  model = create_model(
  opt.arch,
  opt.heads,
  opt.head_conv,
  local_pretrained_path="../models/centernet_model_best_20_epochs.pth",
  opt=opt
)

  optimizer = get_optimizer(opt, model)
  start_epoch = 0
  lr = opt.lr

  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, opt, optimizer)

  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  if opt.val_intervals < opt.num_epochs or opt.eval:
    print('Setting up validation data...')
    val_loader = torch.utils.data.DataLoader(
      Dataset(opt, opt.val_split), batch_size=1, shuffle=False, 
              num_workers=1, pin_memory=True)

    if opt.eval:
      _, preds = trainer.val(0, val_loader)
      val_loader.dataset.run_eval(preds, opt.save_dir, n_plots=opt.eval_n_plots, 
                                  render_curves=opt.eval_render_curves)
      return

  print('Setting up train data...')
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, opt.train_split), batch_size=opt.batch_size, 
        shuffle=opt.shuffle_train, num_workers=opt.num_workers, 
        pin_memory=True, drop_last=True
  )

  print('Starting training...')

  # === Early stopping setup ===
  best_metric = float('inf')
  epochs_without_improvement = 0
  if not hasattr(opt, 'early_stop_patience'):
    opt.early_stop_patience = 15

  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        # === 3-Phase Curriculum Learning Schedule ===
    phase1_end = 5     # Frozen backbone, 2D OD warmup
    phase2_end = 20    # Full model, 2D + light 3D
    phase3_start = phase2_end + 1  # Full 3D focus

    if epoch <= phase1_end:
        opt.freeze_backbone = True
        opt.weights.update({
            'hm': 2.0, 'wh': 1.0, 'reg': 2.0,
            'dep': 0.1, 'dep_sec': 0.1,
            'dim': 0.1, 'rot': 0.1, 'rot_sec': 0.1,
            'amodel_offset': 0.1, 'nuscenes_att': 0.1,
            'velocity': 0.1
        })
        if epoch == 1:
            print("🧊 Phase 1: Frozen backbone, strong 2D supervision")

    elif epoch <= phase2_end:
        opt.freeze_backbone = False
        for param in model.parameters():
            param.requires_grad = True
        opt.weights.update({
            'hm': 1.0, 'wh': 0.5, 'reg': 1.0,
            'dep': 0.5, 'dep_sec': 0.5,
            'dim': 0.5, 'rot': 0.5, 'rot_sec': 0.5,
            'amodel_offset': 0.5, 'nuscenes_att': 0.5,
            'velocity': 0.5
        })
        if epoch == phase1_end + 1:
            print("🪄 Phase 2: Backbone unfrozen, start co-adaptation")

    else:
        opt.freeze_backbone = False
        opt.weights.update({
            'hm': 0.5, 'wh': 0.1, 'reg': 0.5,
            'dep': 2.0, 'dep_sec': 2.0,
            'dim': 2.0, 'rot': 2.0, 'rot_sec': 2.0,
            'amodel_offset': 1.0, 'nuscenes_att': 1.0,
            'velocity': 2.0
        })
        if epoch == phase3_start:
            print("🚀 Phase 3: Full 3D OD focus")

    mark = epoch if opt.save_all else 'last'

    # log learning rate
    for param_group in optimizer.param_groups:
      lr = param_group['lr']
      logger.scalar_summary('LR', lr, epoch)
      break

    # train one epoch
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))

    # log train results
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))

    # evaluate
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)

        # dataset-specific eval
        if opt.run_dataset_eval:
          out_dir = val_loader.dataset.run_eval(preds, opt.save_dir, 
                                                n_plots=opt.eval_n_plots, 
                                                render_curves=opt.eval_render_curves)
          # Load metrics from evaluation
          with open('{}/metrics_summary.json'.format(out_dir), 'r') as f:
              metrics = json.load(f)

          # Save metrics and losses for this epoch
          epoch_metrics = metrics  # Save all available metrics, including NDS and COCO scores

          epoch_losses = {k: v for k, v in log_dict_val.items()}

          history[epoch] = {
              "metrics": epoch_metrics,
              "losses": epoch_losses
          }

          # Log eval results
          for k, v in log_dict_val.items():
            logger.scalar_summary('val_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))


        # # === Early stopping ===
        if epoch >= phase3_start:
          current_metric = -metrics.get('NDS', 0.0)  # Maximize NDS → minimize negative
          print(f"[EarlyStopping - Phase 3] Using NDS = {metrics.get('NDS', 0.0):.6f}")

          if current_metric < best_metric:
              best_metric = current_metric
              epochs_without_improvement = 0
              save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                          epoch, model, optimizer)

              if epoch > 20:
                  # Copy debug folder to debug_best
                  debug_dir = os.path.join(opt.save_dir, 'debug')
                  debug_best_dir = os.path.join(opt.save_dir, 'debug_best')

                  if os.path.exists(debug_dir):
                    if os.path.exists(debug_best_dir):
                        shutil.rmtree(debug_best_dir)
                    os.rename(debug_dir, debug_best_dir)
                    os.makedirs(debug_dir)
                    print(f"📁 Renamed {debug_dir} to {debug_best_dir} and recreated {debug_dir}")
          else:
              epochs_without_improvement += 1
              print(f"No improvement for {epochs_without_improvement} epoch(s).")

          if epochs_without_improvement >= opt.early_stop_patience:
              print(f"Early stopping at epoch {epoch} (best NDS: {-best_metric:.4f})")
              break

    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)

    logger.write('\n')
    if epoch in opt.save_point:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)

    if epoch in opt.lr_step:
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

  # Save full history to JSON after training
  history_path = os.path.join(opt.save_dir, 'training_history.json')
  with open(history_path, 'w') as f:
      json.dump(history, f, indent=2)

  print(f"Saved full training history to {history_path}")
  logger.write(f"Saved full training history to {history_path}\n")

  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)