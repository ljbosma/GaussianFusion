from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.dla import DLASeg
from .networks.resdcn import PoseResDCN
from .networks.resnet import PoseResNet
from .networks.dlav0 import DLASegv0
from .networks.generic_network import GenericNetwork

_network_factory = {
  'resdcn': PoseResDCN,
  'dla': DLASeg,
  'res': PoseResNet,
  'dlav0': DLASegv0,
  'generic': GenericNetwork
}

def create_model(arch, head, head_conv, local_pretrained_path=None, opt=None):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  model_class = _network_factory[arch]
  model = model_class(num_layers, heads=head, head_convs=head_conv, local_pretrained_path="../models/dla34_ba72cf86.pth", opt=opt)

  if local_pretrained_path is not None:
    print(f"Loading pretrained weights from {local_pretrained_path}")
    ckpt = torch.load(local_pretrained_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model_state_dict = model.state_dict()

    # Match and copy weights
    new_state_dict = {}
    for k in state_dict:
      if k in model_state_dict and state_dict[k].shape == model_state_dict[k].shape:
        new_state_dict[k] = state_dict[k]
        print(f"Loaded weight for: {k} with shape state {state_dict[k].shape} and model shape {model_state_dict[k].shape}")
      else:
        print(f"Skipped (no match or shape mismatch): {k} with state_dict shape {state_dict[k].shape} and model shape {model_state_dict[k].shape}")

    model.load_state_dict(new_state_dict, strict=False)
    print("=> Loaded CenterNet weights (partial load: backbone + matched heads)")

  return model


def load_model(model, model_path, opt, optimizer=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
   
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  for k in state_dict:
    if k in model_state_dict:
      if (state_dict[k].shape != model_state_dict[k].shape) or \
        (opt.reset_hm and k.startswith('hm') and (state_dict[k].shape[0] in [80, 1])):
        if opt.reuse_hm:
          print('Reusing parameter {}, required shape{}, '\
                'loaded shape{}.'.format(
            k, model_state_dict[k].shape, state_dict[k].shape))
          # todo: bug in next line: both sides of < are the same
          if state_dict[k].shape[0] < state_dict[k].shape[0]:
            model_state_dict[k][:state_dict[k].shape[0]] = state_dict[k]
          else:
            model_state_dict[k] = state_dict[k][:model_state_dict[k].shape[0]]
          state_dict[k] = model_state_dict[k]
        
        elif opt.warm_start_weights:
          try:
            print('Partially loading parameter {}, required shape{}, '\
                  'loaded shape{}.'.format(
              k, model_state_dict[k].shape, state_dict[k].shape))
            if state_dict[k].shape[1] < model_state_dict[k].shape[1]:
              model_state_dict[k][:,:state_dict[k].shape[1]] = state_dict[k]
            else:
              model_state_dict[k] = state_dict[k][:,:model_state_dict[k].shape[1]]
            state_dict[k] = model_state_dict[k]
          except:
            print('Skip loading parameter {}, required shape{}, '\
                'loaded shape{}.'.format(
                k, model_state_dict[k].shape, state_dict[k].shape))
            state_dict[k] = model_state_dict[k]
        
        else:
          print('Skip loading parameter {}, required shape{}, '\
                'loaded shape{}.'.format(
            k, model_state_dict[k].shape, state_dict[k].shape))
          state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k))
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k))
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # freeze backbone network
  if opt.freeze_backbone:
    for (name, module) in model.named_children():
      if name in opt.layers_to_freeze:
        for (name, layer) in module.named_children():
          for param in layer.parameters():
            param.requires_grad = False

  # resume optimizer parameters
  if optimizer is not None and opt.resume:
    if 'optimizer' in checkpoint:
      start_epoch = checkpoint['epoch']
      start_lr = opt.lr
      for step in opt.lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

