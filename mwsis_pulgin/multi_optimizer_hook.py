# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn import GroupNorm, LayerNorm

from mmcv.utils import _BatchNorm, _InstanceNorm, build_from_cfg, is_list_of
from mmcv.utils.ext_loader import check_ops_exist
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.runner import DefaultOptimizerConstructor

@OPTIMIZER_BUILDERS.register_module()
class MultiTaskOptimizerConstructor(DefaultOptimizerConstructor):
    """Default constructor for optimizers.

    By default each parameter share the same optimizer settings, and we
    provide an argument ``paramwise_cfg`` to specify parameter-wise settings.
    It is a dict and may contain the following fields:

    - ``custom_keys`` (dict): Specified parameters-wise settings by keys. If
      one of the keys in ``custom_keys`` is a substring of the name of one
      parameter, then the setting of the parameter will be specified by
      ``custom_keys[key]`` and other setting like ``bias_lr_mult`` etc. will
      be ignored. It should be noted that the aforementioned ``key`` is the
      longest key that is a substring of the name of the parameter. If there
      are multiple matched keys with the same length, then the key with lower
      alphabet order will be chosen.
      ``custom_keys[key]`` should be a dict and may contain fields ``lr_mult``
      and ``decay_mult``. See Example 2 below.
    - ``bias_lr_mult`` (float): It will be multiplied to the learning
      rate for all bias parameters (except for those in normalization
      layers and offset layers of DCN).
    - ``bias_decay_mult`` (float): It will be multiplied to the weight
      decay for all bias parameters (except for those in
      normalization layers, depthwise conv layers, offset layers of DCN).
    - ``norm_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of normalization
      layers.
    - ``dwconv_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of depthwise conv
      layers.
    - ``dcn_offset_lr_mult`` (float): It will be multiplied to the learning
      rate for parameters of offset layer in the deformable convs
      of a model.
    - ``bypass_duplicate`` (bool): If true, the duplicate parameters
      would not be added into optimizer. Default: False.

    Note:

        1. If the option ``dcn_offset_lr_mult`` is used, the constructor will
        override the effect of ``bias_lr_mult`` in the bias of offset layer.
        So be careful when using both ``bias_lr_mult`` and
        ``dcn_offset_lr_mult``. If you wish to apply both of them to the offset
        layer in deformable convs, set ``dcn_offset_lr_mult`` to the original
        ``dcn_offset_lr_mult`` * ``bias_lr_mult``.

        2. If the option ``dcn_offset_lr_mult`` is used, the constructor will
        apply it to all the DCN layers in the model. So be careful when the
        model contains multiple DCN layers in places other than backbone.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are

                - `type`: class name of the optimizer.

            Optional fields are

                - any arguments of the corresponding optimizer type, e.g.,
                  lr, weight_decay, momentum, etc.
        paramwise_cfg (dict, optional): Parameter-wise options.

    Example 1:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> paramwise_cfg = dict(norm_decay_mult=0.)
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)

    Example 2:
        >>> # assume model have attribute model.backbone and model.cls_head
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, weight_decay=0.95)
        >>> paramwise_cfg = dict(custom_keys={
                'backbone': dict(lr_mult=0.1, decay_mult=0.9)})
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)
        >>> # Then the `lr` and `weight_decay` for model.backbone is
        >>> # (0.01 * 0.1, 0.95 * 0.9). `lr` and `weight_decay` for
        >>> # model.cls_head is (0.01, 0.95).
    """

    def __init__(self,
                 optimizer_cfg: Dict,
                 paramwise_cfg: Optional[Dict] = None):
        if not isinstance(optimizer_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            f'but got {type(optimizer_cfg)}')
        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = {} if paramwise_cfg is None else paramwise_cfg
        # self.base_lr = optimizer_cfg.get('lr', None)
        # self.base_wd = optimizer_cfg.get('weight_decay', None)
        self.img_optimizer_cfg = optimizer_cfg.get('img', None)
        self.pts_optimizer_cfg = optimizer_cfg.get('pts', None)
        self.multi_task = optimizer_cfg.get('multi_task', False)
        self._validate_cfg()
        self.nums = 0

    def _validate_cfg(self) -> None:
        if self.multi_task:
            if self.img_optimizer_cfg is None or self.pts_optimizer_cfg is None:
                raise ValueError('img and pts optimizer should not be None')
        else:
            if not isinstance(self.paramwise_cfg, dict):
                raise TypeError('paramwise_cfg should be None or a dict, '
                                f'but got {type(self.paramwise_cfg)}')

            if 'custom_keys' in self.paramwise_cfg:
                if not isinstance(self.paramwise_cfg['custom_keys'], dict):
                    raise TypeError(
                        'If specified, custom_keys must be a dict, '
                        f'but got {type(self.paramwise_cfg["custom_keys"])}')
                if self.base_wd is None:
                    for key in self.paramwise_cfg['custom_keys']:
                        if 'decay_mult' in self.paramwise_cfg['custom_keys'][key]:
                            raise ValueError('base_wd should not be None')

            # get base lr and weight decay
            # weight_decay must be explicitly specified if mult is specified
            if ('bias_decay_mult' in self.paramwise_cfg
                    or 'norm_decay_mult' in self.paramwise_cfg
                    or 'dwconv_decay_mult' in self.paramwise_cfg):
                if self.base_wd is None:
                    raise ValueError('base_wd should not be None')

    def _is_in(self, param_group: Dict, param_group_list: List) -> bool:
        assert is_list_of(param_group_list, dict)
        param = set(param_group['params'])
        param_set = set()
        for group in param_group_list:
            param_set.update(set(group['params']))

        return not param.isdisjoint(param_set)
  
    def multi_task_add_params(self,
                   params: List[Dict],
                   module: nn.Module,
                   multi_task_params: Dict,
                   prefix: str = '',
                   is_dcn_module: Union[int, float, None] = None) -> None:
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', 1.)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', 1.)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', 1.)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', 1.)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)
        dcn_offset_lr_mult = self.paramwise_cfg.get('dcn_offset_lr_mult', 1.)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
            isinstance(module, torch.nn.Conv2d)
            and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            # 检查是否有非img/pts字段
            if 'pts' not in prefix and 'img' not in prefix:
                print("漏网之鱼{}".format(prefix + name))
                raise ValueError('参数不属于pts和img')
            if not param.requires_grad:
                params.append(param_group)
                multi_task_params['not_requires_grad'].append(param_group)  # ({prefix+name: param_group})
                continue
            if bypass_duplicate and self._is_in(param_group, params):
                warnings.warn(f'{prefix} is duplicate. It is skipped since '
                              f'bypass_duplicate={bypass_duplicate}')
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in sorted_keys:
                if key in f'{prefix}.{name}':
                    is_custom = True
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    self.base_lr = self.img_optimizer_cfg.get('lr', 1.) if 'img' in prefix else self.pts_optimizer_cfg.get('lr', 1.)
                    param_group['lr'] = self.base_lr * lr_mult
                    self.base_wd = self.img_optimizer_cfg.get('weight_decay', None) if 'img' in prefix else self.pts_optimizer_cfg.get('weight_decay', None)
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                        if decay_mult == 0:
                            self.nums = self.nums + 1
                    break

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                self.base_lr = self.img_optimizer_cfg.get('lr', None) if 'img' in prefix else self.pts_optimizer_cfg.get('lr', None)
                self.base_wd = self.img_optimizer_cfg.get('weight_decay', None) if 'img' in prefix else self.pts_optimizer_cfg.get('weight_decay', None)
                if name == 'bias' and not (is_norm or is_dcn_module):
                    param_group['lr'] = self.base_lr * bias_lr_mult

                if (prefix.find('conv_offset') != -1 and is_dcn_module
                        and isinstance(module, torch.nn.Conv2d)):
                    # deal with both dcn_offset's bias & weight
                    
                    param_group['lr'] = self.base_lr * dcn_offset_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    # norm decay
                    if is_norm:
                        param_group[
                            'weight_decay'] = self.base_wd * norm_decay_mult
                    # depth-wise conv
                    elif is_dwconv:
                        param_group[
                            'weight_decay'] = self.base_wd * dwconv_decay_mult
                    # bias lr and decay
                    elif name == 'bias' and not is_dcn_module:
                        # TODO: current bias_decay_mult will have affect on DCN
                        param_group[
                            'weight_decay'] = self.base_wd * bias_decay_mult
            params.append(param_group)
            if 'img' in prefix:
                multi_task_params['img'].append(param_group)  # ({prefix+name: param_group})
            elif 'pts' in prefix:
                multi_task_params['pts'].append(param_group)  # ({prefix+name: param_group})
            else:
                raise ValueError('{} has not set lr, weight decay'.format(name))
        if check_ops_exist():
            from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
            is_dcn_module = isinstance(module,
                                       (DeformConv2d, ModulatedDeformConv2d))
        else:
            is_dcn_module = False
        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.multi_task_add_params(
                params,
                child_mod,
                multi_task_params,
                prefix=child_prefix,
                is_dcn_module=is_dcn_module)

    def __call__(self, model: nn.Module):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.parameters()  # [*model.parameters()]
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # set param-wise lr and weight decay recursively
        params: List[Dict] = []
        multi_task_params: Dict = {'img':[], 'pts':[], 'not_requires_grad':[]}
        if self.multi_task:
            self.multi_task_add_params(params, model, multi_task_params)
        else:
            self.add_params(params, model)
        optimizer_cfg['params'] = params
        optimizer_cfg['multi_task_params'] = multi_task_params
        assert len(params) == len(multi_task_params['img']) + len(multi_task_params['pts']) + len(multi_task_params['not_requires_grad'])
        multi_task_optimizer = dict()
        for i, type in enumerate(['img', 'pts']):
            optimizer_cfg_ = dict()
            optimizer_cfg_.update(optimizer_cfg[type])
            optimizer_cfg_.update(type=optimizer_cfg['type'][type])
            if len(optimizer_cfg['multi_task_params'][type]) != 0:
                optimizer_cfg_.update(params=optimizer_cfg['multi_task_params'][type])
                multi_task_optimizer[type] = build_from_cfg(optimizer_cfg_, OPTIMIZERS)
        # self.nums
        return multi_task_optimizer
        # return build_from_cfg(optimizer_cfg, OPTIMIZERS)


# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
from collections import defaultdict
from itertools import chain
from typing import Optional, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad

from mmcv.utils import TORCH_VERSION, _BatchNorm, digit_version
from mmcv.runner.dist_utils import allreduce_grads
from mmcv.runner.fp16_utils import LossScaler, wrap_fp16_model
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.runner.hooks import OptimizerHook
try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    from torch.cuda.amp import GradScaler
except ImportError:
    pass

@HOOKS.register_module()
class MultiTaskOptimizerHook(OptimizerHook):
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Default: None.
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.
    """

    def __init__(self,
                 grad_clip: Optional[dict] = None,
                 detect_anomalous_params: bool = False):
        self.grad_clip = grad_clip  # list
        self.detect_anomalous_params = detect_anomalous_params
        self.img_params = []
        self.pts_params = []
        self.flag = 0
        self.frozen = 0
    def clip_grads(self, named_params, id, key):
        # params = list(
        #     filter(lambda p: p[0].requires_grad and p[0].grad is not None and key in p[1], named_params))
        params = [value for name, value in named_params if (key in name and value.requires_grad and value.grad is not None)]
        if len(params) > 0 and self.grad_clip[id] is not None:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip[id])

    def filter_params(self, named_params, key):
        params = []
        for name, value in named_params():
            if key in name and value.requireds_grad and value.grad is not None:
                params.append(value)
        return params

    def after_train_iter(self, runner):

        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        
        # # test 检查不同分支是否有反向传播穿透
        # check_key = 'img'
        # no_check_key = 'pts'
        # if self.flag == 0:
        #     pts_param_groups = runner.optimizer[check_key].param_groups
        #     for i in range(len(pts_param_groups)):
        #         self.pts_params.append(pts_param_groups[i]['params'][0].data.clone().detach())
        #     img_param_groups = runner.optimizer[no_check_key].param_groups
        #     for i in range(len(img_param_groups)):
        #         self.img_params.append(img_param_groups[i]['params'][0].data.clone().detach())
        # self.frozen = 0
        # if self.flag == 1:
        #     pts_param_groups = runner.optimizer[check_key].param_groups
        #     for i in range(len(pts_param_groups)):
        #         assert (pts_param_groups[i]['params'][0].data == self.pts_params[i]).all(), "反向传播穿透"
        #         # self.pts_params[i] = pts_param_groups[i]['params'][0].data.clone().detach()
        #     img_param_groups = runner.optimizer[no_check_key].param_groups
        #     for i in range(len(img_param_groups)):
        #         if no_check_key == 'pts':
        #             assert (img_param_groups[i]['params'][0].data != self.img_params[i]).any(), "反向传播穿透"
        #         else:
        #             if not (img_param_groups[i]['params'][0].data != self.img_params[i]).any():
        #                 self.frozen += 1
        #         # self.img_params[i] = img_param_groups[i]['params'][0].data.clone().detach()
        #     # if no_check_key == 'img':
        #     #     print("weight相同的参数{}".format(self.frozen))
        # self.flag = 1

        for key, optimizer in runner.optimizer.items():
            # if key == check_key:
            #     continue
            optimizer.zero_grad()  # 梯度清0

        runner.outputs['loss'].backward()

        for key, optimizer in runner.optimizer.items():
            i = 0 if key == 'img' else 1
            # if key == check_key:
            #     continue
            if self.grad_clip[i] is not None:
                grad_norm = self.clip_grads(runner.model.named_parameters(), int(i), key=key)
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                            runner.outputs['num_samples'])
        
        for key, optimizer in runner.optimizer.items():
            # if key == check_key:
            #     continue
            optimizer.step()
            # lr = 0
            # initial_lr = 0
            # weight_decay = 0
            # for i, param_ in enumerate(optimizer.param_groups):
            #     if i == 0:
            #         lr = param_['lr']
            #         initial_lr = param_['initial_lr']
            #         if key =='pts':
            #             weight_decay = 0.05
            #         else:
            #             weight_decay = param_['weight_decay']
            #     if param_['weight_decay'] == 0:
            #         print('weight_decay is 0')
            #         continue
            #     assert lr==param_['lr'] and initial_lr == param_['initial_lr'] and weight_decay == param_['weight_decay']

    def detect_anomalous_parameters(self, loss: Tensor, runner) -> None:
        logger = runner.logger
        parameters_in_graph = set()
        visited = set()

        def traverse(grad_fn):
            if grad_fn is None:
                return
            if grad_fn not in visited:
                visited.add(grad_fn)
                if hasattr(grad_fn, 'variable'):
                    parameters_in_graph.add(grad_fn.variable)
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        grad_fn = parent[0]
                        traverse(grad_fn)

        traverse(loss.grad_fn)
        for n, p in runner.model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                logger.log(
                    level=logging.ERROR,
                    msg=f'{n} with shape {p.size()} is not '
                    f'in the computational graph \n')

# Copyright (c) OpenMMLab. All rights reserved.
import numbers
from math import cos, pi
from typing import Callable, List, Optional, Union

import mmcv
from mmcv import runner
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.hooks import LrUpdaterHook, StepLrUpdaterHook, CyclicLrUpdaterHook

@HOOKS.register_module()
class MultiLrUpdaterHook(LrUpdaterHook):
    """Cyclic LR Scheduler.

    Implement the cyclical learning rate policy (CLR) described in
    https://arxiv.org/pdf/1506.01186.pdf

    Different from the original paper, we use cosine annealing rather than
    triangular policy inside a cycle. This improves the performance in the
    3D detection area.

    Args:
        by_epoch (bool, optional): Whether to update LR by epoch.
        target_ratio (tuple[float], optional): Relative ratio of the highest LR
            and the lowest LR to the initial LR.
        cyclic_times (int, optional): Number of cycles during training
        step_ratio_up (float, optional): The ratio of the increasing process of
            LR in the total cycle.
        anneal_strategy (str, optional): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing. Default: 'cos'.
        gamma (float, optional): Cycle decay ratio. Default: 1.
            It takes values in the range (0, 1]. The difference between the
            maximum learning rate and the minimum learning rate decreases
            periodically when it is less than 1. `New in version 1.4.4.`
    """

    def __init__(self,
                 by_epoch: bool = False,
                 warmup: Optional[str] = None,
                 warmup_iters: int = 0,
                 warmup_ratio: float = 0.1,
                 warmup_by_epoch: bool = False,
                 img=None,
                 pts=None,
                 **kwargs) -> None:
        # img / pts
        img_lr_config = img
        pts_lr_config = pts
        if img_lr_config is None or pts_lr_config is None:
            raise ValueError('img and pts lr_config should not be None')
        img_lr_config = self.update_config(img_lr_config)
        pts_lr_config = self.update_config(pts_lr_config)
        self.img_lr_hook = mmcv.build_from_cfg(img_lr_config, HOOKS)
        self.pts_lr_hook = mmcv.build_from_cfg(pts_lr_config, HOOKS)
        self.is_pts_linear = False
        super().__init__(by_epoch, **kwargs)

    def update_config(self, lr_config):
        assert 'policy' in lr_config
        policy_type = lr_config.pop('policy')
        # If the type of policy is all in lower case, e.g., 'cyclic',
        # then its first letter will be capitalized, e.g., to be 'Cyclic'.
        # This is for the convenient usage of Lr updater.
        # Since this is not applicable for `
        # CosineAnnealingLrUpdater`,
        # the string will not be changed if it contains capital letters.
        if policy_type == policy_type.lower():
            policy_type = policy_type.title()
        # update hook config
        hook_type = policy_type + 'LrUpdaterHook'
        lr_config['type'] = hook_type
        return lr_config

    def _set_lr(self, runner, lr_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(runner.optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    def before_run(self, runner: 'runner.BaseRunner'):
        # init lr
        super().before_run(runner)  # self.base_lr.update({k: _base_lr})
        self.img_lr_hook.before_run(runner) # is same as super().before_run(runner)
        self.pts_lr_hook.before_run(runner)

    def get_regular_lr(self, runner: 'runner.BaseRunner'):
        if isinstance(runner.optimizer, dict):
            lr_groups = {}
            for k in runner.optimizer.keys():
                if k == 'img':
                    _lr_group = [
                        self.img_lr_hook.get_lr(runner, _base_lr)
                        for _base_lr in self.base_lr[k]
                    ]
                elif k == 'pts':  # if pts is fixed, can be ignored: lr=base_lr
                    _lr_group = [
                        self.pts_lr_hook.get_lr(runner, _base_lr)
                        for _base_lr in self.base_lr[k]
                    ]
                else:
                    raise ValueError('img and pts lr_config should not be None')                    
                lr_groups.update({k: _lr_group})

            return lr_groups
        else:
            return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def before_train_epoch(self, runner: 'runner.BaseRunner'):
        self.img_lr_hook.before_train_epoch(runner)
        self.pts_lr_hook.before_train_epoch(runner)
        if self.warmup_iters is None:
            epoch_len = len(runner.data_loader)  # type: ignore
            self.warmup_iters = self.warmup_epochs * epoch_len  # type: ignore

        if not self.by_epoch:
            return

        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner: 'runner.BaseRunner'):
        cur_iter = runner.iter
        assert isinstance(self.warmup_iters, int)
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)  # get base lr
            self.img_lr_hook.regular_lr = self.regular_lr['img']
            if self.is_pts_linear is False:
                try:
                    if 'linear' in self.pts_lr_hook.warmup:
                        self.pts_lr_hook.regular_lr = self.regular_lr['pts']
                        self.is_pts_linear = True
                except:
                    # print('pts lr is not linear')
                    pass
            if self.img_lr_hook.warmup is None or cur_iter >= self.img_lr_hook.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                self.regular_lr['img'] = self.img_lr_hook.get_warmup_lr(cur_iter)
                self._set_lr(runner, self.regular_lr)
            if self.is_pts_linear:
                if self.pts_lr_hook.warmup is None or cur_iter >= self.pts_lr_hook.warmup_iters:
                    self._set_lr(runner, self.regular_lr)
                else:
                    self.regular_lr['pts'] = self.pts_lr_hook.get_warmup_lr(cur_iter)
                    self._set_lr(runner, self.regular_lr)                
        # not use
        elif self.by_epoch:
            raise ValueError('self.by_epoch need False')   
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)


def annealing_cos(start: float,
                  end: float,
                  factor: float,
                  weight: float = 1.) -> float:
    """Calculate annealing cos learning rate.

    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out


def annealing_linear(start: float, end: float, factor: float) -> float:
    """Calculate annealing linear learning rate.

    Linear anneal from `start` to `end` as percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the linear annealing.
        end (float): The ending learing rate of the linear annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
    """
    return start + (end - start) * factor


def format_param(name, optim, param):
    if isinstance(param, numbers.Number):
        return [param] * len(optim.param_groups)
    elif isinstance(param, (list, tuple)):  # multi param groups
        if len(param) != len(optim.param_groups):
            raise ValueError(f'expected {len(optim.param_groups)} '
                             f'values for {name}, got {len(param)}')
        return param
    else:  # multi optimizers
        if name not in param:
            raise KeyError(f'{name} is not found in {param.keys()}')
        return param[name]