"""
The implementation in this repo was developed upon the ACNet code in the repo https://github.com/DingXiaoH/ACNet.
Changes were made to implement and run KFLO.
"""
from base_config import get_baseconfig_by_epoch
from model_map import get_dataset_name_by_model_name
import argparse
from kflo.kflo_builder import KFLOBuilder
from ndp_train import train_main
import os
# os.environ['PYTHONPATH'] = '.'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from constants import LRSchedule
from builder import ConvBuilder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', default='cfqkbnc')
    parser.add_argument('-b', '--block_type', default='kflo')
    parser.add_argument('-c', '--conti_or_fs', default='fs')        # continue or train_from_scratch
    parser.add_argument(
        '--local_rank', default=0, type=int,
        help='process rank on node')

    start_arg = parser.parse_args()

    network_type = start_arg.arch
    block_type = start_arg.block_type
    conti_or_fs = start_arg.conti_or_fs
    assert conti_or_fs in ['continue', 'fs']
    assert block_type in ['kflo', 'base']
    auto_continue = conti_or_fs == 'continue'
    print('auto continue: ', auto_continue)

    if network_type == 'cfqkbnc':
        weight_decay_strength = 1e-4
        batch_size = 128
        lrs = LRSchedule(base_lr=0.1, max_epochs=150, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        #   ------------------------------------

    elif network_type == 'vc':
        weight_decay_strength = 1e-4
        batch_size = 128
        lrs = LRSchedule(base_lr=0.1, max_epochs=400, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        warmup_epochs = 0
        #   --------------------------------------
    else:
        raise ValueError('...')

    log_dir = 'kflo_exps/{}_{}_train'.format(network_type, block_type)

    weight_decay_bias = weight_decay_strength
    config = get_baseconfig_by_epoch(network_type=network_type,
                                     dataset_name=get_dataset_name_by_model_name(network_type), dataset_subset='train',
                                     global_batch_size=batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=0.9,
                                     max_epochs=lrs.max_epochs, base_lr=lrs.base_lr, lr_epoch_boundaries=lrs.lr_epoch_boundaries, cosine_minimum=lrs.cosine_minimum,
                                     lr_decay_factor=lrs.lr_decay_factor,
                                     warmup_epochs=0, warmup_method='linear', warmup_factor=0,
                                     ckpt_iter_period=40000, tb_iter_period=100, output_dir=log_dir,
                                     tb_dir=log_dir, save_weights=None, val_epoch_period=5, linear_final_lr=lrs.linear_final_lr,
                                     weight_decay_bias=weight_decay_bias, deps=None)

    if block_type == 'kflo':
        builder = KFLOBuilder(base_config=config, deploy=False)
    else:
        builder = ConvBuilder(base_config=config)

    target_weights = os.path.join(log_dir, 'finish.hdf5')
    if not os.path.exists(target_weights):
        train_main(local_rank=start_arg.local_rank, cfg=config, convbuilder=builder,
               show_variables=True, auto_continue=auto_continue)

