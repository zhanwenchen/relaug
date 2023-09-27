# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
from os import environ as os_environ
import time
import datetime

import torch
from torch.nn.utils import clip_grad_norm_

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader, VGStats
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config, setup_seed
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.relation_augmentation import RelationAugmenter
import numpy as np
import random

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


PRED_STR = 'roi_heads.relation.predictor'


def train(cfg, local_rank, distributed, logger):
    debug_print(logger, 'prepare training')
    arguments = {}
    arguments["iteration"] = 0
    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=distributed,
    )
    debug_print(logger, 'end dataloader')
    if cfg.SOLVER.AUGMENTATION.USE_GRAFT is False:
        statistics = train_data_loader.dataset.get_statistics()
        vg_stats = VGStats(
            statistics['fg_matrix'],
            statistics['pred_dist'],
            statistics['obj_classes'],
            statistics['rel_classes'],
            statistics['att_classes'],
            statistics['stats'], # None
        )
    model = build_detection_model(cfg)
    debug_print(logger, 'end model construction')

    clean_classifier = cfg.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER
    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)

    if clean_classifier:
        fix_eval_modules_no_classifier(model, with_grad_name='_clean')
    else:
        fix_eval_modules(eval_modules)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    predictor = cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR
    if predictor == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor",]
    else:
        slow_heads = []

    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor" : "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor" : "roi_heads.box.feature_extractor"}

    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping["roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(cfg, model, logger, num_batch, slow_heads=slow_heads, slow_ratio=10.0)
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, 
                                       update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)
        if arguments["iteration"] != 0:
            raise NotImplementedError(f'Because we moved up the dataloaders for the singleton statistics, we cannot continue training from a non-zero iteration')
    else:
        # load_mapping is only used when we init current model from detection model.
        checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
        # load base model
        if clean_classifier:
            debug_print(logger, 'end load checkpointer')
            
            if predictor == "TransformerPredictor":
                load_mapping_classifier = {
                    f"{PRED_STR}.rel_compress_clean": f"{PRED_STR}.rel_compress",
                    f"{PRED_STR}.ctx_compress_clean": f"{PRED_STR}.ctx_compress",
                    f"{PRED_STR}.freq_bias_clean": f"{PRED_STR}.freq_bias",
                }
            elif predictor == "VCTreePredictor":
                load_mapping_classifier = {
                    f"{PRED_STR}.ctx_compress_clean": f"{PRED_STR}.ctx_compress",
                    f"{PRED_STR}.freq_bias_clean": f"{PRED_STR}.freq_bias",
                    f"{PRED_STR}.post_cat_clean": f"{PRED_STR}.post_cat",
                }
            elif predictor == "MotifPredictor":
                load_mapping_classifier = {
                    f"{PRED_STR}.rel_compress_clean": f"{PRED_STR}.rel_compress",
                    f"{PRED_STR}.freq_bias_clean": f"{PRED_STR}.freq_bias",
                    f"{PRED_STR}.post_cat_clean": f"{PRED_STR}.post_cat",
                }
            #load_mapping_classifier = {}
            if cfg.MODEL.PRETRAINED_MODEL_CKPT != "" :
                debug_print(logger, 'load PRETRAINED_MODEL_CKPT!!!!')
                checkpointer.load(cfg.MODEL.PRETRAINED_MODEL_CKPT, update_schedule=False,
                                 with_optim=False, load_mapping=load_mapping_classifier)
    # debug_print(logger, 'load PRETRAINED_MODEL_CKPT!!!!')
    # checkpointer.load(cfg.MODEL.PRETRAINED_MODEL_CKPT, update_schedule=False,
    #                  with_optim=False)
    debug_print(logger, 'end load checkpointer')

    use_semantic = cfg.SOLVER.AUGMENTATION.USE_SEMANTIC
    if use_semantic:
        debug_print(logger, 'using RelationAugmenter')
        vg_stats = VGStats()
        fg_matrix = vg_stats.fg_matrix
        pred_counts = fg_matrix.sum((0,1))
        strategy = cfg.SOLVER.AUGMENTATION.STRATEGY
        bottom_k = cfg.SOLVER.AUGMENTATION.BOTTOM_K
        num2aug = cfg.SOLVER.AUGMENTATION.NUM2AUG
        max_batchsize_aug = cfg.SOLVER.AUGMENTATION.MAX_BATCHSIZE_AUG
        relation_augmenter = RelationAugmenter(pred_counts, bottom_k, strategy, num2aug, max_batchsize_aug, cfg=cfg) # TODO: read strategy from scripts
        debug_print(logger, 'end RelationAugmenter')
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    # caching loop variables
    val_period = cfg.SOLVER.VAL_PERIOD
    to_val = cfg.SOLVER.TO_VAL is True
    to_test = cfg.SOLVER.TO_TEST is True
    dataset_names_val = cfg.DATASETS.VAL
    dataset_names_test = cfg.DATASETS.TEST
    output_folders_val = [None] * len(dataset_names_val)
    output_folders_test = [None] * len(dataset_names_test)
    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, distributed, logger, dataset_names_val, output_folders_val)

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    logger.info("Number iteration: "+str(max_iter))
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    if to_test:
        logger.info('Started creating test dataloader')
        # Make test dataloader
        if cfg.OUTPUT_DIR:
            for idx, dataset_name in enumerate(dataset_names_test):
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
                mkdir(output_folder)
                output_folders_test[idx] = output_folder
        data_loaders_test = make_data_loader(cfg, mode='test', is_distributed=distributed)
        logger.info('Finished creating test dataloader')
    print_first_grad = True
    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
        time_after_data = time.time()
        data_time = time_after_data - end
        num_before = len(targets)
        if use_semantic:
            images, targets = relation_augmenter.augment(images, targets)
            # print(f'{iteration}: Augmentation: {num_before} => {len(targets)}')
            time_after_semantic = time.time()
            semantic_time = time_after_semantic - time_after_data

        iteration = iteration + 1
        arguments["iteration"] = iteration

        model.train()
        if clean_classifier:
            fix_eval_modules_no_classifier(model, with_grad_name='_clean')
        else :
            fix_eval_modules(eval_modules)

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad # print grad or not
        print_first_grad = False
        clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad], max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)

        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        if use_semantic:
            meters.update(time=batch_time, data=data_time, semantic=semantic_time)
        else:
            meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 200 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[-1]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        result_val = None # used for scheduler updating
        if iteration % val_period == 0:
            if to_val:
                logger.info(f"iteration={iteration}: Start validating")
                result_val = run_val(cfg, model, val_data_loaders, distributed, logger, dataset_names_val, output_folders_val)
                logger.info(f"iteration={iteration}: Validation Result: %.4f" % result_val)
            if to_test:
                logger.info(f"iteration={iteration}: Start testing")
                result_test = run_val(cfg, model, data_loaders_test, distributed, logger, dataset_names_test, output_folders_test)
                logger.info(f"iteration={iteration}: Test Result: %.4f" % result_test)

        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            scheduler.step(result_val, epoch=iteration)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                break
        else:
            scheduler.step()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    return model

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False

def fix_eval_modules_no_classifier(module, with_grad_name='_clean'):
    #for module in eval_modules:
    for name, param in module.named_parameters():
        if with_grad_name not in name:
            param.requires_grad = False
            # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False

def run_val(cfg, model, val_data_loaders, distributed, logger, dataset_names, output_folders):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )

    val_result = []
    for output_folder, dataset_name, val_data_loader in zip(output_folders, dataset_names, val_data_loaders):
        dataset_result = inference(
                            cfg,
                            model,
                            val_data_loader,
                            dataset_name=dataset_name,
                            iou_types=iou_types,
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=output_folder,
                            logger=logger,
                        )
        synchronize()
        val_result.append(dataset_result)
    # support for multi gpu distributed testing
    gathered_result = all_gather(torch.tensor(dataset_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result>=0]
    val_result = float(valid_result.mean())
    del gathered_result, valid_result
    return val_result


def main():
    setup_seed(int(os_environ['SEED']))
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    num_gpus = int(os_environ["WORLD_SIZE"])
    args.distributed = num_gpus > 1

    local_rank = int(os_environ['LOCAL_RANK'])
    if args.distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    train(cfg, local_rank, args.distributed, logger)


if __name__ == "__main__":
    main()
