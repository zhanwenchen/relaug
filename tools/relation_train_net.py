from argparse import ArgumentParser, REMAINDER
from os import environ as os_environ
from os.path import join as os_path_join
from time import time as time_time
from datetime import timedelta as datetime_timedelta
from torch import (
    device as torch_device,
    as_tensor as torch_as_tensor,
    cat as torch_cat,
)
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
from torch.cuda import max_memory_allocated, set_device
from wandb import init as wandb_init
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader, VGStats
from maskrcnn_benchmark.solver import make_lr_scheduler, make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer, clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, all_gather
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config, setup_seed
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.relation_augmentation import RelationAugmenter


PRED_STR = 'roi_heads.relation.predictor'


def train(config, local_rank, distributed, logger, run):
    is_main = local_rank == 0
    debug_print(logger, 'prepare training')
    arguments = {}
    arguments["iteration"] = 0
    train_data_loader = make_data_loader(
        config,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        config,
        mode='val',
        is_distributed=distributed,
    )
    debug_print(logger, 'end dataloader')
    if config.SOLVER.AUGMENTATION.USE_GRAFT is False:
        try:
            statistics = train_data_loader.dataset.get_statistics()
        except AttributeError:
            statistics = train_data_loader.dataset.datasets[0].get_statistics()
        vg_stats = VGStats(
            statistics['fg_matrix'],
            statistics['pred_dist'],
            statistics['obj_classes'],
            statistics['rel_classes'],
            statistics['att_classes'],
            statistics['stats'], # None
        )
    model = build_detection_model(config)
    debug_print(logger, 'end model construction')

    clean_classifier = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER
    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)

    if clean_classifier:
        fix_eval_modules_no_classifier(model, with_grad_name='_clean')
    else:
        fix_eval_modules(eval_modules)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    predictor = config.MODEL.ROI_RELATION_HEAD.PREDICTOR
    if predictor == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor",]
    else:
        slow_heads = []

    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor" : "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor" : "roi_heads.box.feature_extractor"}

    if config.MODEL.ATTRIBUTE_ON:
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping["roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    device = torch_device(config.MODEL.DEVICE)
    model.to(device, non_blocking=True)

    num_batch = config.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(config, model, logger, num_batch, slow_heads=slow_heads, slow_ratio=10.0)
    scheduler = make_lr_scheduler(config, optimizer, logger)
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')

    output_dir = config.OUTPUT_DIR

    save_to_disk = local_rank == 0
    checkpointer = DetectronCheckpointer(
        config, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(config.MODEL.PRETRAINED_DETECTOR_CKPT,
                                       update_schedule=config.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)
        if arguments['iteration'] != 0:
            raise NotImplementedError('Because we moved up the dataloaders for the singleton statistics, we cannot continue training from a non-zero iteration')
    else:
        # load_mapping is only used when we init current model from detection model.
        checkpointer.load(config.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
        # load base model
        if clean_classifier:
            debug_print(logger, 'end load checkpointer')

            if predictor == 'TransformerPredictor':
                load_mapping_classifier = {
                    f'{PRED_STR}.rel_compress_clean': f'{PRED_STR}.rel_compress',
                    f'{PRED_STR}.ctx_compress_clean': f'{PRED_STR}.ctx_compress',
                    f'{PRED_STR}.freq_bias_clean': f'{PRED_STR}.freq_bias',
                }
            elif predictor == 'VCTreePredictor':
                load_mapping_classifier = {
                    f'{PRED_STR}.ctx_compress_clean': f'{PRED_STR}.ctx_compress',
                    f'{PRED_STR}.freq_bias_clean': f'{PRED_STR}.freq_bias',
                    f'{PRED_STR}.post_cat_clean': f'{PRED_STR}.post_cat',
                }
            elif predictor == 'MotifPredictor':
                load_mapping_classifier = {
                    f'{PRED_STR}.rel_compress_clean': f'{PRED_STR}.rel_compress',
                    f'{PRED_STR}.freq_bias_clean': f'{PRED_STR}.freq_bias',
                    f'{PRED_STR}.post_cat_clean': f'{PRED_STR}.post_cat',
                }
            #load_mapping_classifier = {}
            if config.MODEL.PRETRAINED_MODEL_CKPT != '' :
                debug_print(logger, 'load PRETRAINED_MODEL_CKPT!!!!')
                checkpointer.load(config.MODEL.PRETRAINED_MODEL_CKPT, update_schedule=False,
                                 with_optim=False, load_mapping=load_mapping_classifier)
    # debug_print(logger, 'load PRETRAINED_MODEL_CKPT!!!!')
    # checkpointer.load(config.MODEL.PRETRAINED_MODEL_CKPT, update_schedule=False,
    #                  with_optim=False)
    debug_print(logger, 'end load checkpointer')

    use_semantic = config.SOLVER.AUGMENTATION.USE_SEMANTIC
    if use_semantic:
        debug_print(logger, 'using RelationAugmenter')
        vg_stats = VGStats()
        fg_matrix = vg_stats.fg_matrix
        pred_counts = fg_matrix.sum((0,1))
        strategy = config.SOLVER.AUGMENTATION.STRATEGY
        bottom_k = config.SOLVER.AUGMENTATION.BOTTOM_K
        num2aug = config.SOLVER.AUGMENTATION.NUM2AUG
        max_batchsize_aug = config.SOLVER.AUGMENTATION.MAX_BATCHSIZE_AUG
        relation_augmenter = RelationAugmenter(pred_counts, bottom_k, strategy, num2aug, max_batchsize_aug, cfg=config) # TODO: read strategy from scripts
        debug_print(logger, 'end RelationAugmenter')
    checkpoint_period = config.SOLVER.CHECKPOINT_PERIOD

    # caching loop variables
    val_period = config.SOLVER.VAL_PERIOD
    to_val = config.SOLVER.TO_VAL is True
    to_test = config.SOLVER.TO_TEST is True
    dataset_names_val = config.DATASETS.VAL
    dataset_names_test = config.DATASETS.TEST
    output_folders_val = [None] * len(dataset_names_val)
    output_folders_test = [None] * len(dataset_names_test)
    if config.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        run_val(config, model, val_data_loaders, distributed, logger, dataset_names_val, output_folders_val)

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    logger.info("Number iteration: "+str(max_iter))
    start_iter = arguments["iteration"]
    start_training_time = time_time()
    end = time_time()

    if to_test:
        logger.info('Started creating test dataloader')
        # Make test dataloader
        if config.OUTPUT_DIR:
            for idx, dataset_name in enumerate(dataset_names_test):
                output_folder = os_path_join(config.OUTPUT_DIR, "inference", dataset_name)
                mkdir(output_folder)
                output_folders_test[idx] = output_folder
        data_loaders_test = make_data_loader(config, mode='test', is_distributed=distributed)
        logger.info('Finished creating test dataloader')
    print_first_grad = True
    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        if any(len(target) < 1 for target in targets):
            logger.error(f'Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}' )
        time_after_data = time_time()
        data_time = time_after_data - end
        if use_semantic:
            images, targets = relation_augmenter.augment(images, targets)
            # print(f'{iteration}: Augmentation: {num_before} => {len(targets)}')
            time_after_semantic = time_time()
            semantic_time = time_after_semantic - time_after_data

        iteration += 1
        arguments["iteration"] = iteration

        model.train()
        if clean_classifier:
            fix_eval_modules_no_classifier(model, with_grad_name='_clean')
        else :
            fix_eval_modules(eval_modules)

        images = images.to(device, non_blocking=True)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad(set_to_none=True)
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        losses.backward()
        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        verbose = (iteration % config.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad # print grad or not
        print_first_grad = False
        clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad], max_norm=config.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)

        optimizer.step()

        batch_time = time_time() - end
        end = time_time()
        if use_semantic:
            meters.update(time=batch_time, data=data_time, semantic=semantic_time)
        else:
            meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime_timedelta(seconds=int(eta_seconds)))

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
                    memory=max_memory_allocated() / 1048576.0,
                )
            )

        if iteration % checkpoint_period == 0:
            fname = f'model_{iteration:07d}'
            if iteration == max_iter:
                fname += '_final'
            checkpointer.save(fname, **arguments)

        result_val = None # used for scheduler updating
        if iteration % val_period == 0:
            if to_val:
                logger.info('iteration=%s: Start validating', iteration)
                result_val = run_val(config, model, val_data_loaders, distributed, logger, dataset_names_val, output_folders_val)
                logger.info('iteration=%s: Validation Result: %.4f', iteration, result_val)
                if is_main:
                    run.log({'mR@50_val': result_val}, iteration)
            if to_test:
                logger.info('iteration=%s: Start testing', iteration)
                result_test = run_val(config, model, data_loaders_test, distributed, logger, dataset_names_test, output_folders_test)
                logger.info('iteration=%s: Test Result: %.4f', iteration, result_test)
                if is_main:
                    run.log({'mR@50_test': result_test}, iteration)

        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        if config.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            scheduler.step(result_val, epoch=iteration)
            if scheduler.stage_count >= config.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                break
        else:
            scheduler.step()

    total_training_time = time_time() - start_training_time
    total_time_str = str(datetime_timedelta(seconds=total_training_time))
    logger.info('Total training time: %s ({:.4f} s / it)', total_time_str, total_training_time / max_iter)


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


def run_val(config, model, val_data_loaders, distributed, logger, dataset_names, output_folders):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if config.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if config.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if config.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if config.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )

    val_result = []
    for output_folder, dataset_name, val_data_loader in zip(output_folders, dataset_names, val_data_loaders):
        dataset_result = inference(
                            config,
                            model,
                            val_data_loader,
                            dataset_name=dataset_name,
                            iou_types=iou_types,
                            box_only=False if config.MODEL.RETINANET_ON else config.MODEL.RPN_ONLY,
                            device=config.MODEL.DEVICE,
                            expected_results=config.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=config.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=output_folder,
                            logger=logger,
                        )
        synchronize()
        val_result.append(dataset_result)
    # support for multi gpu distributed testing
    gathered_result = all_gather(torch_as_tensor(dataset_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch_cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result>=0]
    val_result = float(valid_result.mean())
    del gathered_result, valid_result
    return val_result


def main():
    setup_seed(int(os_environ['SEED']))
    parser = ArgumentParser(description="PyTorch Relation Detection Training")
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
        nargs=REMAINDER,
    )

    args = parser.parse_args()
    num_gpus = int(os_environ["WORLD_SIZE"])
    distributed = num_gpus > 1

    local_rank = int(os_environ['LOCAL_RANK'])
    if distributed:
        set_device(local_rank)
        init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    config_file = args.config_file

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, local_rank)
    logger.info('Using %s GPUs', num_gpus)
    logger.info(args)

    logger.info("Collecting env info (might take some time)\n")
    logger.info(collect_env_info())

    logger.info('Loaded configuration file %s', config_file)
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info('Running with config:\n%s', cfg)

    output_config_path = os_path_join(output_dir, 'config.yml')
    logger.info('Saving config into: %s', output_config_path)

    if local_rank == 0:
        run = wandb_init(
            # set the wandb project where this run will be logged
            project='relaug',
            # track hyperparameters and run metadata
            config={
                "learning_rate": cfg.SOLVER.BASE_LR,
                "architecture": cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR,
                "epochs": cfg.SOLVER.MAX_ITER,
                "batch_size": cfg.SOLVER.IMS_PER_BATCH,
            }
        )
    else:
        run = None
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    train(cfg, local_rank, distributed, logger, run)

    if local_rank == 0:
        run.finish()


if __name__ == "__main__":
    main()
