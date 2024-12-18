import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from core.utils.colors import get_rgb_colors
from core.engine.validation import do_validation
from torch.utils.tensorboard import SummaryWriter
from core.utils.tensorboard import update_tensorboard_image_samples
from core.engine.metrics import Dice, JaccardIndex, Precision, Recall
from core.data.transforms.transforms import (
    FromTensor,
    Denormalize,
    ConvertToInts,
    TransformCompose
)


def calc_mean_metric(metric: np.ndarray, ignore_indexes: list = []):
    """
    Calculates mean of given metric.

    Parameters:
        metric : array
            Metric with shape (K), where K is number of classes.
        ignore_indexes : list
            List of class indexes to ignore for calculating.

    Returns:
        mean : float
            Resulting mean metric with shape (1)
    """
    idxs_to_mean = [idx for idx in list(range(metric.shape[0])) if idx not in ignore_indexes]

    return np.mean(metric[idxs_to_mean])


def update_summary_writer(cfg, summary_writer, stats, iterations, optimizer, global_step, class_labels, is_train: bool = True):
    domen = "train" if is_train else "val"
    with torch.no_grad():
        # lr
        if is_train:
            summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        # loss
        summary_writer.add_scalar(domen + '/loss', stats['loss_sum'] / iterations, global_step=global_step)

        # metrics mean for classes
        ignore_idxs = cfg.TENSORBOARD.METRICS_IGNORE_CLASS_IDXS
        summary_writer.add_scalar(domen + '/dice_mean', calc_mean_metric(stats['dice_sum'] / iterations, ignore_idxs), global_step=global_step)
        summary_writer.add_scalar(domen + '/jaccard_mean', calc_mean_metric(stats['jaccard_sum'] / iterations, ignore_idxs), global_step=global_step)
        summary_writer.add_scalar(domen + '/precision_mean', calc_mean_metric(stats['precision_sum'] / iterations, ignore_idxs), global_step=global_step)
        summary_writer.add_scalar(domen + '/recall_mean', calc_mean_metric(stats['recall_sum'] / iterations, ignore_idxs), global_step=global_step)

        # metrics for each class
        dice_dict = {}
        jaccard_dict = {}
        precision_dict = {}
        recall_dict = {}
        for idx, label in enumerate(class_labels):
            dice_dict[f"dice_{label}"] = stats['dice_sum'][idx] / iterations
            jaccard_dict[f"jaccard_{label}"] = stats['jaccard_sum'][idx] / iterations
            precision_dict[f"precision_{label}"] = stats['precision_sum'][idx] / iterations
            recall_dict[f"recall_{label}"] = stats['recall_sum'][idx] / iterations
        summary_writer.add_scalars(domen + '/dice', dice_dict, global_step=global_step)
        summary_writer.add_scalars(domen + '/jaccard', jaccard_dict, global_step=global_step)
        summary_writer.add_scalars(domen + '/precision', precision_dict, global_step=global_step)
        summary_writer.add_scalars(domen + '/recall', recall_dict, global_step=global_step)

        # images
        back_transforms = [
            FromTensor(),
            Denormalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_SCALE),
            ConvertToInts()
        ]
        back_transforms = TransformCompose(back_transforms)

        if len(stats['best_samples']):
            tb_images = [s[1] for s in stats['best_samples']]
            image_grid = torch.concatenate(tb_images, dim=1)
            image_grid = back_transforms(image_grid)[0]
            summary_writer.add_image(domen + '/best_samples', image_grid, global_step=global_step, dataformats='HWC')

        if len(stats['worst_samples']):
            tb_images = [s[1] for s in stats['worst_samples']]
            image_grid = torch.concatenate(tb_images, dim=1)
            image_grid = back_transforms(image_grid)[0]
            summary_writer.add_image(domen + '/worst_samples', image_grid, global_step=global_step, dataformats='HWC')

        # save
        summary_writer.flush()


def do_train(cfg,
             model,
             data_loader_train,
             data_loader_val,
             optimizer,
             scheduler,
             checkpointer,
             device,
             arguments,
             args):
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    logger = logging.getLogger("CORE")
    logger.info("Start training ...")

    # set model to train mode
    model.train()

    # create tensorboard writer
    if args.use_tensorboard:
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    # prepare to train
    iters_per_epoch = len(data_loader_train)
    total_steps = iters_per_epoch * cfg.SOLVER.MAX_EPOCH
    start_epoch = arguments["epoch"]
    end_epoch = cfg.SOLVER.MAX_EPOCH
    logger.info("Iterations per epoch: {0}. Total steps: {1}. Start epoch: {2}".format(iters_per_epoch, total_steps, start_epoch))

    # create metrics
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    num_classes = len(cfg.DATASET.CLASS_LABELS)
    dice_metric = Dice(num_classes)
    jaccard_metric = JaccardIndex(num_classes)
    precision_metric = Precision(num_classes)
    recall_metric = Recall(num_classes)

    # epoch loop
    for epoch in range(start_epoch, end_epoch):
        arguments["epoch"] = epoch + 1

        # create progress bar
        print(('\n' + '%10s' * 4 + '%10s' * 4) % ('epoch', 'gpu_mem', 'lr', 'loss', 'dice', 'jaccard', 'prec', 'recall'))
        pbar = enumerate(data_loader_train)
        pbar = tqdm(pbar, total=len(data_loader_train))

        # create stats
        stats = {
            'loss_sum': 0,
            'dice_sum': 0,
            'jaccard_sum': 0,
            'precision_sum': 0,
            'recall_sum': 0,
            'best_samples': [],
            'worst_samples': []
        }

        # iteration loop
        for iteration, data_entry in pbar:
            global_step = epoch * iters_per_epoch + iteration

            # get data
            input, target_labels, roi = data_entry
            input = input.to(device)                               # (n, c, h, w)
            target_labels = target_labels.squeeze(1).to(device)    # (n, h, w)
            roi = roi.squeeze(1).to(device)                        # (n, h, w)

            # forward model
            output = model(input) # (n, k, h, w)

            # calculate loss
            loss = cross_entropy.forward(output, target_labels)     # (n, h, w)
            loss = loss * roi                                       # (n, h, w)

            # optimize model
            optimizer.zero_grad()
            loss_mean = torch.mean(loss)
            loss_mean.backward()
            optimizer.step()

            # calculate metrics
            pred_labels = torch.softmax(output, dim=1).argmax(dim=1)        # (n, h, w)
            dice = dice_metric(pred_labels, target_labels, roi)             # (n, k)
            jaccard = jaccard_metric(pred_labels, target_labels, roi)       # (n, k)
            precision = precision_metric(pred_labels, target_labels, roi)   # (n, k)
            recall = recall_metric(pred_labels, target_labels, roi)         # (n, k)

            # update stats
            stats['loss_sum'] += loss_mean.item()
            stats['dice_sum'] += torch.mean(dice, 0).cpu().numpy()
            stats['jaccard_sum'] += torch.mean(jaccard, 0).cpu().numpy()
            stats['precision_sum'] += torch.mean(precision, 0).cpu().numpy()
            stats['recall_sum'] += torch.mean(recall, 0).cpu().numpy()

            # update best samples
            update_tensorboard_image_samples(limit=cfg.TENSORBOARD.BEST_SAMPLES_NUM,
                                             accumulator=stats['best_samples'], 
                                             input=input,
                                             metric=torch.mean(loss, dim=(1, 2)), 
                                             pred_labels=pred_labels, 
                                             target_labels=target_labels,
                                             min_metric_better=True,
                                             class_colors=get_rgb_colors(num_classes, mean=cfg.INPUT.PIXEL_MEAN, scale=cfg.INPUT.PIXEL_SCALE),
                                             blending_alpha=cfg.TENSORBOARD.ALPHA_BLENDING,
                                             nonzero_factor=cfg.TENSORBOARD.NONZERO_CLASS_PERC)

            # update worst samples
            update_tensorboard_image_samples(limit=cfg.TENSORBOARD.WORST_SAMPLES_NUM,
                                             accumulator=stats['worst_samples'], 
                                             input=input,
                                             metric=torch.mean(loss, dim=(1, 2)), 
                                             pred_labels=pred_labels, 
                                             target_labels=target_labels,
                                             min_metric_better=False,
                                             class_colors=get_rgb_colors(num_classes, mean=cfg.INPUT.PIXEL_MEAN, scale=cfg.INPUT.PIXEL_SCALE),
                                             blending_alpha=cfg.TENSORBOARD.ALPHA_BLENDING,
                                             nonzero_factor=cfg.TENSORBOARD.NONZERO_CLASS_PERC)

            # update progress bar
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)

            dice_avg = stats['dice_sum'] / (iteration + 1)
            jaccard_avg = stats['jaccard_sum'] / (iteration + 1)
            precision_avg = stats['precision_sum'] / (iteration + 1)
            recall_avg = stats['recall_sum'] / (iteration + 1)

            s = ('%10s' * 2 + '%10.4g' * 2 + '%10.4g' * 4) % ('%g/%g' % (epoch + 1, end_epoch),
                                               mem,
                                               optimizer.param_groups[0]["lr"],
                                               stats['loss_sum'] / (iteration + 1),
                                               calc_mean_metric(dice_avg, cfg.TENSORBOARD.METRICS_IGNORE_CLASS_IDXS),
                                               calc_mean_metric(jaccard_avg, cfg.TENSORBOARD.METRICS_IGNORE_CLASS_IDXS),
                                               calc_mean_metric(precision_avg, cfg.TENSORBOARD.METRICS_IGNORE_CLASS_IDXS),
                                               calc_mean_metric(recall_avg, cfg.TENSORBOARD.METRICS_IGNORE_CLASS_IDXS))
            pbar.set_description(s)

        # update learning rate
        if scheduler is not None:
            scheduler.step()

        # do validation
        if (args.val_step > 0) and (epoch % args.val_step == 0) and (data_loader_val is not None):
            print('\n')
            logger.info("Start validation ...")

            torch.cuda.empty_cache()
            model.eval()
            val_stats = do_validation(cfg, model, data_loader_val, device)
            torch.cuda.empty_cache()
            model.train()

            val_loss = val_stats['loss_sum'] / val_stats['iterations']
            val_dice_avg = val_stats['dice_sum'] / val_stats['iterations']
            val_jaccard_avg = val_stats['jaccard_sum'] / val_stats['iterations']
            val_precision_avg = val_stats['precision_sum'] / val_stats['iterations']
            val_recall_avg = val_stats['recall_sum'] / val_stats['iterations']

            log_preamb = 'Validation results: '
            print((log_preamb + '%10s' * 1 + '%10s' * 4) % ('loss', 'dice', 'jaccard', 'prec', 'recall'))
            print((len(log_preamb) * ' ' + '%10.4g' * 1 + '%10.4g' * 4) % (val_loss,
                                                                           calc_mean_metric(val_dice_avg, cfg.TENSORBOARD.METRICS_IGNORE_CLASS_IDXS),
                                                                           calc_mean_metric(val_jaccard_avg, cfg.TENSORBOARD.METRICS_IGNORE_CLASS_IDXS),
                                                                           calc_mean_metric(val_precision_avg, cfg.TENSORBOARD.METRICS_IGNORE_CLASS_IDXS),
                                                                           calc_mean_metric(val_recall_avg, cfg.TENSORBOARD.METRICS_IGNORE_CLASS_IDXS)))
            print('\n')

            if summary_writer:
                update_summary_writer(cfg,
                                      summary_writer,
                                      val_stats,
                                      val_stats['iterations'],
                                      optimizer,
                                      global_step,
                                      cfg.DATASET.CLASS_LABELS,
                                      False)

        # save epoch results
        if epoch % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(global_step), **arguments)
            if summary_writer:
                update_summary_writer(cfg,
                                      summary_writer,
                                      stats,
                                      iteration + 1,
                                      optimizer,
                                      global_step,
                                      cfg.DATASET.CLASS_LABELS,
                                      True)

    # save final model
    checkpointer.save("model_final", **arguments)

    return model
