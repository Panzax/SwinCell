import os
import pdb
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast
from swincell.utils.utils import AverageMeter, distributed_all_gather
from monai.data import decollate_batch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def train_epoch(model, loader, optimizer, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=False):
            logits = model(data)
            # print(logits.shape,target.shape)
            if args.use_flows:
                loss_func1 = loss_func[0]
                loss_func2 = loss_func[1]
                #  weight_factor* flow loss     +      cell probability loss, weight_factor set to 5 according to paper
                loss = 5*loss_func1(logits[:,1:], target[:,1:]) + loss_func2(logits[:,0], target[:,0])

            else:
                # print(logits.shape,target.shape)
                loss = loss_func(logits[:,0], target[:,0])

        loss.backward()
        optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    if args.save_temp_img:
        img_raw = data[0,0,:,:,:].detach().cpu().numpy()
        img_raw = (img_raw - np.min(img_raw)) / (np.max(img_raw) - np.min(img_raw))
        img_raw  =np.max(img_raw,axis=-1)*255
        img_raw = img_raw.astype(np.uint8)
        #dimenstion is (batch_size, channel, height, width) 
        img_gt= target[0,0,:,:,:].detach().cpu().numpy()
        # if normalize:
        # img_gt= (img_gt - np.min(img_gt)) / (np.max(img_gt) - np.min(img_gt))
        img_gt =np.max(img_gt,axis=-1)*255
        img_gt= img_gt.astype(np.uint8)

        img_pred = logits[0][0,:,:,:].detach().cpu().numpy()
        img_pred = (img_pred  - np.min(img_pred)) / (np.max(img_pred) - np.min(img_pred))
        img_pred  =np.max(img_pred,axis=-1)*255
        img_pred = img_pred.astype(np.uint8)
        img_list = [img_raw, img_gt, img_pred]
    else:
        img_list = None
    return run_loss.avg, img_list


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()   # cell probs
    # run_acc2 = AverageMeter()  # flows

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=False):
                logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits) # a list of length 4 (number of input channels =4)

            # cell probs channel
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor[0])) for val_pred_tensor in val_outputs_list]
            val_label_convert = [val_label_tensor[0] for val_label_tensor in val_labels_list]

            # print(len(val_label_convert),len(val_output_convert),val_label_convert[0].shape,val_output_convert[0].shape)
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_label_convert)
            #validate with the binary masks 
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                Dice_Class = run_acc.avg[0]

                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    ", Cell dice class1:",
                    Dice_Class,

                    ", time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
        if args.save_temp_img:
            img_raw = data[0,0,:,:,:].detach().cpu().numpy()
            # print(img_raw.shape)
            img_raw = (img_raw - np.min(img_raw)) / (np.max(img_raw) - np.min(img_raw))
            img_raw  =np.max(img_raw,axis=-1)*255
            img_raw = img_raw.astype(np.uint8)   
            img_gt= target[0,0,:,:,:].detach().cpu().numpy()
            # print(img_gt.shape)
            img_gt =np.max(img_gt,axis=-1)*255
            img_gt= img_gt.astype(np.uint8)
            # print(val_output_convert[0].shape)
            # img_pred = val_output_convert[0][0,:,:,:].detach().cpu().numpy()
            img_pred = val_output_convert[0][:,:,:].detach().cpu().numpy() #modified for flows

            img_pred = (img_pred  - np.min(img_pred)) / (np.max(img_pred) - np.min(img_pred))
            img_pred  =np.max(img_pred,axis=-1)*255
            img_pred = img_pred.astype(np.uint8)
            img_list = [img_raw, img_gt, img_pred]
        else:
            img_list = None

    return run_acc.avg, img_list


def init_wandb(args):
    """
    Initialize Weights & Biases logging.
    
    Args:
        args: Training arguments containing wandb configuration
        
    Returns:
        wandb run object or None if Wandb is disabled/unavailable
    """
    if not WANDB_AVAILABLE:
        if args.rank == 0:
            print("Wandb not available. Install with: pip install wandb")
        return None
    
    # Check if Wandb should be enabled
    if args.wandb_project is None or args.wandb_mode == "disabled":
        return None
    
    # Only initialize on rank 0
    if args.rank != 0:
        return None
    
    # Build config dict from args
    config = {
        "model": args.model,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "sw_batch_size": args.sw_batch_size,
        "optim_lr": args.optim_lr,
        "optim_name": args.optim_name,
        "reg_weight": args.reg_weight,
        "momentum": args.momentum,
        "max_epochs": args.max_epochs,
        "val_every": args.val_every,
        "roi_x": args.roi_x,
        "roi_y": args.roi_y,
        "roi_z": args.roi_z,
        "feature_size": args.feature_size,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "a_min": args.a_min,
        "a_max": args.a_max,
        "b_min": args.b_min,
        "b_max": args.b_max,
        "downsample_factor": args.downsample_factor,
        "use_flows": args.use_flows,
        "use_checkpoint": args.use_checkpoint,
        "lrschedule": args.lrschedule,
        "warmup_epochs": args.warmup_epochs,
        "infer_overlap": args.infer_overlap,
        "distributed": args.distributed,
        "world_size": getattr(args, "world_size", 1),
        "workers": args.workers,
    }
    
    # Add Zarr-specific config if using Zarr metadata
    if hasattr(args, "use_zarr_metadata") and args.use_zarr_metadata:
        config["use_zarr_metadata"] = True
        config["use_synthetic"] = getattr(args, "use_synthetic", True)
        config["max_rois"] = getattr(args, "max_rois", None)
    
    # Generate run name if not provided
    run_name = args.wandb_run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.dataset}_{args.model}_{timestamp}"
        if hasattr(args, "use_zarr_metadata") and args.use_zarr_metadata:
            data_type = "synthetic" if getattr(args, "use_synthetic", True) else "real"
            run_name = f"{run_name}_{data_type}"
    
    # Initialize Wandb
    try:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=config,
            mode=args.wandb_mode,
            dir=args.logdir if hasattr(args, "logdir") else None,
        )
        print(f"Wandb initialized: project={args.wandb_project}, run={run_name}")
        return wandb_run
    except Exception as e:
        print(f"Failed to initialize Wandb: {e}")
        return None


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    # if args.logdir:
    if hasattr(args, "logdir"):
        filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    semantic_classes=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    
    # Initialize Wandb
    wandb_run = None
    if args.rank == 0:
        wandb_run = init_wandb(args)
    
    # scaler = None

    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss, train_img_list = train_epoch(
            model, train_loader, optimizer, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        
        # Log to Wandb
        if args.rank == 0 and wandb_run is not None:
            log_dict = {
                "train_loss": train_loss,
                "epoch": epoch,
            }
            # Log learning rate if scheduler is available
            if scheduler is not None:
                if hasattr(scheduler, "get_last_lr"):
                    lr = scheduler.get_last_lr()[0] if isinstance(scheduler.get_last_lr(), list) else scheduler.get_last_lr()
                elif hasattr(optimizer, "param_groups"):
                    lr = optimizer.param_groups[0]["lr"]
                else:
                    lr = None
                if lr is not None:
                    log_dict["learning_rate"] = lr
            
            # # Log training images if available
            # if args.save_temp_img and train_img_list is not None:
            #     try:
            #         import wandb
            #         log_dict["train_images/raw"] = wandb.Image(train_img_list[0])
            #         log_dict["train_images/ground_truth"] = wandb.Image(train_img_list[1])
            #         log_dict["train_images/prediction"] = wandb.Image(train_img_list[2])
            #     except Exception as e:
            #         print(f"Failed to log training images to Wandb: {e}")
            
            wandb_run.log(log_dict, step=epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc,val_img_list = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )

            if args.rank == 0:

                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    ", Dice_Class1:",
                    val_acc[0],
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )
                # print(epoch, val_acc)
                if writer is not None:
                    writer.add_scalar("Mean_Val_Dice", np.mean(val_acc), epoch)
                    if args.save_temp_img:
                        # for debug purpose, save intermediate results
                        writer.add_image("Validation/x1_raw", val_img_list[0], epoch, dataformats="HW")
                        writer.add_image("Validation/x1_gt", val_img_list[1], epoch, dataformats="HW")
                        writer.add_image("Validation/x1_prediction", val_img_list[2], epoch, dataformats="HW")

                        writer.add_image("Training/x1_raw", train_img_list[0], epoch, dataformats="HW")
                        writer.add_image("Training/x1_gt", train_img_list[1], epoch, dataformats="HW")
                        writer.add_image("Training/x1_prediction", train_img_list[2], epoch, dataformats="HW")
                    if semantic_classes is not None:
                        for val_channel_ind in range(len(semantic_classes)):
                            if val_channel_ind < val_acc.size:
                                writer.add_scalar(semantic_classes[val_channel_ind], val_acc[val_channel_ind], epoch)
                
                # Log validation metrics to Wandb
                if wandb_run is not None:
                    val_log_dict = {
                        "val/mean_dice": np.mean(val_acc),
                        "val/dice_class1": val_acc[0],
                        "epoch": epoch,
                    }
                    
                    # Log per-class dice if semantic_classes are available
                    if semantic_classes is not None:
                        for val_channel_ind in range(len(semantic_classes)):
                            if val_channel_ind < val_acc.size:
                                val_log_dict[f"val/dice_{semantic_classes[val_channel_ind]}"] = val_acc[val_channel_ind]
                    
                    # # Log validation images if available
                    # if args.save_temp_img and val_img_list is not None:
                    #     try:
                    #         import wandb
                    #         val_log_dict["val_images/raw"] = wandb.Image(val_img_list[0])
                    #         val_log_dict["val_images/ground_truth"] = wandb.Image(val_img_list[1])
                    #         val_log_dict["val_images/prediction"] = wandb.Image(val_img_list[2])
                    #     except Exception as e:
                    #         print(f"Failed to log validation images to Wandb: {e}")
                    
                    wandb_run.log(val_log_dict, step=epoch)
                val_avg_acc = np.mean(val_acc)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    
                    # Log best accuracy to Wandb
                    if wandb_run is not None:
                        wandb_run.log({"val/best_dice": val_acc_max, "epoch": epoch}, step=epoch)
                    
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("new best model, saving model.pt")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))
                    
                    # Log best checkpoint as Wandb artifact (optional)
                    if wandb_run is not None and args.save_checkpoint:
                        try:
                            import wandb
                            artifact = wandb.Artifact(
                                name=f"best_model_{wandb_run.id}",
                                type="model",
                                description=f"Best model checkpoint at epoch {epoch} with dice {val_acc_max:.6f}",
                            )
                            artifact.add_file(os.path.join(args.logdir, "model.pt"))
                            wandb_run.log_artifact(artifact)
                        except Exception as e:
                            print(f"Failed to log checkpoint artifact to Wandb: {e}")

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)
    
    # Finish Wandb run
    if wandb_run is not None:
        wandb_run.summary["best_val_dice"] = val_acc_max
        wandb_run.finish()
        print("Wandb run finished")

    return val_acc_max
