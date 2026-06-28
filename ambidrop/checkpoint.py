import os
import torch


def load_checkpoint(checkpoint_path, target_epoch=None, net=None, optimizer=None, scheduler=None):
    """
    Load checkpoint for a specific epoch or the closest available epoch.

    Returns a dict with:
        - epoch: the loaded epoch number
        - loss: the validation loss at that epoch (if stored)
        - learning_rate: the learning rate at that epoch (if stored)
    """
    checkpoint_list = torch.load(checkpoint_path, map_location="cpu")
    available_epochs = [ckpt["epoch"] for ckpt in checkpoint_list]

    if target_epoch is None:
        target_epoch = max(available_epochs)
        print(f"No epoch specified. Loading the latest checkpoint from epoch {target_epoch}")

    if target_epoch in available_epochs:
        chosen_epoch = target_epoch
    else:
        chosen_epoch = min(available_epochs, key=lambda e: abs(e - target_epoch))
        print(f"Epoch {target_epoch} not found. Using closest epoch: {chosen_epoch}")

    checkpoint_to_load = next(ckpt for ckpt in checkpoint_list if ckpt["epoch"] == chosen_epoch)

    if net is not None:
        net.load_state_dict(checkpoint_to_load['model_state_dict'], strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_to_load['optimizer_state_dict'])
        if 'learning_rate' in checkpoint_to_load:
            for param_group in optimizer.param_groups:
                param_group['lr'] = checkpoint_to_load['learning_rate']

    if scheduler is not None and 'scheduler_state_dict' in checkpoint_to_load:
        scheduler.load_state_dict(checkpoint_to_load['scheduler_state_dict'])

    result = {
        'epoch': chosen_epoch,
        'loss': checkpoint_to_load.get('loss', None),
        'learning_rate': checkpoint_to_load.get('learning_rate', None),
    }

    print(f"Loaded checkpoint from epoch {chosen_epoch}" +
          (f", loss: {result['loss']:.4f}" if result['loss'] is not None else ""))

    return result


def save_checkpoint(checkpoint_path, epoch, net, optimizer, loss, lr=None):
    """
    Append a checkpoint to the checkpoint list file.
    Creates a new file if it doesn't exist.
    """
    if os.path.exists(checkpoint_path):
        checkpoint_list = torch.load(checkpoint_path, map_location="cpu")
    else:
        checkpoint_list = []

    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'learning_rate': lr if lr is not None else optimizer.param_groups[0]['lr'],
    }
    checkpoint_list.append(checkpoint_data)
    torch.save(checkpoint_list, checkpoint_path)
