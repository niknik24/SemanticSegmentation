from collections import defaultdict
import torch.nn.functional as F
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb


def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask_cl = np.copy(mask)
    vehicle = np.array(list(map(np.array, [[70.0,  0.0, 0.0], [100.0, 60.0, 0.0], [90.0, 0.0, 0.0], [110.0, 0.0, 0.0], [100.0, 80.0, 0.0],
     [230.0, 0.0, 0.0], [32.0, 11.0, 119.0], [142.0, 0.0, 0.0], [64, 64, 128], [60.0, 20.0, 220.0], [0.0,  0.0,  255.0], [100, 0, 255], [200, 0, 255]])))

    mask[(mask not in vehicle)] = np.array([0., 0., 0.])

    mask[(mask_cl[:, :, 0] == vehicle[0][0]) & (mask_cl[:, :, 1] == vehicle[0][1]) & (mask_cl[:, :, 2] == vehicle[0][2])] = np.array([1.,  0.,  0.])
    mask[(mask_cl[:, :, 0] == vehicle[1][0]) & (mask_cl[:, :, 1] == vehicle[1][1]) & (mask_cl[:, :, 2] == vehicle[1][2])] = np.array([1.,  0.,  0.])
    mask[(mask_cl[:, :, 0] == vehicle[2][0]) & (mask_cl[:, :, 1] == vehicle[2][1]) & (mask_cl[:, :, 2] == vehicle[2][2])] = np.array([1.,  0.,  0.])
    mask[(mask_cl[:, :, 0] == vehicle[3][0]) & (mask_cl[:, :, 1] == vehicle[3][1]) & (mask_cl[:, :, 2] == vehicle[3][2])] = np.array([1.,  0.,  0.])
    mask[(mask_cl[:, :, 0] == vehicle[4][0]) & (mask_cl[:, :, 1] == vehicle[4][1]) & (mask_cl[:, :, 2] == vehicle[4][2])] = np.array([1.,  0.,  0.])
    mask[(mask_cl[:, :, 0] == vehicle[5][0]) & (mask_cl[:, :, 1] == vehicle[5][1]) & (mask_cl[:, :, 2] == vehicle[5][2])] = np.array([1.,  0.,  0.])
    mask[(mask_cl[:, :, 0] == vehicle[6][0]) & (mask_cl[:, :, 1] == vehicle[6][1]) & (mask_cl[:, :, 2] == vehicle[6][2])] = np.array([1.,  0.,  0.])
    mask[(mask_cl[:, :, 0] == vehicle[7][0]) & (mask_cl[:, :, 1] == vehicle[7][1]) & (mask_cl[:, :, 2] == vehicle[7][2])] = np.array([1., 0.,  0.])
    mask[(mask_cl[:, :, 0] == vehicle[8][0]) & (mask_cl[:, :, 1] == vehicle[8][1]) & (mask_cl[:, :, 2] == vehicle[8][2])] = np.array([1.,  0.,  0.])
    mask[(mask_cl[:, :, 0] == vehicle[9][0]) & (mask_cl[:, :, 1] == vehicle[9][1]) & (mask_cl[:, :, 2] == vehicle[9][2])] = np.array([1., 0., 0.])
    mask[(mask_cl[:, :, 0] == vehicle[10][0]) & (mask_cl[:, :, 1] == vehicle[10][1]) & (mask_cl[:, :, 2] == vehicle[10][2])] = np.array([1., 0., 0.])
    mask[(mask_cl[:, :, 0] == vehicle[11][0]) & (mask_cl[:, :, 1] == vehicle[11][1]) & (mask_cl[:, :, 2] == vehicle[11][2])] = np.array([1., 0., 0.])
    mask[(mask_cl[:, :, 0] == vehicle[12][0]) & (mask_cl[:, :, 1] == vehicle[12][1]) & (mask_cl[:, :, 2] == vehicle[12][2])] = np.array([1., 0., 0.])

    mask = np.asarray(mask) # (256, 256, 3)
    mask = mask[:, :, 0]
    return mask


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


def train(train_loader, model, criterion, optimizer, epoch, params, grad_scaler):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True).float()
        target = target.to(params["device"], non_blocking=True).float()
        output = model(images).squeeze(1)
        # output = output.reshape((output.shape[0], output.shape[2], output.shape[3], output.shape[1]))
        loss = criterion(output, target)
        metric_monitor.update("Loss", loss.item())
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        # loss.backward()
        # optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )
    return loss


def validate(val_loader, model, criterion, epoch, params, experiment, optimizer, scheduler):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True).float()
            target = target.to(params["device"], non_blocking=True).float()
            output = model(images).squeeze(1)
            histograms = {}
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
            loss = criterion(output, target)
            scheduler.step(loss)
            metric_monitor.update("Loss", loss.item())

            logging.info('Validation Dice score: {}'.format(loss))
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation Dice': loss,
                'images': wandb.Image(images[0].cpu()),
                'masks': {
                    'true': wandb.Image(target[0].float().cpu()),
                    'pred': wandb.Image(torch.softmax(output, dim=1)[0].float().cpu()),
                },
                'epoch': epoch,
                **histograms
            })

            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )
    return loss


def train_and_validate(model, train_dataset, val_dataset, params):
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
    )

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=params["epochs"], batch_size=params["batch_size"], learning_rate=params["lr"],
                                  val_percent=len(val_dataset), amp=params["amp"]))

    logging.info(f'''Starting training:
        Epochs:          {params["epochs"]}
        Batch size:      {params["batch_size"]}
        Learning rate:   {params["lr"]}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Device:          {params["device"]}
        Mixed Precision: {params["amp"]}
    ''')

    train_criterion = DiceBCELoss().to(params["device"])
    val_criterion = DiceBCELoss().to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    grad_scaler = torch.cuda.amp.GradScaler(enabled=params["amp"])
    best_val_loss = 10
    for epoch in range(1, params["epochs"] + 1):
        tr_loss = train(train_loader, model, train_criterion, optimizer, epoch, params, grad_scaler)

        val_loss = validate(val_loader, model, val_criterion, epoch, params, experiment, optimizer, scheduler)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), "saved_models/best_value_epoch_" + str(epoch) + ".pth")
            try:
                os.remove("saved_models/best_value_epoch_" + str(best_epoch) + ".pth")
                best_epoch = epoch
            except:
                print("not checkpoint")
                best_epoch = epoch
            best_val_loss = val_loss
    return model


def predict(model, params, test_dataset, batch_size):
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=params["num_workers"], pin_memory=True,
    )
    model.eval()

    predictions = []
    with torch.no_grad():
        for images, (original_heights, original_widths) in test_loader:
            images = images.to(params["device"], non_blocking=True)
            output = model(images)
            probabilities = torch.sigmoid(output.squeeze(1))
            predicted_masks = (probabilities >= 0.1).float() * 1
            predicted_masks = predicted_masks.cpu().numpy()
            for predicted_mask, original_height, original_width in zip(
                    predicted_masks, original_heights.numpy(), original_widths.numpy()
            ):
                predictions.append((predicted_mask, original_height, original_width))

    return predictions

