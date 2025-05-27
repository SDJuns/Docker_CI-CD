import torch
import mlflow
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
from .config import DEVICE, SAVE_MODEL_PATH, SAVE_HISTORY_PATH
from .dataset import CustomOrderImageFolder, CustomSubset
from ..model.model import get_model

criterion = torch.nn.CrossEntropyLoss()

def Train(model, train_DL, val_DL, criterion, optimizer, EPOCH, **kwargs):
    scheduler = StepLR(optimizer, step_size=kwargs['LR_STEP'], gamma=kwargs['LR_GAMMA']) if 'LR_STEP' in kwargs else None
    loss_history, acc_history = {'train':[], 'val':[]}, {'train':[], 'val':[]}

    best_loss = float('inf')

    for epoch in range(EPOCH):
        model.train()
        train_loss, train_acc, _ = loss_epoch(model, train_DL, criterion, optimizer)

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, _ = loss_epoch(model, val_DL, criterion)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({'model': model.state_dict()}, SAVE_MODEL_PATH)

        if scheduler:
            scheduler.step()

        loss_history['train'].append(train_loss)
        loss_history['val'].append(val_loss)
        acc_history['train'].append(train_acc)
        acc_history['val'].append(val_acc)

        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

    torch.save({"loss_history": loss_history, "acc_history": acc_history}, SAVE_HISTORY_PATH)
    return loss_history, acc_history

def loss_epoch(model, DL, criterion, optimizer=None):
    epoch_loss, total_correct = 0, 0
    for x_batch, y_batch in tqdm(DL, leave=False):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        y_hat = model(x_batch)
        loss = criterion(y_hat, y_batch)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * x_batch.shape[0]
        total_correct += (y_hat.argmax(dim=1) == y_batch).sum().item()

    epoch_loss /= len(DL.dataset)
    epoch_acc = total_correct / len(DL.dataset) * 100
    return epoch_loss, epoch_acc, total_correct
