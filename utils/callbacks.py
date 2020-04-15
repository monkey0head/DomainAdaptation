import os

def simple_callback(model, epoch_log, current_epoch, total_epoch):
    train_loss = epoch_log['loss']
    val_loss = epoch_log['val_loss']
    trg_metrics = epoch_log['trg_metrics']
    src_metrics = epoch_log['src_metrics']
    message_head = f'Epoch {current_epoch+1}/{total_epoch}\n'
    message_loss = 'loss: {:<10}\t val_loss: {:<10}\t'.format(train_loss, val_loss)
    message_src_metrics = ' '.join(['val_src_{}: {:<10}\t'.format(k, v) for k, v in src_metrics.items()])
    message_trg_metrics = ' '.join(['val_trg_{}: {:<10}\t'.format(k, v) for k, v in trg_metrics.items()])
    print(message_head + message_loss + message_src_metrics + message_trg_metrics)


class ModelSaver:
    def __init__(self, model_type, path="checkpoints"):
        self.model_type = model_type
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, model_type)):
            os.makedirs(os.path.join(path, model_type))

    def __call__(self, model, epoch_log, current_epoch, total_epoch):
        import torch
        filename = os.path.join(self.path, self.model_type, "epoch_{}.pt".format(current_epoch))
        torch.save(model.state_dict(), filename)


class Logger:
    def __init__(self, log_name, path="_log"):
        import json
        from collections import defaultdict

        self.json = json
        self.log_name = log_name
        self.path = path
        self.loss_history = defaultdict(list)

        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, log_name)):
            os.makedirs(os.path.join(path, log_name))

    def __call__(self, model, epoch_log, current_epoch, total_epoch):
        filename = os.path.join(self.path, self.log_name, "epoch_{}.pt".format(current_epoch))
        for item in epoch_log:
            self.loss_history[item].append(epoch_log[item])
        with open(filename, 'w') as f:
            self.json.dump(self.loss_history, f)
