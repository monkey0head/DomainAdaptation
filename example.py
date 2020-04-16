import torch
import os

from trainer import Trainer
from loss import loss_DANN
from models import DANNModel
from data_loader import create_data_generators
from metrics import AccuracyScoreFromLogits
from utils.callbacks import simple_callback, ModelSaver, HistorySaver
import configs.dann_config as dann_config

os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def only_loss(*args, **kwargs):
    loss, rich_loss = loss_DANN(device=device, *args, **kwargs)
    # loss_string = '   '.join(['val_src_{}: {:.5f}\t'.format(k, float(v)) for k, v in rich_loss.items()])
    # print(f"step_loss: {loss_string}")
    return loss


class DebugMetric:
    score = 1
    name = 'test'

    def __init__(self, metric):
        self.metric = metric

    def reset(self):
        self.metric.reset()

    def __call__(self, *args, **kwargs):
        pass
        print(f"Call metric with args:\n{args}\n{kwargs}")
        print("metric: ", self.metric(*args, **kwargs))


if __name__ == '__main__':
    train_gen_s, val_gen_s, test_gen_s = create_data_generators("office-31",
                                                                'amazon',
                                                                batch_size=64,
                                                                infinite_train=True,
                                                                image_size=224,
                                                                device=device,
                                                                num_workers=2)

    train_gen_t, val_gen_t, test_gen_t = create_data_generators("office-31",
                                                                'dslr',
                                                                batch_size=64,
                                                                infinite_train=True,
                                                                image_size=224,
                                                                device=device,
                                                                num_workers=2)
    model = DANNModel().to(device)
    acc = AccuracyScoreFromLogits()
    mmm = DebugMetric(acc)
    val_freq = 3

    tr = Trainer(model, only_loss)
    tr.fit(train_gen_s, train_gen_t,
           n_epochs=51,
           validation_data=[val_gen_s, val_gen_t],
           metrics=[acc],
           steps_per_epoch=200,
           val_freq=val_freq,
           callbacks=[simple_callback, ModelSaver('DANN', val_freq), HistorySaver('test_log', val_freq)])
