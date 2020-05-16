import torch
import os
import wandb

from trainer import Trainer
from loss import loss_DANN, class_prediction_loss
from models import DANNModel, OneDomainModel
    # DANNModelFeatures
from dataloader import create_data_generators
from metrics import AccuracyScoreFromLogits
from utils.callbacks import simple_callback, print_callback, ModelSaver, HistorySaver, WandbCallback
from utils.schedulers import LRSchedulerSGD
import configs.dann_config as dann_config

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # for source_domain in ['webcam', 'amazon', 'dslr']:
    #     dann_config.SOURCE_DOMAIN = source_domain
    #     for target_domain in ['webcam', 'amazon', 'dslr']:

    # for source_domain in ['amazon']:
    #     dann_config.SOURCE_DOMAIN = source_domain
    #     for target_domain in ['webcam']:
    #
    #         dann_config.TARGET_DOMAIN = target_domain
    #         if source_domain == target_domain:
    #             continue

    source_domain = dann_config.SOURCE_DOMAIN
    target_domain = dann_config.TARGET_DOMAIN
    print('source_domain is {}, target_domain is {}'.format(source_domain, target_domain))

    train_gen_s, val_gen_s, _ = create_data_generators(dann_config.DATASET,
                                                                dann_config.SOURCE_DOMAIN,
                                                                batch_size=dann_config.BATCH_SIZE,
                                                                infinite_train=True,
                                                                image_size=dann_config.IMAGE_SIZE,
                                                                split_ratios=[0.9, 0.1, 0],
                                                                num_workers=dann_config.NUM_WORKERS,
                                                                device=device)

    train_gen_t, _, _ = create_data_generators(dann_config.DATASET,
                                                                dann_config.TARGET_DOMAIN,
                                                                batch_size=dann_config.BATCH_SIZE,
                                                                infinite_train=True,
                                                                split_ratios=[1, 0, 0],
                                                                image_size=dann_config.IMAGE_SIZE,
                                                                num_workers=dann_config.NUM_WORKERS,
                                                                device=device)

    val_gen_t, _, _ = create_data_generators(dann_config.DATASET,
                                                                dann_config.TARGET_DOMAIN,
                                                                batch_size=dann_config.BATCH_SIZE,
                                                                infinite_train=False,
                                                                split_ratios=[1, 0, 0],
                                                                image_size=dann_config.IMAGE_SIZE,
                                                                num_workers=dann_config.NUM_WORKERS,
                                                                device=device)

    model = DANNModel().to(device)
    acc = AccuracyScoreFromLogits()

    scheduler = LRSchedulerSGD()
    tr = Trainer(model, loss_DANN)

    experiment_name = 'DANN_R_w_a_141_256'
    details_name = 'd_l_0_5'

    # experiment_name = 'no_labels'
    # details_name = ''
    print(model)
    tr.fit(train_gen_s, train_gen_t,
           n_epochs=dann_config.N_EPOCHS,
           validation_data=[val_gen_s, val_gen_t],
           metrics=[acc],
           steps_per_epoch=dann_config.STEPS_PER_EPOCH,
           val_freq=dann_config.VAL_FREQ,
           opt='sgd',
           opt_kwargs={'lr': 0.01, 'momentum': 0.9},
           lr_scheduler=scheduler,
           callbacks=[print_callback(watch=["loss", "domain_loss", "val_loss",
                                            "val_domain_loss", 'trg_metrics', 'src_metrics']),
                        ModelSaver(str(experiment_name + '_' + source_domain + '_' + target_domain + '_' + details_name),
                                   dann_config.SAVE_MODEL_FREQ, save_by_schedule=True, save_best=True, eval_metric='accuracy'),
                        WandbCallback(config=dann_config,
                                      name=str(source_domain + "_" + target_domain + "_" + details_name),
                                      group=experiment_name),
                        HistorySaver(str(experiment_name + '_' + source_domain + '_' + target_domain + "_" + details_name),
                                     dann_config.VAL_FREQ, path=str('_log/0504_' + experiment_name + "_" + details_name),
                                     # extra_losses={'domain_loss': ['domain_loss', 'val_domain_loss'],
                                     #               'train_domain_loss': ['domain_loss_on_src',
                                     #                                     'domain_loss_on_trg']}
                                     )
    ])
    wandb.join()