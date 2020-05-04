import os
import torch
import argparse
import numpy

from models import DANNModelFeatures
from dataloader import create_data_generators_my
from metrics import AccuracyScoreFromLogits
import configs.dann_config as dann_config

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_classes_features(model, data):
    predictions = []
    classes = []
    for images, true_classes in data:
        predictions.append(model.get_features(images).data.cpu())
        classes.append(true_classes)
    return torch.cat(predictions).data.cpu().numpy(), torch.cat(classes).data.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    args = parser.parse_args()

    gen_t, _, _ = create_data_generators_my(dann_config.DATASET,
                                               dann_config.TARGET_DOMAIN,
                                               batch_size=dann_config.BATCH_SIZE,
                                               infinite_train=False,
                                               split_ratios=[1, 0, 0],
                                               image_size=dann_config.IMAGE_SIZE,
                                               num_workers=dann_config.NUM_WORKERS,
                                               device=device)

    model = DANNModelFeatures().to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    acc = AccuracyScoreFromLogits()
    features, classes = get_classes_features(model, gen_t)
    classes = classes.astype('int').reshape(-1, 1)
    print(features.shape, classes.shape)
    numpy.savetxt('./embeddings_resnet/Df.txt', features, delimiter=',', fmt='%.7f')
    numpy.savetxt('./embeddings_resnet/Dl.txt', classes, newline=',', fmt='%d')
