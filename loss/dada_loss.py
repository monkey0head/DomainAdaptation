import numpy as np
import torch
import configs.dann_config as dann_config

# call loss_DANN instead of this function
def _loss_DADA_splitted(
        lambda_,
        class_logits_on_src,
        class_logits_on_trg,
        true_labels_on_src,
        true_labels_on_trg,
        unk_value=dann_config.UNK_VALUE,
        device=torch.device('cpu')):

    target_len = len(class_logits_on_trg)
    true_labels_on_src = torch.as_tensor(true_labels_on_src).long()
    if dann_config.IS_UNSUPERVISED:
        true_labels_on_trg = unk_value * torch.ones(target_len, dtype=torch.long, device=device)
    else:
        true_labels_on_trg = torch.as_tensor(true_labels_on_trg).long()

    # classification loss source
    probs_all_src = torch.nn.Softmax(-1)(class_logits_on_src)
    # print(probs_all_src[:1])
    loss_source = - torch.mean(
        (torch.ones(len(class_logits_on_src), dtype=torch.long, device=device) - probs_all_src[:, -1]) *\
        torch.log(probs_all_src[torch.arange(probs_all_src.size(0)), true_labels_on_src]
                  ) +
                  probs_all_src[:, -1] *
                  torch.log(
                      (torch.ones(len(class_logits_on_src), dtype=torch.long, device=device) -
                       probs_all_src[torch.arange(probs_all_src.size(0)), true_labels_on_src.flatten()]))
                               )

    crossentropy = torch.nn.CrossEntropyLoss(ignore_index=unk_value, reduction='mean')
    # loss_source = - torch.mean(torch.log(probs_all_src[torch.arange(probs_all_src.size(0)), true_labels_on_src]))
    # print(probs_all_src[torch.arange(probs_all_src.size(0)), true_labels_on_src.flatten()].shape)
    # print('probs_all_src', probs_all_src[torch.arange(probs_all_src.size(0)), true_labels_on_src.flatten()])
    # loss_source = crossentropy(class_logits_on_src, true_labels_on_src)

    # print('loss_source', loss_source)
    # loss target
    probs_all_trg = torch.nn.Softmax(-1)(class_logits_on_trg)
    probs_real_trg = torch.div(probs_all_trg, torch.ones((len(probs_all_trg), 1), dtype=torch.long, device=device) - probs_all_trg[:, -1].reshape(-1,1))
    probs_real_trg[:, -1] = 0

    probs_trg_hat = (probs_all_trg / (probs_all_trg + probs_all_trg[:, -1].reshape(-1, 1)))[:, :-1]
    # print('probs_trg_hat.shape', probs_trg_hat.shape)
    # print('probs_trg_hat', probs_trg_hat)

    loss_trg_classifier = - torch.sum(torch.log(probs_trg_hat) * probs_real_trg[:, :-1], dim=-1).mean()
    # print('loss_trg_classifier', loss_trg_classifier)
    loss_trg_generator = torch.sum(probs_real_trg[:, :-1] * torch.log(torch.ones_like(probs_trg_hat) - probs_trg_hat), dim=-1).mean()
    # print('loss_trg_generator', loss_trg_generator)

    entropy_loss_on_trg = torch.sum(torch.log(probs_real_trg + 10e-6) * probs_real_trg, dim=-1).mean()

    # print('entropy_loss_on_trg', entropy_loss_on_trg)
    # loss_trg_classifier = torch.zeros([1], dtype=torch.long, device=device)
    # loss_trg_generator = torch.zeros([1], dtype=torch.long, device=device)
    # entropy_loss_on_trg = torch.zeros([1], dtype=torch.long, device=device)
    # print(lambda_)
    loss_min = lambda_ * (loss_source + loss_trg_classifier) + entropy_loss_on_trg
    loss_max = lambda_ * (loss_source + loss_trg_generator) - entropy_loss_on_trg

    return [loss_min, loss_max], {
            "classifier_loss_on_src": loss_source.data.cpu().item(),
            "loss_trg_classifier": loss_trg_classifier.data.cpu().item(),
            "loss_trg_generator": loss_trg_generator.data.cpu().item(),
            "entropy_loss_on_trg": - entropy_loss_on_trg.data.cpu().item(),
            "loss_min": loss_min.data.cpu().item(),
            "loss_max": loss_max.data.cpu().item(),
            "lambda": lambda_
    }


def calc_lambda(current_iteration,
                        total_iterations,
                        gamma=dann_config.LOSS_GAMMA):
    progress = current_iteration / total_iterations
    lambda_p = 2 / (1 + np.exp(-gamma * progress)) - 1
    return lambda_p


def loss_DADA(model,
              batch,
              epoch,
              n_epochs,
              device=torch.device('cpu')):
    """
    :param model: model.forward(images) should return dict with keys
        'class' : Tensor, shape = (batch_size, n_classes)  logits  of classes (raw, not logsoftmax)
        'domain': Tensor, shape = (batch_size,) logprob for domain
    :param batch: dict with keys
        'src_images':
        'trg_images':
        'src_classes':np.Array, shape = (batch_size,)
        'trg_classes':np.Array, shape = (batch_size,)
    if true_class is unknown, then class should be dann_config.UNK_VALUE
    :param epoch: current number of iteration
    :param n_epochs: total number of iterations
    :return:
        loss: torch.Tensor,
        losses dict:{
            "domain_loss_on_src"
            "domain_loss_on_trg"
            "domain_loss"
            "prediction_loss_on_src"
            "prediction_loss_on_trg"
            "prediction_loss"
        }
    """
    lambda_ = calc_lambda(epoch, n_epochs)
    
    model_output = model.forward(batch['src_images'].to(device))
    class_logits_on_src = model_output['class']

    model_output = model.forward(batch['trg_images'].to(device))
    class_logits_on_trg = model_output['class']

    return _loss_DADA_splitted(
        lambda_,
        class_logits_on_src,
        class_logits_on_trg,
        true_labels_on_src=batch['src_classes'],
        true_labels_on_trg=batch['trg_classes'],
        device=device)