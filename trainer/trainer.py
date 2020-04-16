import torch
import tqdm

class Trainer:
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss
        self.epoch = 0
        self.last_epoch_history = None

    def _reset_last_epoch_history(self):
        self.last_epoch_history = {
            'loss': 0.0,
            'val_loss': 0.0,
            'src_metrics': {},
            'trg_metrics': {}
        }

    def calc_loss(self, src_batch, trg_batch):
        batch = self._merge_batches(src_batch, trg_batch)
        metadata = {'epoch': self.epoch, 'n_epochs': self.n_epochs}
        loss = self.loss(self.model, batch, **metadata)
        return loss

    def train_on_batch(self, src_batch, trg_batch, opt):
        self.model.train()
        loss = self.calc_loss(src_batch, trg_batch)
        self.last_epoch_history['loss'] += loss.data.cpu().item()

        opt.zero_grad()
        loss.backward()
        opt.step()

    def _merge_batches(self, src_batch, trg_batch):
        src_images, src_classes = src_batch
        trg_images, trg_classes = trg_batch
        batch = dict()
        batch['src_images'] = src_images
        batch['trg_images'] = trg_images
        batch['src_classes'] = src_classes
        batch['trg_classes'] = trg_classes
        return batch

    def fit(self, src_data, trg_data, n_epochs=1000, steps_per_epoch=100, val_freq=1,
            opt='adam', opt_kwargs=None, validation_data=None, metrics=None, callbacks=None):

        self.n_epochs = n_epochs

        if opt_kwargs is None:
            opt_kwargs = dict()

        if opt == 'adam':
            opt = torch.optim.Adam(self.model.parameters(), **opt_kwargs)
        else:
            raise NotImplementedError

        if validation_data is not None:
            src_val_data, trg_val_data = validation_data

        for self.epoch in tqdm.trange(self.epoch, n_epochs):
            self._reset_last_epoch_history()
            for step, (src_batch, trg_batch) in tqdm.tqdm(enumerate(zip(src_data, trg_data)), total=steps_per_epoch):
                if step == steps_per_epoch:
                    break
                self.train_on_batch(src_batch, trg_batch, opt)

            self.last_epoch_history['loss'] /= steps_per_epoch

            # validation
            if self.epoch % val_freq == 0 and validation_data is not None:
                self.model.eval()

                # calculating metrics on validation
                if metrics is not None:
                    if src_val_data is not None:
                        src_metrics = self.score(src_val_data, metrics)
                        self.last_epoch_history['src_metrics'] = src_metrics
                    if trg_val_data is not None:
                        trg_metrics = self.score(trg_val_data, metrics)
                        self.last_epoch_history['trg_metrics'] = trg_metrics

                # calculating loss on validation
                if src_val_data is not None and trg_val_data is not None:
                    actual_val_steps = 0
                    for val_step, (src_batch, trg_batch) in enumerate(zip(src_val_data, trg_val_data)):
                        if val_step == steps_per_epoch:
                            break
                        actual_val_steps += 1
                        loss = self.calc_loss(src_batch, trg_batch).detach().cpu().item()
                        self.last_epoch_history['val_loss'] += loss
                    if actual_val_steps > 0:
                        self.last_epoch_history['val_loss'] /= actual_val_steps
                    else:
                        print('not enough validation data, validation loss was not calculated')

            if callbacks is not None:
                for callback in callbacks:
                    callback(self.model, self.last_epoch_history, self.epoch, n_epochs)

    def score(self, data, metrics):
        for metric in metrics:
            metric.reset()

        data.reload_iterator()
        for images, true_classes in data:
            pred_classes = self.model.predict(images)
            for metric in metrics:
                metric(true_classes, pred_classes)
        data.reload_iterator()
        return {metric.name: metric.score for metric in metrics}

    def predict(self, data):
        predictions = []
        for batch in data:
            predictions.append(self.model.predict(batch))
        return torch.cat(predictions)
