import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np


class GetAUC(nn.Module):
    def __init__(self):
        super(GetAUC, self).__init__()

    def forward(self, all_label, all_out):
        auc = []
        for n in range(1, all_out.shape[1]):
            fpr, tpr, thresholds = metrics.roc_curve(all_label, all_out[:, n], pos_label=n)
            auc.append(metrics.auc(fpr, tpr))

        return auc



class VGG11BinaryClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.vgg11(pretrained=True)
        num_ftrs = 512
        self.model.classifier = nn.Linear(num_ftrs, 1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f_right = self.model.features[:-1](x[0].squeeze())
        f_left = self.model.features[:-1](x[1].squeeze())
        f_right, _ = torch.max(f_right, dim=0)
        f_left, _ = torch.max(f_left, dim=0)  # (512, H, W)
        f = self.avgpool((f_right - f_left).unsqueeze(0))
        f = f[:, :, 0, 0]
        out = self.model.classifier(f)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())
        self.log('train_loss', loss)
        return {'loss': loss, 'y_true': y, 'y_pred': torch.sigmoid(y_hat.squeeze()).unsqueeze(0)}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())
        self.log('val_loss', loss)
        return {'loss': loss, 'y_true': y, 'y_pred': torch.sigmoid(y_hat.squeeze()).unsqueeze(0)}

    def training_epoch_end(self, outputs):
        y_true = torch.cat([x['y_true'] for x in outputs]).cpu().numpy()
        y_pred = torch.cat([x['y_pred'] for x in outputs]).cpu().numpy()
        auc = roc_auc_score(y_true, y_pred)
        self.log('train_auc', auc)

    def validation_epoch_end(self, outputs):
        y_true = torch.cat([x['y_true'] for x in outputs]).cpu().numpy()
        y_pred = torch.cat([x['y_pred'] for x in outputs]).cpu().numpy()
        auc = roc_auc_score(y_true, y_pred)
        self.log('val_auc', auc)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=4)


def main():
    from data.dataloader import Image3DDataset

    root_train = "/media/ExtHDD01/Dataset/paired_images/womac4/full/"
    train_dataset = Image3DDataset([root_train + 'ap/', root_train + 'bp/'])
    root_test = "/media/ExtHDD01/Dataset/paired_images/womac4/full/"
    val_dataset = Image3DDataset([root_test + 'ap/', root_test + 'bp/'])

    model = VGG11BinaryClassifier()
    model.train_dataset = train_dataset
    model.val_dataset = val_dataset

    logger = TensorBoardLogger("lightning_logs", name="vgg11_binary_classifier")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_auc',
        dirpath='checkpoints',
        filename='vgg11-binary-classifier-{epoch:02d}-{val_auc:.2f}',
        save_top_k=3,
        mode='max',
    )

    trainer = pl.Trainer(
        max_epochs=10,
        gpus=1 if torch.cuda.is_available() else None,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()