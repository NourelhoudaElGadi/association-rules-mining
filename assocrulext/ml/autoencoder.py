import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


class AutoEncoderDimensionalityReduction(
    pl.LightningModule, BaseEstimator, TransformerMixin
):
    def __init__(
        self,
        input_dim=256,
        encoding_dim=2,
        batch_size=32,
        learning_rate=0.001,
        patience=5,
        min_delta=0.0,
        epochs=100,
        novelty_score=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.novelty_score = novelty_score

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

        self.early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, min_delta=min_delta
        )

    def _compute_loss(self, batch, batch_idx):
        x = batch[0]
        outputs = self(x)
        if self.novelty_score is not None:
            weights = torch.tensor(
                self.novelty_score[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ],
                dtype=torch.float32,
            ).to(x.device)
            loss = nn.MSELoss(reduction="none")(outputs, x)
            loss = (loss * weights).mean()
        else:
            loss = nn.MSELoss()(outputs, x)
        return loss

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def fit(self, X, y=None):
        X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32))

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        from pytorch_lightning.callbacks import EarlyStopping

        trainer = pl.Trainer(
            callbacks=[EarlyStopping(monitor="val_loss", mode="min")], max_epochs=100
        )
        trainer.fit(self, train_dataloader, val_dataloader)

        return self

    def transform(self, X, y=None):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(torch.tensor(X, dtype=torch.float32)).numpy()
