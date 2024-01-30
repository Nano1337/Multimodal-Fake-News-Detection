import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sentence_transformers import SentenceTransformer

from transformers import DistilBertForSequenceClassification

NUM_CLASSES = 6
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DROPOUT_P = 0.1
MODALITY = "text-image"

DATA_PATH = "./data"
PL_ASSETS_PATH = "./lightning_logs"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
IMAGE_EXTENSION = ".jpg"
RESNET_OUT_DIM = 2048
SENTENCE_TRANSFORMER_EMBEDDING_DIM = 768

losses = []


print("CUDA available:", torch.cuda.is_available())

class JointTextImageModel(nn.Module):

    def __init__(
            self,
            num_classes,
            loss_fn,
            text_module,
            image_module,
            text_feature_dim,
            image_feature_dim,
            fusion_output_size,
            dropout_p,
            hidden_size=512,
        ):
        super(JointTextImageModel, self).__init__()
        self.text_module = text_module # output is text_feature_dim
        self.image_module = image_module # output is image_feature_dim
        self.fc1_image = torch.nn.Linear(in_features=image_feature_dim, out_features=hidden_size)
        self.fc2_image = torch.nn.Linear(in_features=hidden_size, out_features=num_classes)

        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

        self.image_model = nn.Sequential(
            self.image_module, 
            nn.ReLU(), 
            self.fc1_image, 
            nn.ReLU(), 
            self.dropout, 
            self.fc2_image,
        )

    def forward(self, text, image, label):
        image_logits = self.image_model(image)
        text_logits = self.text_module(**text).logits

        # nn.CrossEntropyLoss expects raw logits as model output, NOT torch.nn.functional.softmax(logits, dim=1)
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        image_loss = self.loss_fn(image_logits, label)
        text_loss = self.loss_fn(text_logits, label)

        return (image_logits, text_logits, image_loss, text_loss)

    @classmethod
    def build_image_transform(cls, image_dim=224):
        image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(image_dim, image_dim)),
            torchvision.transforms.ToTensor(),
            # All torchvision models expect the same normalization mean and std
            # https://pytorch.org/docs/stable/torchvision/models.html
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

        return image_transform

class MultimodalFakeNewsDetectionModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super(MultimodalFakeNewsDetectionModel, self).__init__()
        if hparams:
            # Cannot reassign self.hparams in pl.LightningModule; must use update()
            # https://github.com/PyTorchLightning/pytorch-lightning/discussions/7525
            self.hparams.update(hparams)

        self.embedding_dim = self.hparams.get("embedding_dim", 768)
        self.text_feature_dim = self.hparams.get("text_feature_dim", 300)
        self.image_feature_dim = self.hparams.get("image_feature_dim", self.text_feature_dim)

        self.model = self._build_model()

        self.val_metrics = {
            "val_loss": [],
            "val_acc": [],
        }

        self.test_metrics = {
            "test_loss": [],
            "test_acc": [],
        }

    # Required for pl.LightningModule
    def forward(self, text, image, label):
        # pl.Lightning convention: forward() defines prediction for inference

        return self.model(text, image, label)

    # Required for pl.LightningModule
    def training_step(self, batch, batch_idx):

        # Extract text, image, and label from the batch
        text, image, label = batch["text"], batch["image"], batch["label"]

        # Get predictions and loss from the model
        image_logits, text_logits, image_loss, text_loss = self.model(text, image, label)

        # Calculate accuracy
        image_acc = torch.mean((torch.argmax(image_logits, dim=1) == label).float())
        text_acc = torch.mean((torch.argmax(text_logits, dim=1) == label).float())
        avg_logits = (image_logits + text_logits) / 2
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = (image_loss + text_loss) / 2

        # Log loss and accuracy
        self.log("train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # log modality-specific avg and losses
        self.log("image_train_loss", image_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("image_train_acc", image_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("text_train_loss", text_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("text_train_acc", text_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # Return the loss
        return avg_loss

    def validation_step(self, batch, batch_idx): 

        # Extract text, image, and label from the batch
        text, image, label = batch["text"], batch["image"], batch["label"]

        # Get predictions and loss from the model
        image_logits, text_logits, image_loss, text_loss = self.model(text, image, label)

        # Calculate accuracy
        image_acc = torch.mean((torch.argmax(image_logits, dim=1) == label).float())
        text_acc = torch.mean((torch.argmax(text_logits, dim=1) == label).float())
        avg_logits = (image_logits + text_logits) / 2
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = (image_loss + text_loss) / 2

        # Log loss and accuracy
        self.log("val_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # log modality-specific avg and losses
        self.log("image_val_loss", image_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("image_val_acc", image_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("text_val_loss", text_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("text_val_acc", text_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_loss"].append(avg_loss)
        self.val_metrics["val_acc"].append(joint_acc)

        return avg_loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_metrics["val_loss"]).mean()
        avg_accuracy = torch.stack(self.val_metrics["val_acc"]).mean()

        self.log("avg_val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("avg_val_acc", avg_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_loss"].clear()
        self.val_metrics["val_acc"].clear()

    # Optional for pl.LightningModule
    def test_step(self, batch, batch_idx):
        # Extract text, image, and label from the batch
        text, image, label = batch["text"], batch["image"], batch["label"]

        # Get predictions and loss from the model
        image_logits, text_logits, image_loss, text_loss = self.model(text, image, label)

        # Calculate accuracy
        image_acc = torch.mean((torch.argmax(image_logits, dim=1) == label).float())
        text_acc = torch.mean((torch.argmax(text_logits, dim=1) == label).float())
        avg_logits = (image_logits + text_logits) / 2
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = (image_loss + text_loss) / 2

        self.test_metrics["test_loss"].append(avg_loss)
        self.test_metrics["test_acc"].append(joint_acc)
        
        self.log("test_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        self.log("image_test_loss", image_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("image_test_acc", image_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("text_test_loss", text_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("text_test_acc", text_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return avg_loss

    # Optional for pl.LightningModule
    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_metrics["test_loss"]).mean()
        avg_accuracy = torch.stack(self.test_metrics["test_acc"]).mean()

        self.log("avg_test_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("avg_test_acc", avg_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_loss"].clear()
        self.test_metrics["test_acc"].clear()

    # Required for pl.LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=0.9)
        return optimizer

    def _build_model(self):
        text_module = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=NUM_CLASSES)
        image_module = torchvision.models.resnet152(pretrained=True)
        # Overwrite last layer to get features (rather than classification)
        image_module.fc = torch.nn.Linear(
            in_features=RESNET_OUT_DIM, out_features=self.image_feature_dim)

        return JointTextImageModel(
            num_classes=self.hparams.get("num_classes", NUM_CLASSES),
            loss_fn=torch.nn.CrossEntropyLoss(),
            text_module=text_module,
            image_module=image_module,
            text_feature_dim=self.text_feature_dim,
            image_feature_dim=self.image_feature_dim,
            fusion_output_size=self.hparams.get("fusion_output_size", 512),
            dropout_p=self.hparams.get("dropout_p", DROPOUT_P)
        )

class JointTextImageDialogueModel(nn.Module):

    def __init__(
            self,
            num_classes,
            loss_fn,
            text_module,
            image_module,
            dialogue_module,
            text_feature_dim,
            image_feature_dim,
            dialogue_feature_dim,
            fusion_output_size,
            dropout_p,
            hidden_size=512,
        ):
        super(JointTextImageDialogueModel, self).__init__()
        self.text_module = text_module
        self.image_module = image_module
        self.dialogue_module = dialogue_module
        self.fusion = torch.nn.Linear(in_features=(text_feature_dim + image_feature_dim + dialogue_feature_dim),
            out_features=fusion_output_size)
        # self.fc = torch.nn.Linear(in_features=fusion_output_size, out_features=num_classes)
        self.fc1 = torch.nn.Linear(in_features=fusion_output_size, out_features=hidden_size) # trial
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=num_classes) # trial
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, text, image, dialogue, label):
        text_features = torch.nn.functional.relu(self.text_module(text))
        image_features = torch.nn.functional.relu(self.image_module(image))
        dialogue_features = torch.nn.functional.relu(self.dialogue_module(dialogue))
        # print(text_features.size(), image_features.size()) # torch.Size([32, 300]) torch.Size([16, 300])
        combined = torch.cat([text_features, image_features, dialogue_features], dim=1)
        fused = self.dropout(
            torch.nn.functional.relu(self.fusion(combined)))
        # logits = self.fc(fused)
        hidden = torch.nn.functional.relu(self.fc1(fused)) # trial
        logits = self.fc2(hidden) # trial
        # pred = torch.nn.functional.softmax(logits, dim=1)
        pred = logits # nn.CrossEntropyLoss expects raw logits as model output # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss = self.loss_fn(pred, label)
        return (pred, loss)

    @classmethod
    def build_image_transform(cls, image_dim=224):
        image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(image_dim, image_dim)),
            torchvision.transforms.ToTensor(),
            # All torchvision models expect the same normalization mean and std
            # https://pytorch.org/docs/stable/torchvision/models.html
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

        return image_transform

class MultimodalFakeNewsDetectionModelWithDialogue(pl.LightningModule):

    def __init__(self, hparams=None):
        super(MultimodalFakeNewsDetectionModelWithDialogue, self).__init__()
        if hparams:
            self.hparams.update(hparams) # https://github.com/PyTorchLightning/pytorch-lightning/discussions/7525

        self.embedding_dim = self.hparams.get("embedding_dim", 768)
        self.text_feature_dim = self.hparams.get("text_feature_dim", 300)
        self.image_feature_dim = self.hparams.get("image_feature_dim", self.text_feature_dim)
        self.dialogue_feature_dim = self.hparams.get("dialogue_feature_dim", self.text_feature_dim)

        self.model = self._build_model()

    # Required for pl.LightningModule
    def forward(self, text, image, dialogue, label):
        # pl.Lightning convention: forward() defines prediction for inference
        return self.model(text, image, dialogue,  label)

    # Required for pl.LightningModule
    def training_step(self, batch, batch_idx):
        global losses
        # pl.Lightning convention: training_step() defines prediction and
        # accompanying loss for training, independent of forward()
        text, image, dialogue, label = batch["text"], batch["image"], batch["dialogue"], batch["label"]

        pred, loss = self.model(text, image, dialogue, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print(loss.item())
        losses.append(loss.item())
        return loss

    # Optional for pl.LightningModule
    def training_step_end(self, batch_parts):
        """
        Aggregates results when training using a strategy that splits data
        from each batch across GPUs (e.g. data parallel)

        Note that training_step returns a loss, thus batch_parts returns a list
        of 2 loss values (since there are 2 GPUs being used)
        """
        return sum(batch_parts) / len(batch_parts)

    # Optional for pl.LightningModule
    def test_step(self, batch, batch_idx):
        text, image, dialogue, label = batch["text"], batch["image"], batch["dialogue"], batch["label"]
        pred, loss = self.model(text, image, dialogue, label)
        pred_label = torch.argmax(pred, dim=1)
        accuracy = torch.sum(pred_label == label).item() / (len(label) * 1.0)
        output = {
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy).cuda()
        }
        print(loss.item(), output['test_acc'])
        return output

    # Optional for pl.LightningModule
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["test_acc"] for x in outputs]).mean()
        logs = {
            'test_loss': avg_loss,
            'test_acc': avg_accuracy
        }

        # pl.LightningModule has some issues displaying the results automatically
        # As a workaround, we can store the result logs as an attribute of the
        # class instance and display them manually at the end of testing
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1088
        self.test_results = logs

        return {
            'avg_test_loss': avg_loss,
            'avg_test_acc': avg_accuracy,
            'log': logs,
            'progress_bar': logs
        }

    # Required for pl.LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=0.9)
        return optimizer

    def _build_model(self):
        text_module = torch.nn.Linear(
            in_features=self.embedding_dim, out_features=self.text_feature_dim)

        image_module = torchvision.models.resnet152(pretrained=True)
        # Overwrite last layer to get features (rather than classification)
        image_module.fc = torch.nn.Linear(
            in_features=RESNET_OUT_DIM, out_features=self.image_feature_dim)

        dialogue_module = torch.nn.Linear(
            in_features=self.embedding_dim, out_features=self.dialogue_feature_dim)

        return JointTextImageDialogueModel(
            num_classes=self.hparams.get("num_classes", NUM_CLASSES),
            loss_fn=torch.nn.CrossEntropyLoss(),
            text_module=text_module,
            image_module=image_module,
            dialogue_module=dialogue_module,
            text_feature_dim=self.text_feature_dim,
            image_feature_dim=self.image_feature_dim,
            dialogue_feature_dim=self.dialogue_feature_dim,
            fusion_output_size=self.hparams.get("fusion_output_size", 512),
            dropout_p=self.hparams.get("dropout_p", DROPOUT_P)
        )

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started...")

    def on_train_end(self, trainer, pl_module):
        print("Training done...")
        global losses
        for loss_val in losses:
            print(loss_val)
