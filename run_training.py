import os
import argparse

# pylint: disable=import-error
from tqdm import tqdm # pylint: disable=unused-import
import yaml

import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything 

from sentence_transformers import SentenceTransformer

from dataloader import MultimodalDataset, Modality
# for the default implementation, use model.py | model_ensemble.py | model_joint.py | model_joint_proba
from model_joint_proba import *
torch.set_float32_matmul_precision('medium')

seed = 1
seed_everything(seed=seed, workers=True)


# Multiprocessing for dataset batching
# Set to 0 and comment out torch.multiprocessing line if multiprocessing gives errors
NUM_CPUS = 8

DATA_PATH = "./data"
IMAGES_DIR = os.path.join(DATA_PATH, "public_image_set")
TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 1000
SENTENCE_TRANSFORMER_EMBEDDING_DIM = 768
DEFAULT_GPUS = [0]

if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn')

    # pylint: disable=line-too-long
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Running on training data")
    parser.add_argument("--test", action="store_true", help="Running on test (evaluation) data")
    parser.add_argument("--config", type=str, default="", help="config.yaml file with experiment configuration")

    # We default all hyperparameters to None so that their default values can
    # be taken from a config file; if the config file is not specified, then we
    # use the given default values in the `config.get()` calls (see below)
    # Thus the order of precedence for hyperparameter values is
    #   passed manually as an arg -> specified in given config file -> default
    # This allows experiments defined in config files to be easily replicated
    # while tuning specific parameters via command-line args
    parser.add_argument("--modality", type=str, default=None, help="text | image | text-image | text-image-dialogue")
    parser.add_argument("--num_classes", type=int, default=None, help="2 | 3 | 6")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--dropout_p", type=float, default=None)
    parser.add_argument("--gpus", type=str, help="Comma-separated list of ints with no spaces; e.g. \"0\" or \"0,1\"")
    parser.add_argument("--text_embedder", type=str, default=None, help="all-mpnet-base-v2 | all-distilroberta-v1")
    parser.add_argument("--dialogue_summarization_model", type=str, default=None, help="None=Transformers.Pipeline default i.e. sshleifer/distilbart-cnn-12-6 | bart-large-cnn | t5-small | t5-base | t5-large")
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--preprocessed_train_dataframe_path", type=str, default=None)
    parser.add_argument("--preprocessed_val_dataframe_path", type=str, default=None)
    parser.add_argument("--preprocessed_test_dataframe_path", type=str, default=None)
    args = parser.parse_args()

    # Load configuration from YAML file if specified
    if args.config:
        with open(args.config, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
    else:
        config = {}

    # Update args with config values if not already set by command line arguments
    for key, default_value in [
        ("modality", "text-image"),
        ("num_classes", 2),
        ("batch_size", 32),
        ("learning_rate", 1e-4),
        ("num_epochs", 10),
        ("dropout_p", 0.1),
        ("text_embedder", "all-mpnet-base-v2"),
        ("dialogue_summarization_model", "bart-large-cnn"),
        ("train_data_path", os.path.join(DATA_PATH, f"multimodal_train_{TRAIN_DATA_SIZE}.tsv")),
        ("val_data_path", os.path.join(DATA_PATH, f"multimodal_validate.tsv")),
        ("test_data_path", os.path.join(DATA_PATH, f"multimodal_test_{TEST_DATA_SIZE}.tsv")),
        ("preprocessed_train_dataframe_path", None),
        ("preprocessed_val_dataframe_path", "val__text_image_dataframe.pkl"),
        ("preprocessed_test_dataframe_path", None)
    ]:
        if getattr(args, key, None) is None:
            setattr(args, key, config.get(key, default_value))

    # Special handling for 'gpus' because it needs to be converted from string to list[int]
    if args.gpus:
        args.gpus = [int(gpu_num) for gpu_num in args.gpus.split(",")]
    else:
        args.gpus = config.get("gpus", DEFAULT_GPUS)

    text_embedder = SentenceTransformer(args.text_embedder)
    image_transform = None # pylint: disable=invalid-name
    if Modality(args.modality) == Modality.TEXT_IMAGE_DIALOGUE:
        image_transform = JointTextImageDialogueModel.build_image_transform()
    else:
        image_transform = JointTextImageModel.build_image_transform()

    train_dataset = MultimodalDataset(
        from_preprocessed_dataframe=args.preprocessed_train_dataframe_path,
        data_path=args.train_data_path,
        modality=args.modality,
        text_embedder=text_embedder,
        image_transform=image_transform,
        summarization_model=args.dialogue_summarization_model,
        images_dir=IMAGES_DIR,
        num_classes=args.num_classes
    )
    print("Train dataset size: {}".format(len(train_dataset)))

    val_dataset = MultimodalDataset(
        from_preprocessed_dataframe=args.preprocessed_val_dataframe_path,
        data_path=args.val_data_path,
        modality=args.modality,
        text_embedder=text_embedder,
        image_transform=image_transform,
        summarization_model=args.dialogue_summarization_model,
        images_dir=IMAGES_DIR,
        num_classes=args.num_classes
    )
    print("Val dataset size: {}".format(len(val_dataset)))

    test_dataset = MultimodalDataset(
        from_preprocessed_dataframe=args.preprocessed_test_dataframe_path,
        data_path=args.test_data_path,
        modality=args.modality,
        text_embedder=text_embedder,
        image_transform=image_transform,
        summarization_model=args.dialogue_summarization_model,
        images_dir=IMAGES_DIR,
        num_classes=args.num_classes
    )
    print("Test dataset size: {}".format(len(test_dataset)))


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=NUM_CPUS, 
        persistent_workers=True,
        prefetch_factor = 4,
        collate_fn=train_dataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=NUM_CPUS, 
        persistent_workers=True,
        prefetch_factor = 4,
        collate_fn=val_dataset.collate_fn
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=NUM_CPUS, 
        persistent_workers=True,
        prefetch_factor = 4,
        collate_fn=test_dataset.collate_fn
    )

    hparams = {
        "embedding_dim": SENTENCE_TRANSFORMER_EMBEDDING_DIM,
        "num_classes": args.num_classes
    }

    model = None
    if Modality(args.modality) == Modality.TEXT_IMAGE_DIALOGUE:
        model = MultimodalFakeNewsDetectionModelWithDialogue(hparams)
    else:
        model = MultimodalFakeNewsDetectionModel(hparams)

    trainer = None
    callbacks = [PrintCallback()]
    wandb_logger = WandbLogger(log_model="all")
    if torch.cuda.is_available():
        # call pytorch lightning trainer 
        trainer = pl.Trainer(
            strategy="auto",
            callbacks=callbacks,
            max_epochs=args.num_epochs, 
            logger = wandb_logger,
            deterministic=True, 
            default_root_dir="ckpts/",  
            precision="bf16-mixed",
            num_sanity_val_steps=0,
        )
    else:
        trainer = pl.Trainer(
            callbacks=callbacks
        )

    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,
    )

    trainer.test(
        model,
        dataloaders=test_loader
    )