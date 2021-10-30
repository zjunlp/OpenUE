"""Experiment-running framework."""
import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
import openue.lit_models as lit_models
import yaml
import time
from openue.lit_models import MyTrainer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In order to ensure reproducible experiments, we must set random seeds.


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
	
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--litmodel_class", type=str, default="SEQLitModel")
    parser.add_argument("--data_class", type=str, default="REDataset")
    parser.add_argument("--model_class", type=str, default="bert.BertForSequenceClassification")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"openue.data.{temp_args.data_class}")
    model_class = _import_class(f"openue.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

def _save_model(litmodel, tokenizer, path):
    os.system(f"mkdir -p {path}")
    litmodel.model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    litmodel.config.save_pretrained(path)



def main():

    parser = _setup_parser()
    args = parser.parse_args()

    if not os.path.exists("config"):
        os.mkdir("config")
    config_file_name = time.strftime("%H:%M:%S", time.localtime()) + ".yaml"
    day_name = time.strftime("%Y-%m-%d")
    if not os.path.exists(os.path.join("config", day_name)):
        os.mkdir(os.path.join("config", time.strftime("%Y-%m-%d")))
    config = vars(args)
    with open(os.path.join(os.path.join("config", day_name), config_file_name), "w") as file:
        file.write(yaml.dump(config))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_class = _import_class(f"openue.data.{args.data_class}")
    model_class = _import_class(f"openue.models.{args.model_class}")
    litmodel_class = _import_class(f"openue.lit_models.{args.litmodel_class}")

    data = data_class(args)

    lit_model = litmodel_class(args=args, data_config=data.get_config())


    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="dialogue_pl")
        logger.log_hyperparams(vars(args))
    early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=5)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Eval/f1", mode="max",
        filename=args.data_dir.split("/")[-1] +'/'+ args.task_name + r'/{epoch}-{Eval/f1:.2f}',
        dirpath="output",
        save_weights_only=True
    )


    callbacks = [early_callback, model_checkpoint]

    # args.weights_summary = "full"  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs")
    
    
    test_only = "interactive" in args.task_name 

    if not test_only: trainer.fit(lit_model, datamodule=data)




    # two steps

    path = model_checkpoint.best_model_path

   

    # make sure the litmodel is the best model in dev
    if not test_only: lit_model.load_state_dict(torch.load(path)["state_dict"])

    # show the inference function
    if test_only:
        inputs = data.tokenizer("姚明出生在中国。", return_tensors='pt')
        print(lit_model.inference(inputs))


    trainer.test(lit_model, datamodule=data)
    
    if hasattr(lit_model.model, "save_pretrained"):
        _save_model(lit_model, data.tokenizer, path.rsplit("/", 1)[0])
    


if __name__ == "__main__":

    main()
