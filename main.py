
from utils.Training import Training
from utils.Preprocessing import Preprocessing
from utils.Evaluation import evaluate
from utils.Prediction import prediction
import yaml
import argparse
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification


def importYfile(config_yaml_path: str):
    """
        The importYfile function takes a path to a YAML configuration file and returns a dictionary containing the configuration parameters.
        Inputs:
        config_yaml_path (str): Path to the YAML configuration file.
        Outputs:
        yfile (dict): A dictionary containing the configuration parameters.
    """
    yfile = open(config_yaml_path, "r").read()
    yfile = yaml.safe_load(yfile)
    return yfile


def train(yfile: yaml):
    """
        Trains a BERT model for sequence classification using the parameters specified in a YAML file.
        Args:
            config_yaml_path (str): Path to the YAML file containing the training configuration parameters.
        Returns:
            AutoModelForSequenceClassification: The trained BERT model.
        Raises:
            Exception: If an error occurs during the training process.
        Example:
            >>> trained_model = train("config.yaml")
    """
    training = Training(yfile)
    model = training.training()
    return model


def evaluation(yfile: yaml):
    """
        The evaluation function takes in a path to a configuration YAML file as input. 
        The YAML file should contain the necessary parameters for evaluating a trained BERT model on a test dataset.
        The function first imports the YAML file using the importYfile function and initializes a Preprocessing object with 
        the YAML parameters. It then loads the test dataset using the pd.read_csv function and preprocesses it using the main method of the Preprocessing object.
        The function then loads the trained BERT model using the AutoModelForSequenceClassification.
        from_pretrained function and evaluates it on the preprocessed test dataset using the evaluate function. 
        The function returns the evaluation scores in the form of a dictionary.

    """
    preprocessing = Preprocessing(yfile)
    df = pd.read_csv(yfile["data_path"])
    preprocess_dict = preprocessing.trainPreprocess(
        df, yfile["text_column"], yfile["target_column"])
    number_of_categories = len(preprocess_dict["classes"])
    device = torch.device(yfile["device"])
    model = AutoModelForSequenceClassification.from_pretrained(
        yfile["output_path"]+"_model",
        num_labels=number_of_categories,
        output_attentions=True,
        output_hidden_states=False
    ).to(device)
    scores = evaluate(
        model, preprocess_dict["test_dataset"], preprocess_dict["y_test"], yfile["batch_size"])
    return scores


def predictor(yfile: yaml) -> list:
    """
    Uses the configuration file at the given path to predict the classes of input texts.
    Args:
        config_yaml_path (str): Path to the YAML configuration file.
    Returns:
        list: A list of predicted classes for each input text.
    """
    preds = prediction(yfile)
    return preds


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml_path', '-cfg', type=str, default='./config.yaml',
                        help='Config yaml file to set parameters.')
    parser.add_argument('--mode', type=str, default='train',
                        help='Choose mode ["train", "predict", "eval"]')
    opt = parser.parse_args()
    return opt


def main(opt: argparse.ArgumentParser):
    """
        The main function is the main entry point of the program that takes an argument parser opt and executes one of three modes of
        operation based on the mode attribute of opt: "train", "eval", or "predict". 
        The importYfile function is used to load the configuration YAML file specified by config_yaml_path. 
        The appropriate function from mode_dict is called with the parameters specified in parameters_dict. 
        Finally, the result of the mode operation is returned.
    """
    yfile = importYfile(opt.config_yaml_path)
    mode_dict = {
        "train": train,
        "eval": eval,
        "predict": predictor
    }
    parameters_dict = {
        "train": [yfile],
        "eval": [yfile],
        "predict": [yfile]
    }
    result = mode_dict[opt.mode](*parameters_dict[opt.mode])
    return result


if __name__ == "__main__":
    opt = parse_opt()
    result = main(opt)
