from transformers import AutoModelForSequenceClassification
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from utils.Preprocessing import Preprocessing
import yaml
import pandas as pd


@torch.no_grad()
def prediction(yfile: yaml) -> list:
    """
    Predicts the class labels of a list of texts using a pre-trained BERT model.

    Args:
        text_list (list): List of texts to predict the class labels for.
        yfile: A dictionary containing configuration parameters such as tokenizer folder, model folder, classes, etc.
        model_folder (str): Path to the folder containing the pre-trained BERT model.
        classes (list): List of class labels.
        device (str): Device to use for inference. Defaults to "cuda:0".
        batch_size (int): Batch size for evaluating. Defaults to 1.

    Returns:
        A list of predicted class labels for the input texts.
    """
    model_folder = yfile["output_path"]+"_model"
    classes = yfile["classes"]
    batch_size = yfile["batch_size"]
    device = torch.device(yfile["device"])
    text_list = pd.read_csv(
        yfile["data_path"]).loc[:, yfile["text_column"]].values
    preprocessing = Preprocessing(yfile)
    number_of_categories = len(classes)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_folder,
        num_labels=number_of_categories,
        output_attentions=True,
        output_hidden_states=False,
    ).to(device).eval()
    te_input_ids, te_attention_mask = preprocessing.predictionPreprocess(
        text_list)
    test_dataset = TensorDataset(te_input_ids, te_attention_mask)
    prediction_sampler = SequentialSampler(test_dataset)
    prediction_dataloader = DataLoader(
        test_dataset, sampler=prediction_sampler, batch_size=batch_size)
    predictions = []
    for batch in tqdm(prediction_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        pred = np.argmax(logits, axis=1).flatten()
        predictions.append(classes[pred[0]])

    return predictions
