import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from utils.Logger import logger


@torch.no_grad()
def evaluate(model: AutoModelForSequenceClassification, test_dataset: torch.Tensor, y_test, batch_size: int, device: str = "cuda:0") -> dict:
    """
    This code evaluates a trained BERT model on a test dataset and computes metrics such as f1_score, precision_score, recall_score, and accuracy_score.
    The evaluate function takes in the following arguments:
    model: The trained BERT model.
    test_dataset: The test dataset in the form of a PyTorch tensor.
    y_test: The true labels of the test dataset.
    batch_size: The batch size for evaluating the model.
    device: The device on which the evaluation should be performed (CPU or CUDA). The default is "cuda:0" which means the first available GPU.
    The function starts by creating a sampler and dataloader for the test dataset using the SequentialSampler and DataLoader classes from PyTorch.
    It then sets the model to evaluation mode using model.eval(), which turns off the gradients and speeds up computation.
    For each batch in the test dataset, the function feeds the input ids, attention masks, and labels to the model and computes the logits.
    The logits are then converted to numpy arrays and appended to the predictions list, while the labels are appended to the true_labels list.
    The function then computes the predicted labels by taking the argmax of the logits and flattening the result.
    The predicted labels are appended to the prediction_set list. The prediction_scores list is then created by flattening the prediction_set list.
    Finally, the function computes the f1_score and accuracy_score using scikit-learn's corresponding functions and prints them.
    """
    device = torch.device(device)
    prediction_sampler = SequentialSampler(test_dataset)
    prediction_dataloader = DataLoader(
        test_dataset, sampler=prediction_sampler, batch_size=batch_size)
    model.eval()
    predictions, true_labels = [], []

    for batch in tqdm(prediction_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(label_ids)
    prediction_set = []
    for i in range(len(true_labels)):
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        prediction_set.append(pred_labels_i)
    prediction_scores = [
        item for sublist in prediction_set for item in sublist]
    f_score = f1_score(y_test, prediction_scores, average='macro')
    accr = accuracy_score(y_test, prediction_scores)
    logger.info(">>>>>>>>> SCORES: <<<<<<<<<<")
    logger.info("F-Score: ", f_score)
    logger.info("Accuracy: ", accr)
    scores_dict = {
        "f_score": f_score,
        "accuracy": accr
    }
    return scores_dict
