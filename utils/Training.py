# training
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch
import datetime
import time
import random
import numpy as np
import yaml
import json
from utils.Preprocessing import Preprocessing
from utils.Evaluation import evaluate
from utils.Logger import logger


class Training:
    def __init__(self, yfile: yaml) -> None:
        """
        This code defines the __init__ method for a training tool class for training BERT models.
        The method takes a single argument config_yaml_path, which is a string specifying the path to
        a YAML file that contains some settings for the training tool.
        The method first reads the contents of the YAML file specified by config_yaml_path and loads it into
        a Python object using the yaml.safe_load function.
        The resulting object is stored in an instance variable called yfile.
        The method then creates a PyTorch device object based on the "device" key in the YAML file.
        The value of this key is a string that specifies the device to use for training
        (e.g., "cuda:0" for using a GPU, or "cpu" for using the CPU).
        The resulting device object is stored in an instance variable called device.
        """
        self.yfile = yfile
        self.device = torch.device(self.yfile["device"])

    def formatTime(self, elapsed: float) -> str:
        """
        This method is used to format a time duration in seconds as a string in the format "h:mm:ss". 
        It takes an elapsed time duration in seconds as input, rounds it to the nearest second,
        and returns a string representing the elapsed time in hours, minutes, and seconds.
        For example, if elapsed is 3700.5 seconds, it will return the string "1:01:41" (1 hour, 1 minute, and 41 seconds).
        """
        return str(datetime.timedelta(seconds=int(round((elapsed)))))

    def save(self, tokenizer, model, save_path: str):
        """
        The save method saves the trained model and tokenizer to the specified location. The saved model can be later loaded and used for prediction.
        Parameters:
        tokenizer (AutoTokenizer): Tokenizer object from the preprocessing tool.
        model (AutoModelForSequenceClassification): Trained model object.
        save_path (str): Path to save the trained model and tokenizer.
        Returns:
        This method does not return anything.
        """
        tokenizer.save_pretrained(f"{save_path}_tokenizer")
        model.save_pretrained(f"{save_path}_model")

    def training(self) -> AutoModelForSequenceClassification:
        """
        The training method is used to train a BERT-based model for sequence classification.
        It takes no arguments other than self. The method first reads a CSV file containing 
        the training data and preprocesses it using the Preprocessing class. It then creates 
        a DataLoader object using the preprocessed training dataset.
        The method uses the AutoModelForSequenceClassification class from the Hugging Face Transformers library 
        to create a BERT model for sequence classification. The model is initialized with the model_name parameter 
        passed in from the yfile dictionary, and the number of labels is set to the number of classes in the preprocessed dataset. 
        The model is then moved to the device specified in the yfile dictionary.
        The method uses the AdamW optimizer with a learning rate of 5e-5 and an epsilon value of 1e-9. 
        It also uses a linear scheduler with no warmup steps, and the total number of training steps is calculated 
        as the product of the number of batches in the training dataloader and the number of epochs.
        The method then enters a training loop that iterates over the specified number of epochs. 
        For each epoch, the method loops through the batches in the training dataloader and performs forward and backward passes 
        on the model to compute the loss and update the model parameters. The method also clips the gradients to a maximum norm of 1.0.
        After each epoch, the method calculates the average training loss and training time, and saves the model and tokenizer 
        if the current average training loss is less than the previous average training loss. 
        The method also outputs the current and last average training loss values.
        Once training is complete, the method saves the training statistics to a JSON file 
        and calls the evaluate function to evaluate the trained model on the test dataset. Finally, the method returns the trained model.
        """
        seed_val = self.yfile["seed_val"]
        epochs = self.yfile["epochs"]
        batch_size = self.yfile["batch_size"]
        model_name = self.yfile["model_folder"]
        device = torch.device(self.yfile["device"])
        avg_loss = 100
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        preprocessing = Preprocessing(self.yfile)
        df = pd.read_csv(self.yfile["data_path"])
        preprocess_dict = preprocessing.trainPreprocess(
            df, self.yfile["text_column"], self.yfile["target_column"])
        number_of_categories = len(preprocess_dict["classes"])

        train_dataloader = DataLoader(
            preprocess_dict["train_dataset"],
            sampler=RandomSampler(preprocess_dict["train_dataset"]),
            batch_size=batch_size
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=number_of_categories,
            output_attentions=True,
            output_hidden_states=False
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=5e-5,
                                      eps=1e-9
                                      )

        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        training_stats = []
        total_t0 = time.time()
        for epoch_i in range(0, epochs):
            logger.info(
                '>>>>>>>>>>>>>>>>> Epoch {:} / {:} <<<<<<<<<<<<<<<'.format(epoch_i + 1, epochs))
            t0 = time.time()
            total_train_loss = 0
            model.train()

            for step, batch in enumerate(train_dataloader):
                if step % 100 == 0 and not step == 0:
                    elapsed = self.formatTime(time.time() - t0)
                    logger.info('Batch {:>3,}  of  {:>3,}.    Elapsed: {:}.'.format(
                        step, len(train_dataloader), elapsed))

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                model.zero_grad()
                output = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels)
                loss = output['loss']
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = self.formatTime(time.time() - t0)
            logger.info(
                "Training epoch elapsed time: {:}".format(training_time))
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Training Time': training_time,
                }
            )
            if avg_train_loss < avg_loss:
                self.save(preprocess_dict["tokenizer"],
                          model, self.yfile["output_path"])
                logger.info(
                    f"model is saved.\n>>> current average training loss: {avg_train_loss}\n last average training loss: {avg_loss}")
                avg_loss = avg_train_loss
            else:
                logger.info(
                    f"model is not saved. Because current average training loss bigger than last average training loss\n>>> current average training loss: {avg_train_loss}\n last average training loss: {avg_loss}")
        logger.info("Training completed in {:} (h:mm:ss)".format(
            self.formatTime(time.time()-total_t0)))
        train_json = open(
            self.yfile["training_json_file"], "w", encoding="utf-8")
        json.dump(training_stats, train_json, ensure_ascii=False, indent=4)
        train_json.close()
        evaluate(model, preprocess_dict["test_dataset"],
                 preprocess_dict["y_test"], batch_size)
        return model
