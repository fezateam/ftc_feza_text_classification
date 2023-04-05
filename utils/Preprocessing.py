from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset
from unidecode import unidecode
import torch
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import string
import csv
import chardet
from utils.Logger import logger
nltk.download("stopwords")


class Preprocessing:
    def __init__(self, yfile) -> None:
        """
        This is a class constructor that initializes the object of the class. 
        The constructor takes in one argument, yfile, which is a dictionary containing the configuration details for the object.
        The constructor first stores the yfile parameter as an instance variable for later use. It then extracts the path to the tokenizer 
        folder from yfile and creates an instance of the tokenizer using the AutoTokenizer class from the Hugging Face transformers library. 
        The do_lower_case parameter is set to True to ensure that all text is converted to lowercase before tokenization.
        The constructor also initializes a LabelEncoder object and stores it as an instance variable for later use.
        """
        self.yfile = yfile
        tokenizer_folder = yfile["tokenizer_folder"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_folder, do_lower_case=True)
        self.le = LabelEncoder()

    def encodeLabel(self, df: pd.DataFrame, target_field_name: str) -> pd.DataFrame:
        """
        The encodeLabel method is used for label encoding of the target column in a pandas DataFrame.
        Inputs:
            df (DataFrame): Pandas dataframe from dataset
            target_field_name (str): Target column name in the dataframe.
        Outputs:
             (DataFrame): Dataframe which has encoded labels.

        The method returns a pandas DataFrame with an additional column called "target" which contains the encoded label values.
        The encoding is performed using the fit_transform() method of LabelEncoder class.
        """
        target = df[target_field_name].values
        target = self.le.fit_transform(target)
        df["target"] = target
        return df

    def textTokenizer(self, text: str, mode: str, max_len: int = 64) -> dict:
        """
        This is a Python function called textTokenizer that takes in a string of text,
        a mode indicating the type of encoded dictionary to return, and an optional maximum length for each token.
        The function uses the Hugging Face tokenizer object to encode the input text into an encoded dictionary suitable for use in a BERT model.
        The text argument is the input text to be encoded, and the mode argument is a string indicating which part of the encoded dictionary to return.
        The max_len argument is an optional integer specifying the maximum length of each token after encoding.
        If not provided, the default value of 64 is used.
        The function uses the encode_plus() method of the Hugging Face tokenizer object to encode the input text into an encoded dictionary.
        The add_special_tokens=True argument tells the tokenizer to add special tokens to the beginning and end of the sequence,
        which are required for BERT models.
        The max_length argument specifies the maximum length of the encoded sequence,
        and the truncation=True argument tells the tokenizer to truncate any sequence longer than the specified maximum length.
        The pad_to_max_length=True argument tells the tokenizer to pad any sequence shorter than the specified maximum length with zeros.
        Finally, the return_attention_mask=True argument tells the tokenizer to include an attention mask in the encoded dictionary,
        which is also required for BERT models. The return_tensors='pt' argument tells the tokenizer to return the encoded dictionary as PyTorch tensors.
        The function then returns the requested part of the encoded dictionary (input_ids or attention_mask) as a dictionary.
        The returned dictionary can be directly used as input to a BERT model.
        """

        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoded_dict[mode]

    def toTensors(self, input_ids: list, attention_mask: list, target: list = []) -> tuple:
        """
        This is a Python function called toTensors that takes in three lists (input_ids, attention_mask, and target)
        as input and returns a tuple of three tensors. This function is used to convert lists of encoded inputs and labels 
        into PyTorch tensors for use in a neural network.
        The input_ids and attention_mask arguments are both lists of PyTorch tensors, 
        representing the encoded input sequences and attention masks, respectively. 
        These lists are concatenated along the 0th dimension using torch.cat() to create a single tensor for each.
        The target argument is an optional list of labels for the input sequences. If this argument is provided, 
        it is converted to a PyTorch tensor using torch.tensor(). If target is empty, the function returns only 
        input_ids and attention_mask tensors. Otherwise, it returns all three tensors (input_ids, attention_mask, and target) as a tuple.
        The docstring for this function describes the arguments and return values, as well as their data types. 
        The input_ids and attention_mask arguments are both lists of PyTorch tensors, while the target argument is a list of labels. 
        The return value is a tuple of three PyTorch tensors, representing the converted input_ids, attention_mask, and target (if provided).

        """
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        if len(target) == 0:
            return input_ids, attention_mask
        else:
            target = torch.tensor(target)
            return input_ids, attention_mask, target

    def charConverter(self, text: str) -> str:
        """
        This is a Python function called charConverter 
        that takes in a string of text as input and returns a converted version of the text. 
        The function uses the unidecode function from the unidecode library to convert any non-ASCII characters to their closest ASCII equivalents.
        The unidecode function takes a Unicode string as input and returns an ASCII representation of the string. 
        This is useful for text processing tasks that require ASCII input, as it can help avoid errors or inconsistencies in the text.
        The function simply takes in the input text, applies the unidecode function to it, and returns the converted text as output.
        Note that this function only performs character conversion, and does not perform any other text cleaning or preprocessing tasks. 
        If you need to perform additional text cleaning or preprocessing, you may need to use additional functions or libraries in your code.
        """
        return unidecode(text)

    def textCleaning(self, text: str) -> str:
        """
        This is a Python function called textCleaning that takes in a string of text as input and returns a cleaned version of the text.
        The function performs various text cleaning operations on the input text, as described below:
        Remove stop words: The function removes common stop words in Turkish language by using stopwords.words("turkish") from nltk library.
        Decode text: The text is encoded to ASCII format to remove any non-ASCII characters or symbols.
        Remove white spaces: Any additional white spaces are removed from the text.
        Remove mentions: Any mentions in the text (starting with @) are removed.
        Remove market tickers: Any dollar signs ($) are removed from the text.
        Remove non-alphabetic: Any numeric characters are removed from the text.
        Remove urls: Any URLs in the text are removed.
        Remove hashtags: Any hashtags (#) are removed from the text.
        Remove old style retweet text "RT": Any old style retweet text (starting with RT) is removed.
        Remove punctuations: Any punctuation marks are removed from the text.
        Remove one chars: Any words with a length of 2 characters or less are removed from the text.
        """
        # Remove stop words
        stop_words: list = stopwords.words("turkish")
        text = " ".join([word for word in text.split()
                        if word not in stop_words])
        # Decode text
        text = text.encode(encoding="ascii", errors="ignore")
        text = text.decode()
        # Remove white spaces
        text = " ".join([word for word in text.split()])
        # Remove mentions
        text = re.sub("@\S+", "", text)
        # Remove market tickers
        text = re.sub("\$", "", text)
        # Remove nonalphabetic
        text = re.sub("\d+", "", text)
        # Remove urls
        text = re.sub("https?:\/\/.*[\r\n]*", "", text)
        # Remove hashtags
        text = re.sub("#", "", text)
        # remove old style retweet text "RT"
        text = re.sub(r'^RT[\s]+', '', text)
        # Remove punctuations
        punct = set(string.punctuation)
        text = "".join([ch for ch in text if ch not in punct])
        # Remove one chars
        text = " ".join([i for i in text.split() if len(i) > 2])
        return text

    def train_test_split(self, df: pd.DataFrame, target_column: str) -> tuple:
        """
        Train and test split for training with dataframe. The classes in the data are shared equally to Train and test datasets.
        Args:
            df (pd.DataFrame): Dataframe
            target_column (str): Column name which has labels
        Returns:
            tuple: A tuple containing two dataframes - `df_train` and `df_test`
        """

        df_train = df.groupby(target_column).apply(
            lambda x: x.sample(frac=self.yfile["train_split_ratio"]))
        df_test = pd.concat([df, df_train]).drop_duplicates(keep=False)
        logger.info(
            f">>> Training data count: {len(df_train)}\n>>> Test data count: {len(df_test)}")
        return df_train, df_test

    def predictionPreprocess(self, text_list: list) -> tuple:
        """Preprocessing for prediction.

        Args:
            text_list (list): List of strings for prediction.

        Returns:
            tuple: Tensors required for prediction.
        """

        text_list = [self.charConverter(text) for text in text_list]
        text_list = [self.textCleaning(text) for text in text_list]
        text_input_ids = list(
            map(lambda x: self.textTokenizer(x, "input_ids"), text_list))
        text_attention_mask = list(
            map(lambda x: self.textTokenizer(x, "attention_mask"), text_list))
        text_input_ids, text_attention_mask = self.toTensors(
            text_input_ids, text_attention_mask)
        return text_input_ids, text_attention_mask

    def augmentationPreprocess(self):
        # Define input and output filenames
        input_filename = "teknofest_train_final.csv"
        output_filename = "reformatted_with_classes.txt"

        # Detect encoding of input file
        with open(input_filename, 'rb') as input_file:
            result = chardet.detect(input_file.read())
        encoding = result['encoding']

        # Read input CSV file and split values in first column by delimiter "|"
        data = []
        with open(input_filename, newline='', encoding=encoding, errors='ignore') as csvfile:
            reader = csv.reader(csvfile)
            x = 0
            for row in reader:
                x+=1
                # try:
                values = row[0].split("|")
                data.append(values)
                # except Exception as e:
                    # print(e)
                    # continue

        # Create new list with second to last and second values from original data, but only for rows where first column is an integer
        new_data = [[row[-2], row[-1],row[1]] for row in data if row[-2].isdigit()]

        # Remove rows where second column in output file has only one character
        new_data = [row for row in new_data if len(row[-1]) > 1]

        # Write reformatted data to output TXT file
        with open(output_filename, 'w', newline='',encoding=encoding, errors='ignore') as txtfile:
            for row in new_data:
                txtfile.write("\t".join(row) + "\n")

    def trainPreprocess(self, df: pd.DataFrame, text_column: str, target_column: str) -> dict:
        """
        This function is doing data preprocessing steps for training. It first drops any rows containing missing values and then applies the following steps in order:

        charConverter: Converts the texts in the specified column into a standardized format.
        textCleaning: Applies text cleaning to the texts in the specified column.
        encodeLabel: Encodes the labels in the specified column using label encoding.
        train_test_split: Splits the dataset into training and testing datasets with the ratio specified in the yfile parameter.
        Tokenizes the texts using the BERT tokenizer and converts them into tensors.
        Wraps the tensors and labels in TensorDataset objects.
        Finally, it returns a dictionary containing the following items:

        "train_dataset": TensorDataset object containing the training data.
        "test_dataset": TensorDataset object containing the testing data.
        "tokenizer": The BERT tokenizer used for tokenization.
        "classes": An array containing the class labels.
        "x_train": An array containing the cleaned and preprocessed training texts.
        "x_test": An array containing the cleaned and preprocessed testing texts.
        "y_train": An array containing the label-encoded training labels.
        "y_test": An array containing the label-encoded testing labels.

        """
        result_dict = {}
        df = df.dropna().reset_index(drop=True)
        cleaned_text_col = df[text_column].apply(self.charConverter)
        cleaned_text_col = cleaned_text_col.apply(self.textCleaning)
        df["cleaned_text"] = cleaned_text_col.values
        df = self.encodeLabel(df, target_column)
        train_df, test_df = self.train_test_split(df, target_column)

        x_train = train_df["cleaned_text"].values
        x_test = test_df["cleaned_text"].values
        y_train = train_df["target"].values
        y_test = test_df["target"].values

        tr_input_ids = list(
            map(lambda x: self.textTokenizer(x, "input_ids"), x_train))
        te_input_ids = list(
            map(lambda x: self.textTokenizer(x, "input_ids"), x_test))
        tr_attention_mask = list(
            map(lambda x: self.textTokenizer(x, "attention_mask"), x_train))
        te_attention_mask = list(
            map(lambda x: self.textTokenizer(x, "attention_mask"), x_test))

        tr_input_ids, tr_attention_mask, y_train = self.toTensors(
            tr_input_ids, tr_attention_mask, y_train)
        te_input_ids, te_attention_mask, y_test = self.toTensors(
            te_input_ids, te_attention_mask, y_test)

        train_dataset = TensorDataset(tr_input_ids, tr_attention_mask, y_train)
        test_dataset = TensorDataset(te_input_ids, te_attention_mask, y_test)
        result_dict["train_dataset"] = train_dataset
        result_dict["test_dataset"] = test_dataset
        result_dict["tokenizer"] = self.tokenizer
        result_dict["classes"] = self.le.classes_
        result_dict["x_train"] = x_train
        result_dict["x_test"] = x_test
        result_dict["y_train"] = y_train
        result_dict["y_test"] = y_test

        return result_dict
