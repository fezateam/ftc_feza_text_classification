import argparse
import datetime
import chardet
from transformers import MarianMTModel, AutoTokenizer
import torch
def translate_file(src_file, output_file, model_name):
    device=torch.device('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    # Detect encoding of input file
    with open(src_file, 'rb') as input_file:
        result = chardet.detect(input_file.read())
    encoding = result['encoding']


    # Generate unique output file name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"{output_file}_{timestamp}.txt"

    with open(src_file, "r", encoding=encoding) as infile, open(output_file, "w", encoding='utf-8') as outfile:
        count = 0
        for line in infile:
            count += 1
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) != 3:
                continue
            src_text = cols[0]
            translated = model.generate(**tokenizer([src_text], return_tensors="pt", padding=True).to(device)).to(device)
            tgt_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            if tgt_text:
                print(count,tgt_text)
                outfile.write(f"{tgt_text}\t{cols[1]}\t{cols[-1]}\n")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Translate text from a file using MarianMTModel")
    parser.add_argument("-i", "--src_file", type=str, help="Source file name")
    parser.add_argument("-o", "--output_file", type=str, help="Output file name")
    parser.add_argument("-m", "--model_name", type=str, choices=["Helsinki-NLP/opus-mt-tc-big-tr-en", "Helsinki-NLP/opus-mt-tc-big-en-tr"], help="Model name")
    args = parser.parse_args()

    translate_file(args.src_file, args.output_file, args.model_name)


