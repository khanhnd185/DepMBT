import os
import torch
import pickle
import argparse
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import BertModel, BertTokenizerFast


audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

text_tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
text_model = BertModel.from_pretrained("bert-base-chinese")


def pad_sequences(sequences, max_len=None, padding_value=0.0):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded_sequences = np.full((len(sequences), max_len, sequences[0].shape[1]), padding_value)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    
    return padded_sequences


def extract_score(file_paths):
    with open(file_paths, 'r', encoding='utf-8') as f:
        score = float(f.readline())

    return score

def extract_text(file_paths):
    with open(file_paths, 'r', encoding='utf-8') as f:
        text = f.readline()

    token = text_tokenizer.tokenize(text)
    token = text_tokenizer.convert_tokens_to_ids(token)
    token = torch.tensor(token).unsqueeze(0)
    with torch.no_grad():
        output = text_model(token).last_hidden_state.squeeze(0).cpu().numpy()  

    return output


def extract_audio(file_path):
    speech, sample_rate = torchaudio.load(file_path)
    #speech = speech.squeeze(0)  # Remove the unnecessary channel dimension
    speech = speech[0]
    inputs = audio_processor(speech, return_tensors="pt", padding=True, sampling_rate=sample_rate)
    with torch.no_grad():
        outputs = audio_model(**inputs)
        logits = outputs.logits.squeeze(0).cpu().numpy()  
    
    return logits


def extract(data_path):
    dataset = []
    samples = os.listdir(data_path)

    for sample in (samples):
        print(sample)
        sample_path = data_path + sample

        score = extract_score(sample_path + '/new_label.txt')
        text_neg = extract_text(sample_path + '/negative.txt')
        text_pos = extract_text(sample_path + '/positive.txt')
        text_neu = extract_text(sample_path + '/neutral.txt')
        audio_neg = extract_audio(sample_path + '/negative_out.wav')
        audio_pos = extract_audio(sample_path + '/positive_out.wav')
        audio_neu = extract_audio(sample_path + '/neutral_out.wav')

        dataset.append({"name": sample,
                        "textneg": text_neg,
                        "textpos": text_pos,
                        "textneu": text_neu,
                        "audioneg": audio_neg,
                        "audiopos": audio_pos,
                        "audioneu": audio_neu,
                        "score": score})
        
    return dataset

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Generate image feature')
    parser.add_argument('--datadir', '-d', default='../../../../Data/EATD-Corpus/EATD-Corpus/', help='Data folder path')
    parser.add_argument('--output', '-o', default='eatd.pickle', help='Output file name')
    args = parser.parse_args()

    dataset = extract(args.datadir)

    with open(args.datadir + '../' + args.output, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


