import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
import pandas as pd
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
from bm import compute_accuracy, compute_precision, compute_recall
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    #print(sampled_ids)
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        #print(word_id)
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (sentence)
   
    #add
    #message = raw_input("Enter message to encode: ")

# =============================================================================
#     print("Decoded string (in ASCII):")
#     for ch in sentence:
#         print(ord(ch))
#     print("\t")
# =============================================================================
    
    sen=list(sentence.split(" "))
    #print(sen)
    sen1 = sen[1:-1]
    #print([sen1])
    #end
    
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))
    #print(args.image)
    
    #add2
    if args.image=="png/ex1.jpg":
        caption = ['a', 'close', 'up', 'of', 'an', 'elephant', 'on', 'a', 'road', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex4.jpg":
        caption = ['a', 'man', 'is', 'sitting', 'at', 'a', 'table', 'with', 'a', 'laptop', 'on', 'it']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
        
    elif args.image=="png/ex2.jpg":
        caption = ['a', 'man', 'holding', 'tennis', 'racket', 'in', 'a', 'tennis', 'court']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex3.jpg":
        caption = ['a', 'man', 'and', 'woman', 'are', 'standing', 'near', 'a', 'beach', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex5.jpg":
        caption = ['a', 'group', 'of', 'people', 'sitting', 'in', 'a', 'room', 'working', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex6.jpg":
        caption = ['a', 'man', 'playing', 'tennis', 'in', 'a', 'court', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex7.jpg":
        caption = ['a', 'fire', 'hydrant', 'is', 'on', 'a', 'snowy', 'streets', 'with', 'trees', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex9.jpg":
        caption = ['a', 'man', 'sitting', 'at', 'a', 'table', 'talking', 'to', 'another', 'man', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/img5.jpg":
        caption = ['a', 'vase', 'filled', 'with', 'flowers', 'on', 'a', 'table', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/img10.jpg":
        caption = ['a', 'woman', 'is', 'sitting', 'at', 'a', 'table', 'with', 'a', 'cake', 'on', 'it', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex12.jpg":
        caption = ['motocycles', 'parked', 'in', 'a', 'parking', 'lot', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex13.jpg":
        caption = ['a', 'zebra', 'standing', 'next', 'to', 'a', 'zebra', 'on', 'an', 'ice', 'road', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex14.jpg":
        caption = ['a', 'black', 'dog', 'and', 'two', 'cats', 'laying', 'on', 'a', 'bed', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex16.jpg":
        caption = ['a', 'woman', 'is', 'cutting', 'apples', 'at', 'a', 'table', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex19.jpg":
        caption = ['a', 'black', 'bear', 'is', 'walking', 'through', 'a', 'stony', 'road', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex21.jpg":
        caption = ['a', 'table', 'with', 'many', 'plates', 'of', 'food', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex22.jpg":
        caption = ['a', 'brown', 'bear', 'sitting', 'in', 'the', 'grass', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex24.jpg":
        caption = ['a', 'group', 'of', 'people', 'playing', 'in', 'a', 'field', 'with', 'a', 'frisbee', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/ex25.jpg":
        caption = ['a', 'group', 'of', 'sheep', 'standing', 'in', 'a', 'field', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/example2.jpg":
        caption = ['a', 'truck', 'and', 'a', 'car', 'parked', 'in', 'a', 'parking', 'lot', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/2.jpg":
        caption = ['a', 'street', 'scene', 'with', 'a', 'traffic', 'light', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/3.jpg":
        caption = ['a', 'group', 'of', 'people', 'sitting', 'around', 'tables', 'with', 'their', 'laptops', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/5.jpg":
        caption = ['two', 'plates', 'of', 'food', 'with', 'a', 'fork', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/7.jpg":
        caption = ['a', 'train', 'is', 'pulling', 'up', 'to', 'a', 'railway', 'station', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/8.jpg":
        caption = ['a', 'man', 'standing', 'at', 'a', 'table', 'holding', 'plates', 'containing', 'food', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/9.jpg":
        caption = ['people', 'sitting', 'in', 'a', 'living', 'room', 'with', 'a', 'couch', 'and', 'television', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/10.jpg":
        caption = ['a', 'dog', 'sitting', 'in', 'a', 'car', 'looking', 'out', 'from', 'the', 'window', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/14.jpg":
        caption = ['a', 'man', 'flying', 'a', 'kite', 'in', 'a', 'ground', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/15.jpg":
        caption = ['a', 'bathroom', 'with', 'a', 'sink', 'and', 'towels', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/17.jpg":
        caption = ['motorcycles', 'parked', 'in', 'a', 'parking', 'lot', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/18.jpg":
        caption = ['a', 'man', 'standing', 'an', 'a', 'road', 'in', 'front', 'of', 'a', 'shop', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/19.jpg":
        caption = ['a', 'man', 'and', 'a', 'woman', 'playing', 'tennis', 'in', 'a', 'tennis', 'court', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/20.jpg":
        caption = ['a', 'plate', 'of', 'food', 'with', 'a', 'fork', 'and', 'a', 'knife', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/41.jpg":
        caption = ['a', 'baby', 'sleeping', 'on', 'a', 'bed', 'with', 'a', 'stuffed', 'toy', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/43.jpg":
        caption = ['a', 'group', 'of', 'men', 'standing', 'next', 'to', 'each', 'other', 'holiding', 'tennis', 'rackets', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/50.jpg":
        caption = ['a', 'group', 'of', 'people', 'riding', 'on', 'horses', 'ín', 'a', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/52.jpg":
        caption = ['a', 'man', 'is', 'surfing', 'in', 'the', 'ocean', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/63.jpg":
        caption = ['different', 'types', 'of', 'doughnuts', 'on', 'a', 'table', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/67.jpg":
        caption = ['a', 'man', 'is', 'jumping', 'with', 'a', 'skateboard', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/68.jpg":
        caption = ['a', 'herd', 'of', 'cows', 'grazing', 'on', 'a', 'hillside', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/69.jpg":
        caption = ['a', 'dog', 'sitting', 'in', 'a', 'car', 'looking', 'out', 'from', 'the', 'window', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/70.jpg":
        caption = ['a', 'dog', 'sitting', 'in', 'a', 'car', 'looking', 'out', 'from', 'the', 'window', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/71.jpg":
        caption = ['a', 'dog', 'sitting', 'in', 'a', 'car', 'looking', 'out', 'from', 'the', 'window', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    elif args.image=="png/72.jpg":
        caption = ['a', 'dog', 'sitting', 'in', 'a', 'car', 'looking', 'out', 'from', 'the', 'window', '.']
        print(caption)
        score1 = bluescore([sen1],caption)
        print(score1)
    
# =============================================================================

# =============================================================================
#     Accuracy= compute_accuracy(tp_rf, tn_rf, fn_rf, fp_rf)
#     Precision= compute_precision(tp_rf, fp_rf)
#     Recall= compute_recall(tp_rf, fn_rf)
#=============================================================================
#    print('Accuracy :', compute_accuracy(tp_rf, tn_rf, fn_rf, fp_rf))
#   print('Precision :', compute_precision(tp_rf, fp_rf))
 #   print('Recall :', compute_recall(tp_rf, fn_rf))
#=============================================================================
    return sentence,score1
    
#blue
def bluescore(ref,cap):
    score = sentence_bleu(ref, cap, weights=(1,0,0,0,0,0))
    return score
#endbleu       
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()   #I changed here
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)