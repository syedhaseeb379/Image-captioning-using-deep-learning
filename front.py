# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:23:28 2019

@author: lenovo
"""
import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from sample1 import main
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from aa import show_graphs
import tkinter

from tkinter import *
from PIL import Image
from PIL import ImageTk
top = Tk()
top.title('Image Captioning Using Deep Learning')
top.geometry('625x800')

top_row = Frame(top).grid(row=0)

left = Frame(top_row).grid(row=0, column=0)
L1 = Label(left, font=('helvetica', 14)).grid(row=0, column=2)
# =============================================================================
# L1.config(font=('helvetica', 14))
# =============================================================================
E1 = Entry(left)
#E1.grid(row=1)

right = Frame(top_row).grid(row=0, column=1)

canvas = Canvas(right, width=600,height=500, bd=0,bg='white', relief= 'raised')
canvas.grid(row=1, column=2)
# =============================================================================
load = Image.open("png/logo2.png")
w, h = load.size
load = load.resize((600, 500))
imgfile = ImageTk.PhotoImage(load)
   
canvas.image = imgfile  # <--- keep reference of your image
canvas.create_image(2,2,anchor='nw',image=imgfile)
# =============================================================================

def GenerateImage():
    #parser = argparse.ArgumentParser()   #I changed here

    #args = parser.parse_args()

    #sentence,image=main(args)
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
    sentence,score=main(args)
    save_name=args.image
    text1=sentence[7:-5]
    L1 = Label(left, text="Caption:"+text1, font=('helvetica', 16)).grid(row=5, column=2)
    text2=str(score)
    L1 = Label(left, text="Blue Score:"+text2, font=('helvetica', 16)).grid(row=6, column=2)

            
    load = Image.open(save_name)
    w, h = load.size
    load = load.resize((600, 500))
    imgfile = ImageTk.PhotoImage(load)
    
    canvas.image = imgfile  # <--- keep reference of your image
    canvas.create_image(2,2,anchor='nw',image=imgfile)
    
    #E1.delete(0, END)
def close_window():
    top.destroy()
    
submit_button = Button(top, text ='Generate Caption', command = GenerateImage, bg='brown', fg='white', font=('helvetica', 12, 'bold'))
submit_button.grid(row=2, column=2)
submit_button = Button(top, text ='Show Graphs', command =show_graphs, bg='brown', fg='white', font=('helvetica', 12, 'bold'))
submit_button.grid(row=3, column=2)
submit_button = Button(top, text ='Exit', command =close_window, bg='red', fg='white', font=('helvetica', 12, 'bold'))
submit_button.grid(row=4, column=2)


top.mainloop()

