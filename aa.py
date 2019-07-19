# -*- coding: utf-8 -*-
"""
Created on Sun May  5 20:56:30 2019

@author: Sushma
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
import tkinter as tk
class mclass:
    def __init__(self,  window):
        self.window = window
        self.box = Entry(window)
        self.button = Button (window, text="Click here", command=self.plot)
        self.plot
        self.box.pack ()
        self.button.pack()
    def plot (self):
        from sklearn.metrics import roc_curve
        fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.y_act, df.y_pred_rf)
        fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.y_act, df.y_pred_lr)
        fpr = [0.,0.5,1.]
        acu = compute_accuracy(tp_rf, tn_rf, fn_rf, fp_rf)
        fig = Figure(figsize=(6,6))
        a = fig.add_subplot(111)
        a.plot([0.,0.25,0.50,0.75,0.90], [0.50,0.50,0.67,0.51,0.50],'b-',label = 'ACU: %.3f'%acu)
        #a.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
        a.legend()
        #a.invert_yaxis()

        a.set_title ("Estimation Grid", fontsize=16)
        a.set_ylabel("Accuracy", fontsize=14)
        a.set_xlabel("Threshold", fontsize=14)
# =============================================================================
        from sklearn.metrics import roc_auc_score
        auc_RF = roc_auc_score(df.y_act, df.y_pred_rf)
        auc_LR = roc_auc_score(df.y_act, df.y_pred_lr)
        pre = compute_precision(tp_rf, fp_rf)
        fig1 = Figure(figsize=(6,6))
        b = fig1.add_subplot(111)
        b.plot([0.,0.25,0.50,0.75,0.90], [0.50,0.50,0.68,0.99,1.00],'r-',label = 'PRE: %.3f'%pre)
        #b.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
        b.legend()
        #b.invert_yaxis()

        b.set_title ("Estimation Grid", fontsize=16)
        b.set_ylabel("Precision", fontsize=14)
        b.set_xlabel("Threshold", fontsize=14)
# =============================================================================
        from sklearn.metrics import roc_curve
        fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.y_act, df.y_pred_rf)
        fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.y_act, df.y_pred_lr)
        fpr = [0.,0.5,1.]
        rec = compute_recall(tp_rf, fn_rf)
        fig2 = Figure(figsize=(6,6))
        c = fig2.add_subplot(111)
        c.plot([0.,0.25,0.50,0.75,0.90], [1.00,1.00,0.64,0.02,0.06],'k-',label = 'REC: %.3f'%rec)
        #c.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
        c.legend()
        #c.invert_yaxis()

        c.set_title ("Estimation Grid", fontsize=16)
        c.set_ylabel("Recall", fontsize=14)
        c.set_xlabel("Threshold", fontsize=14)

        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        canvas.draw()
        canvas = FigureCanvasTkAgg(fig1, master=self.window)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        canvas.draw()
        canvas = FigureCanvasTkAgg(fig2, master=self.window)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        canvas.draw()

file_path = 'data.csv'
df = pd.read_csv(file_path)
df.head()
thresh = 0.5
df['y_pred_rf'] = (df.y_pred_random_forest>=0.5).astype('int')
df['y_pred_lr'] = (df.y_pred_logistic >= 0.5).astype('int')
df.head()
#print(df.shape)
def compute_tp_tn_fn_fp(y_act, y_pred):
	'''
	True positive - actual = 1, predicted = 1
	False positive - actual = 1, predicted = 0
	False negative - actual = 0, predicted = 1
	True negative - actual = 0, predicted = 0
	'''
	tp = sum((y_act == 1) & (y_pred == 1))
	tn = sum((y_act == 0) & (y_pred == 0))
	fn = sum((y_act == 1) & (y_pred == 0))
	fp = sum((y_act == 0) & (y_pred == 1))
	return tp, tn, fp, fn

tp_lr, tn_lr, fp_lr, fn_lr = compute_tp_tn_fn_fp(df.y_act, df.y_pred_lr)
tp_rf, tn_rf, fp_rf, fn_rf = compute_tp_tn_fn_fp(df.y_act, df.y_pred_rf)

from sklearn.metrics import confusion_matrix
tn_rf1, fp_rf1, fn_rf1, tp_rf1 = confusion_matrix(df.y_act, df.y_pred_rf).ravel()

def compute_accuracy(tp, tn, fn, fp):
	'''
	Accuracy = TP + TN / FP + FN + TP + TN

	'''
	return ((tp + tn) * 100)/ float( tp + tn + fn + fp)

print('Accuracy :', compute_accuracy(tp_rf, tn_rf, fn_rf, fp_rf))


def compute_precision(tp, fp):
	'''
	Precision = TP  / FP + TP 

	'''
	return (tp  * 100)/ float( tp + fp)

print('Precision :', compute_precision(tp_rf, fp_rf))


def compute_recall(tp, fn):
	'''
	Recall = TP /FN + TP 

	'''
	return (tp  * 100)/ float( tp + fn)

print('Recall :', compute_recall(tp_rf, fn_rf))

print(' ')
# =============================================================================

def show_graphs():
    window= tk.Tk()
    start=mclass(window)
    window.mainloop()