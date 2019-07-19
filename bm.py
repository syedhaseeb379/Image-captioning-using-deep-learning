# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:47:41 2019

@author: Sushma
"""

import pandas as pd
import numpy as np

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
#print('TP for Logistic Reg :', tp_lr)
#print('TN for Logistic Reg :', tn_lr)
#print('FP for Logistic Reg :', fp_lr)
#print('FN for Logistic Reg :', fn_lr)

tp_rf, tn_rf, fp_rf, fn_rf = compute_tp_tn_fn_fp(df.y_act, df.y_pred_rf)
#print('TP for Random Forest :', tp_rf)
#print('TN for Random Forest :', tn_rf)
#print('FP for Random Forest :', fp_rf)
#print('FN for Random Forest :', fn_rf)

from sklearn.metrics import confusion_matrix
tn_rf1, fp_rf1, fn_rf1, tp_rf1 = confusion_matrix(df.y_act, df.y_pred_rf).ravel()

#print('TP for Random Forest :', tp_rf1)
#print('TN for Random Forest :', tn_rf1)
#print('FP for Random Forest :', fp_rf1)
#print('FN for Random Forest :', fn_rf1)

def compute_accuracy(tp, tn, fn, fp):
	'''
	Accuracy = TP + TN / FP + FN + TP + TN

	'''
	return ((tp + tn) * 100)/ float( tp + tn + fn + fp)

print('Accuracy for Logistic Regression :', compute_accuracy(tp_lr, tn_lr, fn_lr, fp_lr))
print('Accuracy for Random Forest :', compute_accuracy(tp_rf, tn_rf, fn_rf, fp_rf))


def compute_precision(tp, fp):
	'''
	Precision = TP  / FP + TP 

	'''
	return (tp  * 100)/ float( tp + fp)

print('Precision for Logistic Regression :', compute_precision(tp_lr, fp_lr))
print('Precision for Random Forest :', compute_precision(tp_rf, fp_rf))


def compute_recall(tp, fn):
	'''
	Recall = TP /FN + TP 

	'''
	return (tp  * 100)/ float( tp + fn)

print('Recall for Logistic Regression :', compute_recall(tp_lr, fn_lr))
print('Recall for Random Forest :', compute_recall(tp_rf, fn_rf))

