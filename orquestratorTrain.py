'''
Created on Feb 14, 2020

@author: YANI STRATEV
'''
from train import train
import gc
from parameters import  date_now,LOSS,CELL,N_STEPS,NUM_LAYERS,UNITS,ticker, LOOKUP_STEP,COLUMNAS
import tensorflow as tf
import os

for step in range(1,LOOKUP_STEP):
      # Creamos las carpetas por si no existen 
    
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    
    
    if not os.path.isdir("results"):
        os.mkdir("results")
    
    if not os.path.isdir("data"):
        os.mkdir("data")

    model_name = "{now}_{ticker_name}-{error_loss}-{cell_name}-seq-{sequence_lenght}-step-{step}-layers-{layers}-units-{neurons}".format(
        now=date_now,
        ticker_name=ticker,
        error_loss=LOSS,
        cell_name=CELL.__name__,
        sequence_lenght=N_STEPS,
        step=step,
        layers=NUM_LAYERS,
        neurons=UNITS
    )
    model_name += "b"
    gc.collect()
    train(step, model_name)
    gc.collect()
    