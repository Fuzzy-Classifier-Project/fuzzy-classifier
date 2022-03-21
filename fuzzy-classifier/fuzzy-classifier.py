import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging as log
import random


@dataclass
class norm_params:
    shift_param:np.array()
    scale_param:np.array()

class FuzzyClassifier:
    """Fuzzy logic system for classification problems using type-1 and type-2 approaches.

    The fuzzy rules are defined according to the data.

    """
    def __init__(
            self,
            epochs=None,
            batch_size=16,
            step_size=1e-3,
            n_rules=2,
            normalization=True,
            normalization_type='StandardScaler',
            validation=True,
            validation_percentage=0.15,
            validation_checks=20,
        ):
    
            # Name
            self.name = "FuzzyClassifier"

            # Normalization
            self.normalization = normalization
            self.normalization_type = normalization_type
            '''
            StandardScaler - Standardize features by removing the mean and scaling to unit variance.
            MinMaxScaler - Rescales the data set such that all feature values are in the range [0, 1].
            '''

            # Missing Data (Future)

            # Training Settings
            self.epochs = epochs
            self.n_rules = n_rules
            self.batch_size = batch_size
            self.step_size = step_size

            # Validation Settings
            self.validation = validation
            self.validation_percentage = validation_percentage
            self.validation_checks = validation_checks

    def get_normalization_params(self,X):
        if self.normalization_type is None or self.normalization is not True:
            log.debug('No normalization is used')
            shift,scale = 1.0,0.0
            norm_params=(shift,scale)
        elif self.normalization_type == 'StandardScaler':
            log.debug('StandardScaler is used')
            shift = X.mean(axis=0)
            scale = X.std(axis=0)
            norm_params=(shift,scale)
        elif self.normalization_type == 'MinMaxScaler': #Not exactly the same as scikit
            log.debug('MinMaxScaler is used')
            shift = X.min(axis=0)
            scale = X.max(axis=0)-X.min(axis=0)
            norm_params=(shift,scale)
        elif self.normalization_type == 'MaxAbsScaler': #Not exactly the same as scikit
            log.debug('MinMaxScaler is used')
            shift = (X.min(axis=0) +  X.max(axis=0))/2
            scale =(X.max(axis=0) - X.min(axis=0))/2
            norm_params=(shift,scale)
        return norm_params
    
    def normalize(self,X,norm_params):
        X_norm = (X - norm_params.shift)/norm_params.scale
        return X_norm
    
    def separate_classes(self,X,y):
        classes=np.unique(y)
        data_dict={}
        for i in range(0,len(classes)):
            data_dict[i]=X[np.where(y==i)[0],:]
        return data_dict
    

    def create_initial_weights(self,X,y,n_rules):
        data_dict=self.separate_classes(X,y)
        n_class=len(data_dict)
        n_feats=np.shape(data_dict[0])[1]
        # Mean and std initial values
        data_dict_mean = {}
        data_dict_var={}
        for i in data_dict:
            if np.shape(data_dict[i])[0]==1:
                data_dict_mean[i]=data_dict[i]
                data_dict_var[i]=np.ones((1,n_feats))
            else:
                data_dict_mean[i]=data_dict[i].mean(axis=0)
                data_dict_var[i]=data_dict[i].var(axis=0)
            X_mean =np.array(pd.DataFrame(data_dict_mean).T)
            X_var =np.array(pd.DataFrame(data_dict_var).T)
        mat_mean0=np.ones((n_rules*n_class,n_feats))
        mat_std0=np.ones((n_rules*n_class,n_feats))
        theta0=np.ones((n_class,n_class*n_rules))*(-1)
        cont=0
        for i in range(0,n_class):
            theta0[i,i*2:i*2+n_rules]=theta0[i,i*2:i*2+n_rules]*(-1)
            for j in range(0,n_rules):
                if j==0:
                    mat_mean0[cont,:]=X_mean[i,:]
                    mat_std0[cont,:]=X_var[i,:]
                else:
                    mat_mean0[cont,:]=X_mean[i,:]*random.uniform(-1,1)+X_mean[i,:]
                    mat_std0[cont,:]=X_var[i,:]*random.uniform(-1,1)+X_var[i,:]
                cont=cont+1
        return mat_mean0, mat_std0, theta0    


    
    # def fit(self, df, validation_df, verbose):

    


def convert(my_name):
    """
    Print a line about converting a notebook.
    Args:
        my_name (str): person's name
    Returns:
        None
    """

    print(f"I'll convert a notebook for you some day, {my_name}.")

convert('Mateus')