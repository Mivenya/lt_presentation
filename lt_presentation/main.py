# Main File to run project. Aggregate data from Kaggle ML and DS Surveys from 2017 - 2022 to analyze trends for lightning talk presentation purposes

# The purpose of the lightning talk is to show trends of growth in ML and DS.
# Based on finding possibly highlight not just number of people but gender/location etc?

#Begin with importing Libraries

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

import os
import sys
import pandas as pd
import numpy as np
from random import sample
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Next we will be pulling data from multiple datasets but want to narrow down what info we want.  
# I know showing number of responses, gender, locations, employed status could all be useful for showing trends

# Read dataset = function
def read_dataset(dataset_path: str) -> pd.DataFrame: 
    """ This function is to read the dataset
    """
    dataset = pd.read_csv(filepath_or_buffer=dataset_path, encoding='latin-1', low_memory=False)
    return dataset

# Describe dataset = function
def describe(dataset: pd.DataFrame): 
    """ This function is to describe the dataset
    """
    print(dataset.describe()) 
    
# Class 1 = Data Manipulation class, any below are members

class DataManipulation:
    def __init__(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df_raw = df
        self.df = df.copy()

    
    def drop_missing_data(self, df: pd.DataFrame) -> pd.DataFrame: 
        """Drops rows with missing values in the specified column.

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: pd.DataFrame: The DataFrame with missing values dropped.
        """
        self.df = self.df.dropna()
        return self.df
    
    def fill_missing_data(self, df: pd.DataFrame) -> pd.DataFrame: 
        """Drops rows with missing values in the specified column.

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: pd.DataFrame: The DataFrame with missing values dropped.
        """
        self.df = self.df.fillna(0)
        return self.df
    
    def drop_column(self, column_names: str) -> pd.DataFrame:
        """Drop entire specified column
    
        Args:
        column_name (str): The name of the column to completely drop.
        
        Returns: pd.DataFrame: The DataFrame with the selected column removed
        """
        self.df = self.df.drop([column_names], axis=1)
        return self.df
    
    def encode_features(self, column_names: str) -> pd.DataFrame:
        """One hot encodes features of the specified column.

        Args:
            column_name (str): The name of the column to encode values from.

        Returns: pd.DataFrame: The DataFrame with features encoded.
        """
        enc = OneHotEncoder()
        self.df_onehot = pd.get_dummies(self.df[[column_names]], dtype=int)
        self.df = pd.concat([self.df, self.df_onehot], axis=1) 
        self.df = self.df.drop([column_names], axis=1)
        return self.df
    
    def encode_label(self, column_names: str) -> pd.DataFrame:
        """Label encodes in the specified column.

        Args:
            column_name (str): The name of the column to label encode values from.

        Returns: pd.DataFrame: The DataFrame with label encoded.
        """
        lbl_enc = LabelEncoder()
        self.df.loc[:, column_names] = lbl_enc.fit_transform(self.df[column_names])
        return self.df
    
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data in the dataset.

        Args:
            df (pd.DataFrame): The name of the dataset to standardize.

        Returns: pd.DataFrame: The DataFrame with standardization.
        """
        scaler = StandardScaler().set_output(transform="pandas")
        self.df = scaler.fit_transform(self.df)
        return self.df

    def shuffle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shuffle the dataset

        Args:
            df (pd.DataFrame): The name of the dataset to shuffle.

        Returns: pd.DataFrame: The DataFrame shuffled.
        """
        shuffled_df = self.df.sample(n=len(self.df))
        return shuffled_df
    
    def sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get 50% sample (0.5 frac) of the dataset

        Args:
            df (pd.DataFrame): The name of the dataset to shuffle.

        Returns: 50% of the pd.DataFrame:
        """
        self.df = self.df.sample(frac=0.5, replace=True, random_state=1)
        return self.df
    
    def retrieve_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retrieve data at it's current

        Args:
            df (pd.DataFrame): The name of the dataset to retrieve.

        retrieved current (transformed) verion of pd.DataFrame:
        """
        return self.df
    
def main(): 
#read data
    path = 'lt_presentation/lt_presentation/multipleChoiceResponses2017.csv'
    df = read_dataset(dataset_path=path)

#data exploration and preprocessing
    my_data_manipulation = DataManipulation(df=df)

#describe dataset
    #describe(dataset=df)

#Begin Data Manipulation

# Drop Missing Data
    data_df = my_data_manipulation.fill_missing_data(df)
    print(data_df)
    #print(df)
main()