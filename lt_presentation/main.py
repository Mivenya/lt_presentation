# Main File to run project. Aggregate data from Kaggle ML and DS Surveys from 2017 - 2022 to analyze trends for lightning talk presentation purposes

# The purpose of the lightning talk is to show trends of growth in ML and DS.
# Based on finding possibly highlight not just number of people but gender/location etc?

#Begin with importing Libraries
from lt_presentation import analyzer

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

# Next we will be pulling data from multiple datasets but want to narrow down what info we want.  
# I know showing number of responses, gender, locations, employed status could all be useful for showing trends

def main(): 
#read data
    path = 'lt_presentationt/multipleChoiceResponses2017.csv'
    df = analyzer.read_dataset(dataset_path=path)

#data exploration and preprocessing
    my_data_manipulation = analyzer.DataManipulation(df=df)

#describe dataset
    #analyzer.describe(dataset=df)
main()