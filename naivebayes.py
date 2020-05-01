
import sys
import json
from sklearn.externals import joblib
import numpy as np
import pickle

#Retrieval
import abc
from abc import abstractmethod

class Retrieval:
    __metaclass__ = abc.ABCMeta
    @classmethod
    def __init__(self): #constructor for the abstract class
        pass

    @classmethod
    def getLongSnippets(self, question):
        longSnippets = question['contexts']['long_snippets']
        fullLongSnippets = ' '.join(longSnippets)
        return fullLongSnippets


    @classmethod
    def getShortSnippets(self, question):
        shortSnippets = question['contexts']['short_snippets']
        fullShortSnippets = ' '.join(shortSnippets)
        return fullShortSnippets

#Featurization
import abc
from abc import abstractmethod

class Featurizer:
    __metaclass__ = abc.ABCMeta
    @classmethod
    def __init__(self): #constructor for the abstract class
        pass

    #This is the abstract method that is implemented by the subclasses.
    @abstractmethod
    def getFeatureRepresentation(self, X_train, X_val):
        pass

#TFIDF
#from Featurizer import Featurizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


#This is a subclass that extends the abstract class Featurizer.
class TfidfFeaturizer(Featurizer):

    #The abstract method from the base class is implemeted here to return count features
    def getFeatureRepresentation(self, X_train, X_val):
        tfidf_vect = TfidfVectorizer(smooth_idf=True,min_df=20,max_df=2000)
        X_train_counts = tfidf_vect.fit_transform(X_train)
        X_val_counts = tfidf_vect.transform(X_val)
        return X_train_counts, X_val_counts

#Count Featurizer
#from Featurizer import Featurizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


#This is a subclass that extends the abstract class Featurizer.
class CountFeaturizer(Featurizer):

    #The abstract method from the base class is implemeted here to return count features
    ### got information about the CountVectorizer from Stack overflow answers - https://stackoverflow.com/questions/27697766/understanding-min-df-and-max-df-in-scikit-countvectorizer
    def getFeatureRepresentation(self, X_train, X_val):
        count_vect = CountVectorizer(min_df=20)
        X_train_counts = count_vect.fit_transform(X_train)
        X_val_counts = count_vect.transform(X_val)
        return X_train_counts, X_val_counts

#Classifier

import abc
from abc import abstractmethod

class Classifier:
    __metaclass__ = abc.ABCMeta
    @classmethod
    def __init__(self): #constructor for the abstract class
        pass

    #This is the abstract method that is implemented by the subclasses.
    @abstractmethod
    def buildClassifier(self, X_features, Y_train):
        pass

#from Classifier import Classifier
from sklearn.naive_bayes import MultinomialNB


#This is a subclass that extends the abstract class Classifier.
class MultinomialNaiveBayes(Classifier):

    #The abstract method from the base class is implemeted here to return multinomial naive bayes classifier
    def buildClassifier(self, X_features, Y_train):
        clf = MultinomialNB().fit(X_features, Y_train)
        return clf

from sklearn.neural_network import MLPClassifier

class MLP(Classifier):
    def buildClassifier(self,X_features,Y_features):
        clf=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 10), random_state=1).fit(X_features,Y_features)
        return clf

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

class SVM(Classifier):
    def buildClassifier(self,X_features,Y_features):
        clf=svm.LinearSVC(C=0.004).fit(X_features,Y_features)
        return clf

#Evaluation
import abc
from abc import abstractmethod
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

class Evaluator:
    __metaclass__ = abc.ABCMeta
    @classmethod
    def __init__(self): #constructor for the abstract class
        pass

    #This is a class method that gets accuracy of the model
    @classmethod
    def getAccuracy(self, Y_true, Y_pred):
        accuracy = accuracy_score(Y_true, Y_pred)
        return accuracy
    
    #This is a class method that gets precision, recall and f-measure of the model  
    @classmethod
    def getPRF(self, Y_true, Y_pred):
        prf = precision_recall_fscore_support(Y_true, Y_pred, average='weighted')
        precision = prf[0]
        recall = prf[1]
        f_measure = prf[2]
        return precision, recall, f_measure

#PipelineQA
class Pipeline(object):
    def __init__(self, trainFilePath, valFilePath, retrievalInstance, featurizerInstance, classifierInstance):
        self.retrievalInstance = retrievalInstance
        self.featurizerInstance = featurizerInstance
        self.classifierInstance = classifierInstance
        trainfile = open(trainFilePath, 'r')
        self.trainData = json.load(trainfile)
        trainfile.close()
        valfile = open(valFilePath, 'r')
        self.valData = json.load(valfile)
        valfile.close()
        self.question_answering()

    def makeXY(self, dataQuestions):
        X = []
        Y = []
        for question in dataQuestions:
            
            long_snippets = self.retrievalInstance.getLongSnippets(question)
            short_snippets = self.retrievalInstance.getShortSnippets(question)
            
            X.append(short_snippets)
            Y.append(question['answers'][0])
            
        return X, Y


    def question_answering(self):
        dataset_type = self.trainData['origin']
        candidate_answers = self.trainData['candidates']
        X_train, Y_train = self.makeXY(self.trainData['questions'])
        X_val, Y_val_true = self.makeXY(self.valData['questions'])
        with open('/content/drive/My Drive/NLPA_Project/val_true.txt','wb') as f:
            pickle.dump(Y_val_true,f)

        #featurization
        X_features_train, X_features_val = self.featurizerInstance.getFeatureRepresentation(X_train, X_val)
        print(np.shape(X_features_train), np.shape(X_features_val))
        self.clf = self.classifierInstance.buildClassifier(X_features_train, Y_train)
        
        #Prediction
        Y_val_pred = self.clf.predict(X_features_val)
        
        print(Y_val_pred)
        with open('/content/drive/My Drive/NLPA_Project/val_predict.txt','wb') as f:
            pickle.dump(Y_val_pred,f)

        self.evaluatorInstance = Evaluator()
        a =  self.evaluatorInstance.getAccuracy(Y_val_true, Y_val_pred)
        p,r,f = self.evaluatorInstance.getPRF(Y_val_true, Y_val_pred)
        print("Accuracy: " + str(a))
        print("Precision: " + str(p))
        print("Recall: " + str(r))
        print("F-measure: " + str(f))
  

if __name__ == '__main__':
    trainFilePath ='./data/train_formatted.json' 
    valFilePath = './data/test_formatted.json' 
    retrievalInstance = Retrieval()
  #classifierInstance = MultinomialNaiveBayes()
    featurizerInstance = CountFeaturizer()
  #featurizerInstance = TfidfFeaturizer()()
    classifierInstance = MultinomialNaiveBayes()
    trainInstance = Pipeline(trainFilePath, valFilePath, retrievalInstance, featurizerInstance, classifierInstance)


