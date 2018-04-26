###############################################################
#    Author : Remeshkumar K K
#    Date : 04/19/2018
#    Code snippet for ML training and prediction
###############################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import logging
import pickle

logging.basicConfig(filename="results.log", level=logging.INFO)
model = None

# print model.score(X_test,y_test)
model = None
def classifyImage(model):

    unknown = pd.read_csv("predict.csv")
    y_pred = unknown.iloc[:, 1:]
    y_out = model.predict(y_pred)
    print "Prediction is " , y_out
    logging.info("Prediction : "+y_out)


def executeMachine():
    logging.info("Inside executeMachine()")
    data = pd.read_csv("images.csv")
        # X = data.iloc[:, 1:]
        # y = data['label']

    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = svm.SVC(kernel="linear")
    model.fit(X_train, y_train)
    return model

if __name__=="__main__":

    mlModelObj = executeMachine()

    pickleFilename = "MLPickle"
    fileObject = open(pickleFilename, 'wb')
    pickle.dump(mlModelObj,fileObject)
    fileObject.close()