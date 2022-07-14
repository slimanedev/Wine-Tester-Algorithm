"""
Team:
<<<<< DEDALE >>>>>
Authors:
<<<<< Katia Jodogne-del Litto - 2160101 >>>>>
<<<<< Slimane Aglagal - 2103355 >>>>>
"""
"""
Copyright (C) 2021, Katia Jodogne-del Litto and Slimane Aglagal
"""
from wine_testers import WineTester
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

class MyWineTester(WineTester):
    def __init__(self):
        #We initiate the Random Forest Classifier
        #The choice of parameters for this model is justified in the report
        self.model=RandomForestClassifier(criterion = 'entropy', 
                                            max_depth = 20,
                                            class_weight={6:0.047,5:0.032,7:0.16,4:0.34,8:0.42}, 
                                            min_samples_leaf = 1, 
                                            n_estimators = 250,
                                            bootstrap=True,
                                            random_state=42)
        #Scaler to normalize and standardize the data
        self.standard_scaler = preprocessing.StandardScaler()
        

    def train(self, X_train, y_train):
        """
        train the current model on train_data
        :param X_train: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :param y_train: 2D array of labels.
                each line is a different example.
                the first column is the example ID.
                the second column is the example label.
        """
        #We delete the classes 3 and 9 from the training data
        x_data_train,y_data_train=self.delete_classes(X_train,y_train)
        #Delete the index column
        y_train=[z[1] for z in y_data_train]
        ##We process the data before training the model
        x_train_scaled=self.preprocessing_attributs(x_data_train)
        
        #Training the model
        self.model.fit(x_train_scaled, y_train)
        

    def predict(self, X_data):
        """
        predict the labels of the test_data with the current model
        and return a list of predictions of this form:
        [
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            ...
        ]
        :param X_data: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :return: a 2D list of predictions with 2 columns: ID and prediction
        """
        #We process the testing data before putting it in the model
        x_test_scaled =self.preprocessing_attributs(X_data)
        #We use the trained model to predict new labels for the test data
        rf_predictions = self.model.predict(x_test_scaled)

        #We prepare the output
        prediction=[]
        for index in range(len(rf_predictions)):
            prediction.append([int(X_data[index][0]),rf_predictions[index]])
        return prediction
    
    def preprocessing_attributs(self,X):
        #This function is used to convert categorical values and scale data
        #Delete the index column
        X_data=[z[1:] for z in X]
        #Replace "white" by 1 and "red" by 0
        for idx, row in enumerate(X_data):
            if X_data[idx][0]=="white":
                X_data[idx][0]=1
            else: X_data[idx][0]=0
        #Scaling
        x_scaled = self.standard_scaler.fit_transform(X_data)
        return x_scaled
    
    def delete_classes(self,X,y):
        #This function is used to delete classes 3 and 9
        #The justification is detailed in the report
        for idx, row in enumerate(X):
            if y[idx][1]==9:
                X.remove(X[idx])
                y.remove(y[idx])
            elif y[idx][1]==3:
                X.remove(X[idx])
                y.remove(y[idx])
        return X,y
