import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict

class ensembler():
    
    train = None
    test = None
    predictors = None
    target_column = None
    base_model_list = None
    top_model = None
    predict_prob = None
    nfold = None
    seed = None
    
    
    def __init__(self, train, test, predictors, target_column, base_model_list, top_model, predict_prob, nfold, seed = 321):
        
        self.train = train
        self.test = test
        self.ntrain = self.train.shape[0]
        self.ntest = self.test.shape[0]
        self.target_column = target_column
        self.predictors = predictors
        self.base_model_list = base_model_list
        self.top_model = top_model
        self.predict_prob = predict_prob
        self.nfold = nfold
        self.seed = seed
        
    @staticmethod
    def get_oof(model, x_train, y_train, x_test, nfold, predict_prob):
        
        oof_train = np.zeros((x_train.shape[0],))
        oof_test = np.zeros((x_test.shape[0],))

        
        if predict_prob == True:
            oof_train = cross_val_predict(model, x_train, y_train, cv=nfold, method='predict_proba')[:,1]
            
            model.fit(x_train, y_train)
            oof_test = model.predict_proba(x_test)[:,1]
            
        else:
            oof_train = cross_val_predict(model, x_train, y_train, cv=nfold)
            
            model.fit(x_train, y_train)
            oof_test = model.predict(x_test)

        
        return oof_train, oof_test

    
    def train_base_models(self):
        
        self.oof_train = []
        self.oof_test = []

        for i in range(0,len(self.base_model_list)):        
            a, b = self.get_oof(self.base_model_list[i],
                                self.train[self.predictors],
                                self.train[self.target_column],
                                self.test[self.predictors],
                                self.nfold,
                                predict_prob = self.predict_prob
                               )
        
            self.oof_train.append(a)
            self.oof_test.append(b)
            print(a)
        
        self.base_predictions_train_df = pd.DataFrame(self.oof_train).T
        self.base_predictions_test_df = pd.DataFrame(self.oof_test).T
    
    def train_top_model(self):
        self.top_model.fit(self.base_predictions_train_df, self.train[self.target_column])
    
    def predictions(self):
        if self.predict_prob == True:
            return self.top_model.predict_proba(self.base_predictions_test_df)
        else:
            return self.top_model.predict(self.base_predictions_test_df)
