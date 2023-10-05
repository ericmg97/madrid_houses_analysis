from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, columns, threshold=1.5):
        self.columns = columns
        self.threshold = threshold

    def fit(self, X, y=None):
        self.iqr = {}
        self.q1 = {}
        self.q3 = {}
        for column in self.columns:
            self.q1[column] = X[column].quantile(0.25)
            self.q3[column] = X[column].quantile(0.75)
            self.iqr[column] = self.q3[column] - self.q1[column]
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for column in self.columns:
            if column == 'rent_price':
                X_copy['rent_price'] = X_copy['rent_price'].apply(lambda x: x if 0 <= x <= 2500 else np.nan)
            else:
                X_copy[column] = X_copy[column].apply(lambda x: x if x >= self.q1[column] - self.threshold * self.iqr[column] and x <= self.q3[column] + self.threshold * self.iqr[column] else np.nan)
        
        return X_copy
    
    def get_feature_names_in(self):
        return self.columns
    
    def get_feature_names_out(self, input_features=None):
        return self.columns

class FillNA(BaseEstimator, TransformerMixin):
    def __init__(self, columns, type_fill='mode'):
        self.type_fill = type_fill
        self.columns = columns

    def fit(self, X, y=None):
        if self.type_fill == 'mode':
            self.values = {}
            for column in self.columns:
                self.values[column] = X[column].mode()[0]
        elif self.type_fill == 'mean':
            self.values = {}
            for column in self.columns:
                self.values[column] = X[column].mean()
        elif self.type_fill == 'False':
            self.values = {}
            for column in self.columns:
                self.values[column] = False
        elif self.type_fill == 'True':
            self.values = {}
            for column in self.columns:
                self.values[column] = True
        
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column in self.columns:
            X_copy[column].fillna(self.values[column], inplace=True)
            
        return X_copy
    
    def get_feature_names_in(self):
        return self.columns
    
    def get_feature_names_out(self, input_features=None):
        return self.columns


class ExtractDataNeighborhood(BaseEstimator, TransformerMixin):
    def __init__(self, data = None):
        self.data = data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.data is 'categorical':
            X_copy['district_id'] =  X_copy['neighborhood_id'].str.extract(r'District (\d+)').astype('int')
            X_copy['neighborhood_id'] = X_copy['neighborhood_id'].str.extract(r'Neighborhood (\d+)').astype('int')
        elif self.data is 'numerical':
            X_copy['neighborhood_id'] = X_copy['neighborhood_id'].str.extract(r'\(([\d.]+) €/m2\)').astype('float')
        else:
            X_copy['district_id'] =  X_copy['neighborhood_id'].str.extract(r'District (\d+)').astype('int')
            X_copy['neighborhood_id'] = X_copy['neighborhood_id'].str.extract(r'\(([\d.]+) €/m2\)').astype('float')

        return X_copy
    
    def get_feature_names_in(self):
        return ['neighborhood_id']
    
    def get_feature_names_out(self, input_features=None):
        if self.data == 'categorical':
            return ['neighborhood_id', 'district_id']
        elif self.data == 'numerical':
            return ['neighborhood_mean_price']
        else:  
            return ['neighborhood_mean_price', 'district_id']
    