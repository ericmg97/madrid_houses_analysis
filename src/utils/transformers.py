from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class RentPriceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Apply the transformation to the 'rent_price' column
        X_copy = X.copy()
        X_copy['rent_price'] = X_copy['rent_price'].apply(lambda x: x if 0 <= x <= 2500 else np.nan)
        return X_copy
    
    def get_feature_names_in(self):
        return ['rent_price']
    
    def get_feature_names_out(self, input_features=None):
        return ['rent_price']

# Custom transformer for handling missing values using the mode
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
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['district_id'] =  X_copy['neighborhood_id'].str.extract(r'District (\d+)')
        X_copy['neighborhood_id'] = X_copy['neighborhood_id'].str.extract(r'\(([\d.]+) €/m2\)').astype('float')
        return X_copy
    
    def get_feature_names_in(self):
        return ['neighborhood_id']
    
    def get_feature_names_out(self, input_features=None):
        return ['neighborhood_mean_price', 'district_id']
    