class Imputer:
    def fit(self, X, y=None):
        ignore_columns = ['start_cluster']
        self.string_fill_value_ = X.drop(columns=ignore_columns).select_dtypes(exclude='number').mode().iloc[0]
        self.number_fill_value_ = X.select_dtypes(include='number').median()
    
    def transform(self, X, y=None):
        X = X.copy()
        X = X.fillna(self.string_fill_value_)
        X = X.fillna(self.number_fill_value_)
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    

class FeatureExtractor:
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        pass
    
    def transform(self, X, y=None):
        return X[self.columns]
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    