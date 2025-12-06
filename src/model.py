# src/model.py
from xgboost import XGBClassifier
from data_pipeline import preprosses
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def build_model_pipeline():
    pipeline = Pipeline([
        ('preprocessor', preprosses()),
        ('model', XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            colsample_bytree=0.8,
            gamma=0.2,
            learning_rate=0.01,
            max_depth=6,           
            n_estimators=300,      
            subsample=0.8,
            reg_alpha=0.5,
            reg_lambda=1
        ))
    ])
    return pipeline
