from typing import Dict, Any, List
import pandas as pd

class RawFeatureMapping:
    def __init__(self, rules: Dict[Any, Dict[Any, Any]], feature_names: List[Any]):
        """
        实现非数值特征到数值特征的映射与反映射
        """
        self.rules = rules
        self.feature_names = feature_names
        inverse_rules = {}
        for name in rules:
            inverse_rules[name] = {v: k for k, v in rules[name].items()}
        self.inverse_rules = inverse_rules


    def mapping(self, X: pd.DataFrame) -> pd.DataFrame:
        for feature in self.feature_names:
            X[feature] = X[feature].map(self.rules[feature].get)
        return X


    def inverse_mapping(self, X: pd.DataFrame) -> pd.DataFrame:
        for feature in self.feature_names:
            X[feature] = X[feature].map(self.inverse_rules[feature].get)
        return X



if __name__ == "__main__":
    feature_mapping = {
        'age': {
            1: "young",
            2: "pre-presbyopic",
            3: "presbyopic"
        },
        'prescription': {
            1: "myope",
            2: "hypermetrope"
        },
        'astigmatic': {
            1: "no",
            2: "yes"
        },
        'tear_production': {
            1: "reduced",
            2: "normal"
        }
    }
    new_feature_mapping = {}
    for name in feature_mapping:
        new_feature_mapping[name] = {v:k for k, v in feature_mapping[name].items()}
    fm = RawFeatureMapping(new_feature_mapping, list(feature_mapping.keys()))
    path = "../data/lenses/lenses.text"
    X = pd.read_csv(path)
    print(X.head(5))
    X_map = fm.mapping(X)
    print(X.head(5))
    print(X_map.head(5))
