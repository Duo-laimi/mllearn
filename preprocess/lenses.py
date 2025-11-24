import pandas as pd

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

def process_features_with_pandas(file_path):
    """
    使用Pandas处理特征文件的主函数

    Args:
        file_path (str): 特征数据文件路径
    """
    # 读取数据文件
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, index_col=0)
    column_names = ['age', 'prescription', 'astigmatic', 'tear_production', 'label']
    df.columns = column_names

    for name in column_names[:-1]:
        df[name] = df[name].map(feature_mapping[name].get)
    return df


if __name__ == "__main__":
    path = "../data/lenses/lenses.data"
    df = process_features_with_pandas(path)
    print(df.head())
    df.to_csv("../data/lenses/lenses.text", index=False)

