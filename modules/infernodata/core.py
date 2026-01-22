import pandas as pd

def preprocess_data(data: list):
    df = pd.DataFrame(data)
    df = df.dropna()
    return {
        "rows": df.shape[0],
        "columns": list(df.columns),
        "preview": df.head().to_dict()
    }
