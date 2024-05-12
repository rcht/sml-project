import pandas as pd

df = pd.read_csv('heart_attack_prediction_dataset.csv')
data_headers = df.columns.to_list()[1:]

df1 = pd.read_csv('heart_disease_health_indicators.csv')
data_headers1 = df1.columns.to_list()

if __name__ == '__main__':
    print(data_headers1)
