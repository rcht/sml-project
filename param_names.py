import pandas as pd

df = pd.read_csv('heart_attack_prediction_dataset.csv')

data_headers = df.columns.to_list()[1:]

if __name__ == '__main__':
    print(data_headers)
