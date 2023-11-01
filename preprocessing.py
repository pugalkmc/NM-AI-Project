import pandas as pd

input_file = 'ai_dataset.xlsx'
output_file = 'preprocessed_data.xlsx'

df = pd.read_excel(input_file)

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')

df = pd.get_dummies(df, columns=['Country'], prefix='Country')

df.to_excel(output_file, index=False)

print("Preprocessing complete. Preprocessed data saved to", output_file)
