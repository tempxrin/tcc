import kagglehub
import pandas as pd
import shutil
import os

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
shutil.copy2(os.path.join(path, "creditcard.csv"), r'C:\Users\dados1\Documents\tcc\tcc\dados_cartao_credito.csv')

df = pd.read_csv(r'C:\Users\dados1\Documents\tcc\tcc\dados_cartao_credito.csv')