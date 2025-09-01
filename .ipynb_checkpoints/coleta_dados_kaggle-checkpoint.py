import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "creditcard.csv"  # Certifique-se de não ter espaços extras

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "mlg-ulb/creditcardfraud",
    file_path
)

print(df.head())
print(df.shape)
