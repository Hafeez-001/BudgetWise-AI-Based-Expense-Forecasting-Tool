import pandas as pd
from src.preprocessing import load_data, create_time_features

def test_load_and_time_features(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text("date,amount,category\n2023-01-01,100,food\n2023-01-02,200,transport\n")
    df = load_data(str(csv))
    assert 'date' in df.columns and 'amount' in df.columns
    df2 = create_time_features(df)
    assert 'month' in df2.columns
