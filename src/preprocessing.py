import pandas as pd
from typing import Optional, Union
import re

def load_data(file: Union[str, pd.io.common.ReadCsvBuffer],
              date_col: str = "date",
              amount_col: str = "amount",
              category_col: Optional[str] = "category") -> pd.DataFrame:
    """
    Load CSV, clean dates, amounts, and categories.
    Handles malformed numbers like '.-1000' or '(1000)'.
    """
    df = pd.read_csv(file)

    # parse date
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    # clean amount column
    def parse_amount(x):
        x = str(x).strip()
        # convert parentheses to negative
        x = re.sub(r"\(([\d\.]+)\)", r"-\1", x)
        # remove all non-digit, non-dot, non-minus chars
        x = re.sub(r"[^\d\.-]", "", x)
        # fix cases like '.-1000' => '-1000'
        x = re.sub(r"^\.\-", "-", x)
        try:
            return float(x)
        except:
            return 0.0

    df[amount_col] = df[amount_col].apply(parse_amount)
    df[amount_col] = df[amount_col].clip(upper=1_000_000)

    # category cleanup
    if category_col and category_col in df.columns:
        df[category_col] = df[category_col].astype(str).str.strip().str.lower()
    else:
        df['category'] = 'uncategorized'
        category_col = 'category'

    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def create_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df['weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)
    return df
