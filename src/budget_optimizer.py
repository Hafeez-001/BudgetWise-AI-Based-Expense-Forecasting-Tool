import pandas as pd

def recommend_budget(cat_forecast_df: pd.DataFrame,
                     periods: int,
                     income: float,
                     savings_target_pct: float = 0.2,
                     category_col: str = 'category',
                     amount_col: str = 'yhat') -> dict:
    """
    Recommend budget per category, scaled to actual income and savings target.
    """
    total_forecast = cat_forecast_df[amount_col].sum()
    budget_available = income * (1 - savings_target_pct)
    scale = budget_available / max(total_forecast, 1e-6)

    cat_sum = cat_forecast_df.groupby(category_col)[amount_col].sum() * scale
    recommended_by_category = cat_sum.to_dict()

    return {
        'recommended_by_category': recommended_by_category,
        'total_forecast': total_forecast,
        'budget_available': budget_available,
        'scale': scale
    }
