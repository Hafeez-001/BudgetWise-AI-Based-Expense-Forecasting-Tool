import streamlit as st
import pandas as pd
import plotly.express as px

from preprocessing import load_data, create_time_features
from models import forecast_prophet, forecast_random_forest, forecast_xgboost
from forecasting import evaluate_forecast
from budget_optimizer import recommend_budget

# ----------------------------------------------------
# Streamlit page setup
# ----------------------------------------------------
# st.set_page_config(layout="wide", page_title="💰 BudgetWise — Expense Forecaster")
# st.title("💰 BudgetWise — Expense Forecaster (MVP)")
import streamlit as st

st.set_page_config(
    page_title="BudgetWise X — Personal Forecast",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for subtle brand colors
st.markdown("""
<style>
/* larger heading */
[data-testid="stHeader"] {display:none;}
h1 {font-size:32px}
.app-logo {height:48px;border-radius:8px}
.sidebar .css-1d391kg {background-color: #f8f9fb}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1,6])
with col1:
    st.image(r"C:\Users\moham\Downloads\BudgetWise-AI-Expense-Forecasting-main\BudgetWise-AI-Expense-Forecasting-main\src\assets\logo.jpeg", width=200, use_container_width=False)  # add your logo file
with col2:
    st.markdown("# BudgetWise - An AI based Budget Forecaster")
    st.caption("Your personal expense forecaster — by Hafeez")
st.markdown("""
    <style>
    h1, h2, h3 { color: #0ea5a4 !important; font-family: 'Poppins', sans-serif; }
    .stButton button { background-color: #4f46e5; color: white; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
st.divider()

# ----------------------------------------------------
# Tabs for navigation
# ----------------------------------------------------
tab_upload, tab_eda, tab_forecast = st.tabs(["📂 Upload Data", "📊 EDA & Visualization", "🔮 Forecast & Budget"])

# ----------------------------------------------------
# Tab 1: File Upload
# ----------------------------------------------------
with tab_upload:
    uploaded = st.file_uploader("Upload your expenses CSV", type=["csv"])
    if uploaded:
        df = load_data(uploaded)
        df = df.dropna(subset=['category'])
        df = df[df['category'].astype(str).str.lower() != 'nan']
        st.write("### Raw data sample", df.head())

# ----------------------------------------------------
# # Tab 2: EDA & Visualization
# # ----------------------------------------------------
# with tab_eda:
#     if 'df' in locals():
#         # Category normalization
#         category_map = {
#             'food': 'food', 'fod': 'food', 'foodd': 'food', 'foods': 'food',
#             'education': 'education', 'edu': 'education', 'educaton': 'education',
#             'entertainment': 'entertainment', 'entertain': 'entertainment', 'entrtnmnt': 'entertainment',
#             'rent': 'rent', 'rentt': 'rent', 'rnt': 'rent',
#             'utilities': 'utilities', 'utility': 'utilities', 'utlities': 'utilities', 'utilties': 'utilities',
#             'travel': 'travel', 'traval': 'travel', 'travl': 'travel',
#             'health': 'health', 'helth': 'health',
#             'saving': 'savings', 'savings': 'savings', 'salary': 'income', 'bonus': 'income',
#             'misc': 'misc', 'other': 'misc', 'others': 'misc'
#         }
#         df['category_clean'] = df['category'].map(category_map).fillna(df['category'])

#         df_tf = create_time_features(df)

#         # Category spending summary
#         cat_counts = (
#             df_tf.groupby('category_clean')['amount']
#             .sum()
#             .reset_index()
#             .sort_values('amount', ascending=False)
#         )
#         st.write("#### Spending by category (total)")
#         st.dataframe(cat_counts)

#         # Pie chart
#         fig = px.pie(cat_counts, names='category_clean', values='amount', title="Category spend share")
#         st.plotly_chart(fig, use_container_width=True)

#         # Time series plot
#         ts = df_tf.groupby('date')['amount'].sum().reset_index()
#         fig_ts = px.line(ts, x='date', y='amount', title='Total expenses over time')
#         st.plotly_chart(fig_ts, use_container_width=True)
# ---------------------------
# Tab 2: EDA & Visualization
# ---------------------------
with tab_eda:
    if 'df' in locals():
        # Category normalization
        category_map = {
            'food': 'food', 'fod': 'food', 'foodd': 'food', 'foods': 'food',
            'education': 'education', 'edu': 'education', 'educaton': 'education',
            'entertainment': 'entertainment', 'entertain': 'entertainment', 'entrtnmnt': 'entertainment',
            'rent': 'rent', 'rentt': 'rent', 'rnt': 'rent',
            'utilities': 'utilities', 'utility': 'utilities', 'utlities': 'utilities', 'utilties': 'utilities',
            'travel': 'travel', 'traval': 'travel', 'travl': 'travel',
            'health': 'health', 'helth': 'health',
            'saving': 'savings', 'savings': 'savings', 'salary': 'income', 'bonus': 'income',
            'misc': 'misc', 'other': 'misc', 'others': 'misc'
        }
        df['category_clean'] = df['category'].map(category_map).fillna(df['category'])

        df_tf = create_time_features(df)

        # Category spending summary table
        cat_counts = (
            df_tf.groupby('category_clean')['amount']
            .sum()
            .reset_index()
            .sort_values('amount', ascending=False)
        )
        st.write("#### Spending by category (total)")
        st.dataframe(cat_counts.style.format({"amount": "{:,.2f}"}))

        # Pie chart: Category share
        color_map = {
            'food':'#FF6B6B', 
            'education':'#4ECDC4', 
            'entertainment':'#556270', 
            'rent':'#C7F464',
            'utilities':'#FFCC5C',
            'travel':'#C44D58',
            'health':'#556B2F',
            'savings':'#1E90FF',
            'income':'#FF8C00',
            'misc':'#A9A9A9'
        }
        fig = px.pie(
            cat_counts,
            names='category_clean',
            values='amount',
            title="Category spend share",
            color='category_clean',
            color_discrete_map=color_map
        )
        fig.update_traces(textinfo='percent+label', pull=[0.05]*len(cat_counts))
        fig.update_layout(template="plotly_white", title_font_size=20, title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

        # Line chart: Total expenses over time
        ts = df_tf.groupby('date')['amount'].sum().reset_index()
        fig_ts = px.line(
            ts,
            x='date',
            y='amount',
            title='Total expenses over time',
            markers=True,
            line_shape='spline'
        )
        fig_ts.update_traces(line=dict(color='#4f46e5', width=3), marker=dict(size=6, color='#0ea5a4'))
        fig_ts.update_layout(
            plot_bgcolor='#f8f9fb',
            paper_bgcolor='#ffffff',
            font=dict(color='#0ea5a4', size=12),
            title=dict(font_size=20, x=0.5)
        )
        st.plotly_chart(fig_ts, use_container_width=True)

# ----------------------------------------------------
# Tab 3: Forecast & Budget
# # ----------------------------------------------------
# with tab_forecast:
#     if 'df' in locals():
#         st.sidebar.header("Forecast settings")
#         horizon_days = st.sidebar.slider("Forecast horizon (days)", min_value=30, max_value=365, value=90, step=30)
#         model_choice = st.sidebar.selectbox("Model (MVP)", ['Prophet', 'Random Forest', 'XGBoost'])

#         if st.sidebar.button("Run Forecast"):
#             with st.spinner(f"Training {model_choice} model..."):
#                 if model_choice == 'Prophet':
#                     forecast_df = forecast_prophet(df, periods=horizon_days)
#                 elif model_choice == 'Random Forest':
#                     forecast_df = forecast_random_forest(df, periods=horizon_days)
#                 elif model_choice == 'XGBoost':
#                     forecast_df = forecast_xgboost(df, periods=horizon_days)

#             st.success(f"Forecast ready ✅ ({model_choice})")

#             # Plot forecast
#             fig_f = px.line(forecast_df, x='ds', y='yhat', title="Forecast (yhat)")
#             fig_f.add_scatter(
#                 x=forecast_df['ds'],
#                 y=forecast_df['yhat_upper'],
#                 mode='lines',
#                 name='upper',
#                 line=dict(width=0),
#                 showlegend=False
#             )
#             fig_f.add_scatter(
#                 x=forecast_df['ds'],
#                 y=forecast_df['yhat_lower'],
#                 mode='lines',
#                 name='lower',
#                 fill='tonexty',
#                 line=dict(width=0),
#                 showlegend=False
#             )
#             st.plotly_chart(fig_f, use_container_width=True)

#             # In-sample accuracy
#             if st.sidebar.checkbox("Show in-sample accuracy (on history)"):
#                 in_sample = forecast_df[forecast_df['ds'] <= df['date'].max()]
#                 y_true = df.groupby('date')['amount'].sum().reindex(in_sample['ds']).fillna(0).values
#                 y_pred = in_sample['yhat'].values[:len(y_true)]
#                 metrics = evaluate_forecast(y_true, y_pred)
#                 st.write("### In-sample metrics", metrics)

#             # Budget Recommendation
#             st.subheader("Budget recommendation")
#             income = st.number_input("Monthly income (currency)", min_value=0.0, value=5000.0)
#             savings_target = st.slider("Desired savings (%)", 0.0, 0.8, 0.2, step=0.05)

#             hist_share = df.groupby('category_clean')['amount'].sum() / df['amount'].sum()
#             cat_forecast_df = pd.concat([
#                 pd.DataFrame({
#                     'ds': forecast_df['ds'],
#                     'category': cat,
#                     'yhat': forecast_df['yhat'] * share
#                 }) for cat, share in hist_share.items()
#             ])

#             out = recommend_budget(
#                 cat_forecast_df,
#                 periods=horizon_days,
#                 income=income,
#                 savings_target_pct=savings_target,
#                 category_col='category',
#                 amount_col='yhat'
#             )

#             # Clean recommendations
#             out['recommended_by_category'] = {
#                 k: v for k, v in out['recommended_by_category'].items()
#                 if k and pd.notna(v) and v > 0 and str(k).lower() != 'nan'
#             }

#             st.json(out)

#             # Download Forecast CSV
#             csv = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
#             st.download_button(
#                 "Download forecast CSV",
#                 csv,
#                 file_name="forecast.csv",
#                 mime="text/csv"
#             )
# ---------------------------
# Tab 3: Forecast & Budget
# ---------------------------
with tab_forecast:
    if 'df' in locals():
        # Sidebar controls visible only in this tab
        # with st.sidebar:
        #     st.header("Forecast settings")
        #     horizon_days = st.slider("Forecast horizon (days)", min_value=30, max_value=365, value=90, step=30)
        #     model_choice = st.selectbox("Model (MVP)", ['Prophet', 'Random Forest', 'XGBoost'])
        #     run_forecast = st.button("Run Forecast")
        with tab_forecast:
            if 'df' in locals():
                # -------------------------
                # Forecast settings inside tab
                # -------------------------
                st.subheader("Forecast Settings")
                horizon_days = st.slider(
                    "Forecast horizon (days)",
                    min_value=30,
                    max_value=365,
                    value=90,
                    step=30
                )
                model_choice = st.selectbox(
                    "Select Model",
                    ['Prophet', 'Random Forest', 'XGBoost']
                )
                run_forecast = st.button("Run Forecast")

        if run_forecast:
            with st.spinner(f"Training {model_choice} model..."):
                if model_choice == 'Prophet':
                    forecast_df = forecast_prophet(df, periods=horizon_days)
                elif model_choice == 'Random Forest':
                    forecast_df = forecast_random_forest(df, periods=horizon_days)
                elif model_choice == 'XGBoost':
                    forecast_df = forecast_xgboost(df, periods=horizon_days)

            st.success(f"Forecast ready ✅ ({model_choice})")

            # Forecast plot with styled colors
            fig_f = px.line(forecast_df, x='ds', y='yhat', title="Forecast (yhat)", markers=True)
            fig_f.add_scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat_upper'],
                mode='lines',
                name='upper',
                line=dict(width=0),
                showlegend=False
            )
            fig_f.add_scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat_lower'],
                mode='lines',
                name='lower',
                fill='tonexty',
                line=dict(width=0),
                showlegend=False
            )
                        # Define colors for each model
            model_colors = {
                'Prophet': '#1f77b4',       # Blue
                'Random Forest': '#2ca02c', # Green
                'XGBoost': '#d62728'        # Red
            }

            line_color = model_colors.get(model_choice, '#FF6B6B')  # default if model not found
            fig_f.update_traces(line=dict(color=line_color, width=3), marker=dict(size=5, color=line_color))

            # fig_f.update_traces(line=dict(color='#FF6B6B', width=3), marker=dict(size=5, color='#FF6B6B'))
            fig_f.update_layout(
                plot_bgcolor='#f8f9fb',
                paper_bgcolor='#ffffff',
                font=dict(color='#0ea5a4', size=12),
                title=dict(font_size=20, x=0.5)
            )
            st.plotly_chart(fig_f, use_container_width=True)

            # In-sample metrics
            if st.checkbox("Show in-sample accuracy (on history)"):
                in_sample = forecast_df[forecast_df['ds'] <= df['date'].max()]
                y_true = df.groupby('date')['amount'].sum().reindex(in_sample['ds']).fillna(0).values
                y_pred = in_sample['yhat'].values[:len(y_true)]
                metrics = evaluate_forecast(y_true, y_pred)
                st.write("### In-sample metrics", metrics)

            # Budget Recommendation
            st.subheader("Budget recommendation")
            income = st.number_input("Monthly income (currency)", min_value=0.0, value=5000.0)
            savings_target = st.slider("Desired savings (%)", 0.0, 0.8, 0.2, step=0.05)

            hist_share = df.groupby('category_clean')['amount'].sum() / df['amount'].sum()
            cat_forecast_df = pd.concat([
                pd.DataFrame({
                    'ds': forecast_df['ds'],
                    'category': cat,
                    'yhat': forecast_df['yhat'] * share
                }) for cat, share in hist_share.items()
            ])

            out = recommend_budget(
                cat_forecast_df,
                periods=horizon_days,
                income=income,
                savings_target_pct=savings_target,
                category_col='category',
                amount_col='yhat'
            )

            # Clean recommendations
            out['recommended_by_category'] = {
                k: v for k, v in out['recommended_by_category'].items()
                if k and pd.notna(v) and v > 0 and str(k).lower() != 'nan'
            }

            st.json(out)

            # Download Forecast CSV
            csv = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download forecast CSV",
                csv,
                file_name="forecast.csv",
                mime="text/csv"
            )
