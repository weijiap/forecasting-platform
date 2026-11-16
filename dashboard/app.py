import pandas as pd
import streamlit as st

from forecasting_platform.pipeline import run_forecast_pipeline

st.set_page_config(
    page_title="AutoForecast Lab",
    page_icon="📈",
    layout="wide",
)

st.title("AutoForecast Lab")
st.markdown(
    """
Automatically validate a time-series dataset, engineer forecasting features,
compare models against naive baselines, and generate downloadable forecasts.
"""
)
with st.expander("About this application"):
    st.write(
        """
        AutoForecast Lab demonstrates an end-to-end machine learning
        forecasting workflow, including:

        • Data validation

        • Feature engineering

        • Walk-forward model selection against naive baselines

        • Holdout evaluation

        • Multi-step forecasts beyond the last observation

        • Forecast export
        """
    )
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"],
)

timestamp_column = st.sidebar.text_input(
    "Timestamp column",
    value="date",
)

target_column = st.sidebar.text_input(
    "Target column",
    value="sales",
)

horizon = st.sidebar.slider(
    "Forecast horizon (periods)",
    min_value=1,
    max_value=90,
    value=14,
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    summary_col1.metric(
        "Rows",
        len(df),
    )

    summary_col2.metric(
        "Columns",
        len(df.columns),
    )

    summary_col3.metric(
        "Missing Values",
        int(df.isna().sum().sum()),
    )

    summary_col4, summary_col5 = st.columns(2)

    summary_col4.metric(
        "Target",
        target_column,
    )

    summary_col5.metric(
        "Timestamp",
        timestamp_column,
    )

    st.subheader("Data Preview")

    st.dataframe(
        df.head(),
        width="stretch",
    )

    st.divider()

    if st.button("Run Forecast"):
        try:
            with st.spinner("Comparing models and generating forecast..."):
                result = run_forecast_pipeline(
                    df=df,
                    target_column=target_column,
                    timestamp_column=timestamp_column,
                    horizon=horizon,
                )

            st.success("Forecast completed!")

            st.subheader("Model Performance (holdout)")

            model_column, rmse_column, mae_column = st.columns(3)

            model_column.metric(
                "Selected model",
                result.selected_model,
            )

            rmse_column.metric(
                "RMSE",
                f"{result.evaluation.rmse:.3f}",
            )

            mae_column.metric(
                "MAE",
                f"{result.evaluation.mae:.3f}",
            )

            st.subheader("Model Comparison")
            st.caption(
                "Models are selected on walk-forward validation within the "
                "training period; the holdout is used only for reporting."
            )

            holdout = {e.model_name: e for e in result.holdout}
            comparison = pd.DataFrame(
                [
                    {
                        "model": b.model_name,
                        "backtest_rmse": round(b.rmse, 3),
                        "backtest_mae": round(b.mae, 3),
                        "holdout_rmse": round(holdout[b.model_name].rmse, 3),
                        "holdout_mae": round(holdout[b.model_name].mae, 3),
                    }
                    for b in result.backtest
                ]
            ).sort_values("backtest_rmse")

            st.dataframe(
                comparison,
                width="stretch",
                hide_index=True,
            )

            st.subheader("Holdout: Actual vs. Predicted")

            chart_data = result.forecast.set_index(timestamp_column)[
                ["actual", "prediction"]
            ]

            st.line_chart(chart_data)

            st.subheader(f"Forecast: next {horizon} periods")

            history = df[[timestamp_column, target_column]].copy()
            history[timestamp_column] = pd.to_datetime(history[timestamp_column])
            history = history.sort_values(timestamp_column).tail(90)

            future_chart = pd.concat(
                [
                    history.set_index(timestamp_column).rename(
                        columns={target_column: "history"}
                    ),
                    result.future_forecast.set_index(timestamp_column),
                ]
            )

            st.line_chart(future_chart)

            download_col1, download_col2 = st.columns(2)

            download_col1.download_button(
                label="Download Holdout Predictions CSV",
                data=result.forecast.to_csv(index=False),
                file_name="holdout_predictions.csv",
                mime="text/csv",
            )

            download_col2.download_button(
                label="Download Future Forecast CSV",
                data=result.future_forecast.to_csv(index=False),
                file_name="future_forecast.csv",
                mime="text/csv",
            )

        except (KeyError, ValueError) as exc:
            st.error(f"Unable to run forecast: {exc}")


st.divider()

st.caption("Built by Weijia Peng | Portfolio")
