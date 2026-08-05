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
train a machine learning model, and generate downloadable forecasts.
"""
)
with st.expander("About this application"):
    st.write(
        """
        AutoForecast Lab demonstrates an end-to-end machine learning
        forecasting workflow, including:

        • Data validation

        • Feature engineering

        • Time-series model training

        • Forecast evaluation

        • Interactive visualization

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
            with st.spinner("Training model and generating forecast..."):
                result = run_forecast_pipeline(
                    df=df,
                    target_column=target_column,
                    timestamp_column=timestamp_column,
                )

            st.success("Forecast completed!")

            st.subheader("Model Performance")

            model_column, rmse_column, mae_column = st.columns(3)

            model_column.metric(
                "Model",
                result.evaluation.model_name,
            )

            rmse_column.metric(
                "RMSE",
                f"{result.evaluation.rmse:.3f}",
            )

            mae_column.metric(
                "MAE",
                f"{result.evaluation.mae:.3f}",
            )

            st.subheader("Actual vs. Predicted")

            chart_data = result.forecast.set_index(timestamp_column)[
                ["actual", "prediction"]
            ]

            st.line_chart(chart_data)

            st.subheader("Forecast Results")

            st.dataframe(
                result.forecast,
                width="stretch",
            )

            forecast_csv = result.forecast.to_csv(index=False)

            st.download_button(
                label="Download Forecast CSV",
                data=forecast_csv,
                file_name="forecast.csv",
                mime="text/csv",
            )

        except (KeyError, ValueError) as exc:
            st.error(f"Unable to run forecast: {exc}")


st.divider()

st.caption("Built by Weijia Peng | Portfolio")
