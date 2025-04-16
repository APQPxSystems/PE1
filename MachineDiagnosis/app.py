import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Machine Health | kentjk", layout="wide")

# CSS styling
st.markdown("""
    <style>
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}
    .block-container {padding-top: 0rem; padding-bottom: 0rem;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.subheader("Status Judgment Reference")
st.sidebar.image("MachineDiagnosis/Boxplot.png")
st.sidebar.markdown("PE1 | K. Katigbak | rev. 02 | 2025")

# --- CACHE: File loading (expensive operation) ---
@st.cache_data(show_spinner="📂 Loading data...")
def load_data(file):
    df = pd.read_excel(file)
    df["Call Date No Time"] = pd.to_datetime(df["Call Date No Time"], errors="coerce")
    return df

# --- HELPER: Health logic ---
def categorize_health(col):
    q1, q3 = np.percentile(col, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    median = np.median(col)

    def status(val):
        if val < lower or val < median:
            return "Healthy"
        elif val > upper:
            return "Critical"
        else:
            return "Warning"

    return col.apply(status)

def highlight_status(val):
    if isinstance(val, str):
        if 'Critical' in val:
            return 'background-color: red; color: white'
        elif 'Warning' in val:
            return 'background-color: yellow; color: black'
        elif 'Healthy' in val:
            return 'background-color: green; color: white'
    return ''

# ARIMA Forecast function
def arima_forecast(data, forecast_days):
    def forecast_series(series):
        # Ensure series has no NaNs and proper indexing
        series = series.dropna()
        if len(series) < 2:
            return pd.Series([0]*forecast_days)
        model = ARIMA(series, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_days)
        return pd.Series(forecast)

    andon_forecast = forecast_series(data['Andon_Count'])
    loss_forecast = forecast_series(data['Total_Loss_Time'])
    mttr_forecast = forecast_series(data['MTTR'])

    return andon_forecast, loss_forecast, mttr_forecast

# ---------------- MAIN APP ----------------
st.title("🛠️ Machine Health Monitoring")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Upload & Filter", "📊 Summary & Health", "🔍 Diagnosis", "📈 Trends", "📅 ARIMA Forecast"])

# ---------- TAB 1: Upload & Filter ----------
with tab1:
    uploaded_file = st.file_uploader("📂 Upload Excel file:", type="xlsx")

    if uploaded_file:
        raw = load_data(uploaded_file)

        # Validate columns
        required = ["Call Date No Time", "Line", "Machine", "Machine No.",
                    "Waiting Time (mins.)", "Fixing Time Duration (mins.)", "Total Loss Time"]
        if not all(col in raw.columns for col in required):
            st.error("❌ Missing one or more required columns.")
        else:
            st.session_state['raw'] = raw[required]

            st.success("✅ File loaded successfully. Proceed to filtering.")
    else:
        st.info("📤 Please upload an Excel file to start.")

# ---------- TAB 2: Summary & Health ----------
with tab2:
    if 'raw' not in st.session_state:
        st.warning("⚠️ Please upload a file in the Upload tab.")
    else:
        raw = st.session_state['raw']

        line = st.selectbox("📍 Select Line", raw["Line"].unique())
        filtered = raw[raw["Line"] == line]

        machine = st.selectbox("⚙️ Select Machine", filtered["Machine"].unique())
        filtered = filtered[filtered["Machine"] == machine]

        date_range = st.date_input("📅 Date Range", [])
        if len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filtered = filtered[(filtered["Call Date No Time"] >= start) & (filtered["Call Date No Time"] <= end)]
            num_days = (end - start).days + 1
            st.session_state['filtered'] = filtered
            st.session_state['num_days'] = num_days

            if st.button("✅ Run Analysis"):
                df = filtered.groupby("Machine No.", as_index=True).agg(
                    Total_Defects=("Machine No.", "count"),
                    Total_Waiting_Time=("Waiting Time (mins.)", "sum"),
                    Total_Fixing_Time=("Fixing Time Duration (mins.)", "sum"),
                    Total_Loss_Time=("Total Loss Time", "sum")
                )

                # Calculate averages
                for col in df.columns:
                    df[f"Avg_{col}"] = df[col] / num_days

                # Round all float columns to 2 decimals
                df = df.round(2)

                # Categorize health
                for metric in ["Avg_Total_Defects", "Avg_Total_Waiting_Time",
                            "Avg_Total_Fixing_Time", "Avg_Total_Loss_Time"]:
                    df[f"{metric}_Status"] = categorize_health(df[metric])

                st.session_state['summary_df'] = df

        if 'summary_df' in st.session_state:
            st.subheader("📊 Machine Number Summary")
            st.dataframe(st.session_state['summary_df'], use_container_width=True)

            st.subheader("✅ Machine Health Status")
            health_cols = [col for col in st.session_state['summary_df'].columns if "Status" in col]
            st.dataframe(st.session_state['summary_df'][health_cols].style.applymap(highlight_status), use_container_width=True)

# ---------- TAB 3: Diagnosis ----------
def update_selected_machine():
    # This function will be called when the selected machine changes
    st.session_state['selected_machine'] = st.session_state['selected_machine']

with tab3:
    if 'summary_df' not in st.session_state:
        st.warning("⚠️ Please run the summary analysis first.")
    else:
        summary_df = st.session_state['summary_df']
        filtered = st.session_state['filtered']
        num_days = st.session_state['num_days']

        machine_list = summary_df.index.tolist()
        
        # Use the selectbox with on_change to update the session state
        selected_machine = st.selectbox(
            "🔎 Select Machine No. for diagnosis", 
            machine_list, 
            key='selected_machine', 
            on_change=update_selected_machine
        )

        diag_data = filtered[filtered["Machine No."] == selected_machine]

        st.markdown(f"#### Diagnosis for `{line} {machine}` Machine No. `{selected_machine}`")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("📉 Total Defects", len(diag_data))
            st.metric("⏱️ Total Wait Time", round(diag_data["Waiting Time (mins.)"].sum(), 2))
            st.metric("🔧 Total Fixing Time", round(diag_data["Fixing Time Duration (mins.)"].sum(), 2))

        with col2:
            total_loss = diag_data["Total Loss Time"].sum()
            st.metric("💸 Total Loss Time", round(total_loss, 2))
            st.metric("📊 Avg Daily Defects", round(len(diag_data) / num_days, 2))
            st.metric("🧮 Avg Daily Loss Time", round(total_loss / num_days, 2))

        st.markdown("#### 📌 Improvement Suggestions")
        for metric in ["Avg_Total_Defects", "Avg_Total_Waiting_Time",
                    "Avg_Total_Fixing_Time", "Avg_Total_Loss_Time"]:
            status = summary_df.loc[selected_machine, f"{metric}_Status"]
            label = metric.replace("Avg_Total_", "").replace("_", " ")
            if status == "Critical":
                st.warning(f"⚠️ {label} is CRITICAL – take action.")
            elif status == "Warning":
                st.info(f"🟡 {label} is in WARNING – monitor it.")
            elif status == "Healthy":
                st.success(f"✅ {label} is HEALTHY – all good.")

# ---------- TAB 4: Trends ----------
with tab4:
    if 'filtered' not in st.session_state or 'summary_df' not in st.session_state or 'num_days' not in st.session_state:
        st.warning("⚠️ Please upload a file and run the summary analysis first.")
    else:
        filtered = st.session_state['filtered']
        num_days = st.session_state['num_days']
        summary_df = st.session_state['summary_df']

        machine_list = summary_df.index.tolist()
        selected_machine = st.session_state.get('selected_machine', machine_list[0])

        # Add a title indicating the source of the trend data
        st.subheader(f"📈 Trendline and Slope of the Line Formula: \n Line: `{line}`, Machine: `{machine}`, Machine No.: `{selected_machine}`")

        pm_date_recorded = st.checkbox("Is there a recorded Preventive Maintenance Date?")
        if pm_date_recorded:
            pm_date = st.date_input("📅 Select Last Preventive Maintenance Date")
            pm_date = pd.to_datetime(pm_date)

        # Filter data for selected machine
        trend_data = filtered[filtered["Machine No."] == selected_machine]

        # Aggregate data by day
        trend_data['Date'] = trend_data['Call Date No Time'].dt.date
        daily_data = trend_data.groupby('Date').agg(
            Total_Loss_Time=('Total Loss Time', 'sum'),
            Andon_Count=('Total Loss Time', 'count')
        ).reset_index()
        daily_data['MTTR'] = daily_data['Total_Loss_Time'] / daily_data['Andon_Count'] / 60

        # Calculate trendline slopes
        def calculate_slope(data, y_col):
            if len(data) < 2:
                return 0, 0
            x = np.arange(len(data))
            y = data[y_col].values
            m, b = np.polyfit(x, y, 1)
            return m, b

        slope_loss_time, intercept_loss_time = calculate_slope(daily_data, "Total_Loss_Time")
        slope_andon_count, intercept_andon_count = calculate_slope(daily_data, "Andon_Count")
        slope_mttr, intercept_mttr = calculate_slope(daily_data, "MTTR")

        daily_data["Trend_Loss_Time"] = slope_loss_time * np.arange(len(daily_data)) + intercept_loss_time
        daily_data["Trend_Andon_Count"] = slope_andon_count * np.arange(len(daily_data)) + intercept_andon_count
        daily_data["Trend_MTTR"] = slope_mttr * np.arange(len(daily_data)) + intercept_mttr

        # Create trendline charts
        fig_loss_time = px.line(daily_data, x="Date", y="Total_Loss_Time", title="Total Loss Time Trend")
        fig_loss_time.update_traces(line=dict(color='darkblue', width=3))
        fig_loss_time.add_scatter(x=daily_data["Date"], y=daily_data["Trend_Loss_Time"], mode='lines', name=f"Trendline: y = {slope_loss_time:.2f}x + {intercept_loss_time:.2f}", line=dict(dash="dash", color="red", width=2))

        fig_loss_time.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            title_font=dict(color='black'),
            legend=dict(font=dict(color='black')),
            xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'))
        )

        fig_andon_count = px.line(daily_data, x="Date", y="Andon_Count", title="Andon Count Trend")
        fig_andon_count.update_traces(line=dict(color='darkblue', width=3))
        fig_andon_count.add_scatter(x=daily_data["Date"], y=daily_data["Trend_Andon_Count"], mode='lines', name=f"Trendline: y = {slope_andon_count:.2f}x + {intercept_andon_count:.2f}", line=dict(dash="dash", color="red", width=2))

        fig_andon_count.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            title_font=dict(color='black'),
            legend=dict(font=dict(color='black')),
            xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'))
        )

        fig_mttr = px.line(daily_data, x="Date", y="MTTR", title="MTTR Trend")
        fig_mttr.update_traces(line=dict(color='darkblue', width=3))
        fig_mttr.add_scatter(x=daily_data["Date"], y=daily_data["Trend_MTTR"], mode='lines', name=f"Trendline: y = {slope_mttr:.2f}x + {intercept_mttr:.2f}", line=dict(dash="dash", color="red", width=2))

        fig_mttr.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            title_font=dict(color='black'),
            legend=dict(font=dict(color='black')),
            xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'))
        )

        # Add target lines for specific machines
        fig_mttr.add_shape(type="line", x0=pd.Timestamp(daily_data["Date"].min()), x1=pd.Timestamp(daily_data["Date"].max()), y0=0.74, y1=0.74, line=dict(color="darkblue", dash="dash", width=2))
        fig_mttr.add_annotation(x=pd.Timestamp(daily_data["Date"].max()), y=0.74, text="SAM Target", showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(color="darkblue"))
        fig_mttr.add_shape(type="line", x0=pd.Timestamp(daily_data["Date"].min()), x1=pd.Timestamp(daily_data["Date"].max()), y0=0.38, y1=0.38, line=dict(color="darkblue", dash="dash", width=2))
        fig_mttr.add_annotation(x=pd.Timestamp(daily_data["Date"].max()), y=0.38, text="TRD Target", showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(color="darkblue"))

        # Add Preventive Maintenance date marker
        if pm_date_recorded:
            fig_loss_time.add_shape(type="line", x0=pm_date, x1=pm_date, y0=0, y1=daily_data["Total_Loss_Time"].max(), line=dict(color="Green", width=5))
            fig_andon_count.add_shape(type="line", x0=pm_date, x1=pm_date, y0=0, y1=daily_data["Andon_Count"].max(), line=dict(color="Green", width=5))
            fig_mttr.add_shape(type="line", x0=pm_date, x1=pm_date, y0=0, y1=daily_data["MTTR"].max(), line=dict(color="Green", width=5))

            # Calculate trends after the PM date
            post_pm_data = daily_data[daily_data["Date"] > pm_date.date()]
            if not post_pm_data.empty:
                slope_post_pm_loss_time, intercept_post_pm_loss_time = calculate_slope(post_pm_data, "Total_Loss_Time")
                slope_post_pm_andon_count, intercept_post_pm_andon_count = calculate_slope(post_pm_data, "Andon_Count")
                slope_post_pm_mttr, intercept_post_pm_mttr = calculate_slope(post_pm_data, "MTTR")

                post_pm_data["Trend_Loss_Time"] = slope_post_pm_loss_time * np.arange(len(post_pm_data)) + intercept_post_pm_loss_time
                post_pm_data["Trend_Andon_Count"] = slope_post_pm_andon_count * np.arange(len(post_pm_data)) + intercept_post_pm_andon_count
                post_pm_data["Trend_MTTR"] = slope_post_pm_mttr * np.arange(len(post_pm_data)) + intercept_post_pm_mttr

                # Add post-PM trendlines to the figures
                fig_loss_time.add_scatter(x=post_pm_data["Date"], y=post_pm_data["Trend_Loss_Time"], mode='lines', name=f"Post-PM Trendline: y = {slope_post_pm_loss_time:.2f}x + {intercept_post_pm_loss_time:.2f}", line=dict(dash="dash", color="green", width=2))
                fig_andon_count.add_scatter(x=post_pm_data["Date"], y=post_pm_data["Trend_Andon_Count"], mode='lines', name=f"Post-PM Trendline: y = {slope_post_pm_andon_count:.2f}x + {intercept_post_pm_andon_count:.2f}", line=dict(dash="dash", color="green", width=2))
                fig_mttr.add_scatter(x=post_pm_data["Date"], y=post_pm_data["Trend_MTTR"], mode='lines', name=f"Post-PM Trendline: y = {slope_post_pm_mttr:.2f}x + {intercept_post_pm_mttr:.2f}", line=dict(dash="dash", color="green", width=2))

                # Individual analysis for each graph
                st.markdown("#### Preventive Maintenance Analysis")

                # Total Loss Time Analysis
                st.plotly_chart(fig_loss_time, use_container_width=True)
                if slope_post_pm_loss_time < 0:
                    st.success("✅ The Total Loss Time trend shows a downtrend after the Preventive Maintenance. This indicates an improvement in machine performance.")
                elif slope_post_pm_loss_time > 0:
                    st.warning("⚠️ The Total Loss Time trend shows an uptrend after the Preventive Maintenance. Further investigation may be needed.")
                else:
                    st.info("🟡 The Total Loss Time trend remains stable after the Preventive Maintenance. Continue monitoring.")

                # Andon Count Analysis
                st.plotly_chart(fig_andon_count, use_container_width=True)
                if slope_post_pm_andon_count < 0:
                    st.success("✅ The Andon Count trend shows a downtrend after the Preventive Maintenance, suggesting fewer incidents requiring attention.")
                elif slope_post_pm_andon_count > 0:
                    st.warning("⚠️ The Andon Count trend shows an uptrend after the Preventive Maintenance. This may indicate an increase in issues that need to be addressed.")
                else:
                    st.info("🟡 The Andon Count trend remains stable after the Preventive Maintenance. Keep monitoring for any changes.")

                # MTTR Analysis
                st.plotly_chart(fig_mttr, use_container_width=True)
                if slope_post_pm_mttr < 0:
                    st.success("✅ The MTTR trend shows a downtrend after the Preventive Maintenance, indicating improved response times.")
                elif slope_post_pm_mttr > 0:
                    st.warning("⚠️ The MTTR trend shows an uptrend after the Preventive Maintenance. This could suggest delays in addressing issues.")
                else:
                    st.info("🟡 The MTTR trend remains stable after the Preventive Maintenance. Continue to monitor closely.")

            else:
                st.warning("⚠️ No data available after the selected Preventive Maintenance date.")

        # Display charts for the initial trends if PM date is not recorded
        else:
            st.plotly_chart(fig_loss_time, use_container_width=True)
            st.plotly_chart(fig_andon_count, use_container_width=True)
            st.plotly_chart(fig_mttr, use_container_width=True)
            
# ---------- TAB 5: ARIMA Forecast ----------
with tab5:
    if 'filtered' not in st.session_state or 'summary_df' not in st.session_state:
        st.warning("⚠️ Please upload a file and run the summary analysis first.")
    else:
        # Retrieve selected filters
        filtered = st.session_state['filtered']
        num_days = st.session_state['num_days']
        summary_df = st.session_state['summary_df']
        machine_list = summary_df.index.tolist()
        selected_machine = st.session_state.get('selected_machine', machine_list[0])
        
        # Add a title indicating the source of the trend data
        st.subheader(f"📅 Autoregressive Integrated Moving Average (ARIMA) Forecast: \n Line: `{line}`, Machine: `{machine}`, Machine No.: `{selected_machine}`")

        # Filter the data based on the selected machine
        forecast_data = filtered[filtered["Machine No."] == selected_machine]
        forecast_data['Date'] = forecast_data['Call Date No Time'].dt.date
        
        # Aggregate by date (just in case)
        daily_data = forecast_data.groupby('Date').agg(
            Total_Loss_Time=('Total Loss Time', 'sum'),
            Andon_Count=('Total Loss Time', 'count')
        ).reset_index()
        daily_data['MTTR'] = daily_data['Total_Loss_Time'] / daily_data['Andon_Count'] / 60

        # Input: Number of forecast days
        forecast_days = st.number_input("Enter the number of days to forecast", min_value=1, max_value=365, value=30)

        # Perform ARIMA Forecast with error handling
        try:
            andon_forecast, loss_forecast, mttr_forecast = arima_forecast(daily_data, forecast_days)

            # Plotting results
            fig, axes = plt.subplots(3, 1, figsize=(10, 15))

            # Plot Andon Count Forecast
            axes[0].plot(daily_data['Date'], daily_data['Andon_Count'], label='Actual', color='blue')
            axes[0].plot(pd.date_range(daily_data['Date'].iloc[-1], periods=forecast_days + 1, freq='D')[1:], andon_forecast, label='Forecast', color='red')
            axes[0].set_title('Andon Count Forecast')
            axes[0].legend()

            # Plot Total Loss Time Forecast
            axes[1].plot(daily_data['Date'], daily_data['Total_Loss_Time'], label='Actual', color='blue')
            axes[1].plot(pd.date_range(daily_data['Date'].iloc[-1], periods=forecast_days + 1, freq='D')[1:], loss_forecast, label='Forecast', color='red')
            axes[1].set_title('Total Loss Time Forecast')
            axes[1].legend()

            # Plot MTTR Forecast
            axes[2].plot(daily_data['Date'], daily_data['MTTR'], label='Actual', color='blue')
            axes[2].plot(pd.date_range(daily_data['Date'].iloc[-1], periods=forecast_days + 1, freq='D')[1:], mttr_forecast, label='Forecast', color='red')
            axes[2].set_title('MTTR Forecast')
            axes[2].legend()

            # Display the plots
            # st.pyplot(fig)

            # Display interactive forecast charts using Plotly
            # Total Loss Time Forecast Chart
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=daily_data['Date'], y=daily_data['Total_Loss_Time'], mode='lines', name='Actual Total Loss Time'))
            fig_loss.add_trace(go.Scatter(x=pd.date_range(daily_data['Date'].iloc[-1], periods=forecast_days + 1, freq='D')[1:], y=loss_forecast, mode='lines', name='Forecast Total Loss Time', line=dict(dash='dot')))
            fig_loss.update_traces(line=dict(color='darkblue', width=3))
            fig_loss.update_layout(title="Total Loss Time Forecast",xaxis_title="Date", yaxis_title="Total Loss Time",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            title_font=dict(color='black'),
            legend=dict(font=dict(color='black')),
            xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')))
            st.plotly_chart(fig_loss)
                        
            # Andon Count Forecast Chart
            fig_andon = go.Figure()
            fig_andon.add_trace(go.Scatter(x=daily_data['Date'], y=daily_data['Andon_Count'], mode='lines', name='Actual Andon Count'))
            fig_andon.add_trace(go.Scatter(x=pd.date_range(daily_data['Date'].iloc[-1], periods=forecast_days + 1, freq='D')[1:], y=andon_forecast, mode='lines', name='Forecast Andon Count', line=dict(dash='dot')))
            fig_andon.update_traces(line=dict(color='darkblue', width=3))
            fig_andon.update_layout(title="Andon Count Forecast", xaxis_title="Date", yaxis_title="Andon Count",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            title_font=dict(color='black'),
            legend=dict(font=dict(color='black')),
            xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')))
            st.plotly_chart(fig_andon)

            # MTTR Forecast Chart
            fig_mttr = go.Figure()
            fig_mttr.add_trace(go.Scatter(x=daily_data['Date'], y=daily_data['MTTR'], mode='lines', name='Actual MTTR'))
            fig_mttr.add_trace(go.Scatter(x=pd.date_range(daily_data['Date'].iloc[-1], periods=forecast_days + 1, freq='D')[1:], y=mttr_forecast, mode='lines', name='Forecast MTTR', line=dict(dash='dot')))
            fig_mttr.update_traces(line=dict(color='darkblue', width=3))
            fig_mttr.update_layout(title="MTTR Forecast", xaxis_title="Date", yaxis_title="MTTR",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            title_font=dict(color='black'),
            legend=dict(font=dict(color='black')),
            xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')))
            st.plotly_chart(fig_mttr)

        except Exception as e:
            # Display a user-friendly error message
            st.error(f"Cannot forecast due to: {str(e)}")

st.write("---")
