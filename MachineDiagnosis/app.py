# Machine Diagnosis App
# Kent Katigbak --- PE1

# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile

st.set_page_config(page_title="Machine Health Num | kentjk", layout="wide")

# CSS styling
st.markdown("""
    <style>
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}
    .block-container {padding-top: 0rem; padding-bottom: 0rem;}
    </style>
""", unsafe_allow_html=True)

footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: black;
    color: white;
    align-items: center;
    padding: 3px 10px;
    font-size: 12px !important;
    z-index: 1000;
}

.footer-center {
    text-align: center;
}

</style>

<div class="footer">
    <div class="footer-center">&copy; Kent Katigbak | rev. 2.0 | 2025</div>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)

# Define the correct password
correct_password = st.secrets["password"]

# Create a session state to store the authentication status
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Function to check the password
def check_password(password):
    if password == correct_password:
        st.session_state.authenticated = True
    else:
        st.session_state.authenticated = False

# Main app logic
st.title("🛠️ Machine Health Monitoring")
if not st.session_state.authenticated:
    # Display the password input if not authenticated
    password = st.text_input("Enter your password and click submit twice:", type="password")
    
    if st.button("Submit"):
        check_password(password)
        if st.session_state.authenticated:
            st.success("Password is correct!")
        else:
            st.error("Incorrect password. Please try again.")
else:

    # Sidebar
    with st.sidebar:
        # st.info("©Kent Katigbak | PE1 | rev02 | 2025")
        # st.divider()
        st.subheader("Full App Guide")

        faq= st.selectbox("Please select FAQ",
                    ["How to use the app?", "Does the app store the uploaded data?",
                    "What are the criteria for machine health?",
                    "What does the trend formula mean?",
                    "What is ARIMA forecasting?"])

        if faq == "How to use the app?":
            st.markdown("1. Go to the  PE1 machine monitoring dashboard.")
            st.markdown("2. Click on the MH Data on the upper rght corner under the red cross.")
            st.markdown("3. An excel file will open. Save a copy of the file.")
            st.markdown("4. Drag and drop the file into the app (first tab).")
            st.markdown("5. Proceed to the next tabs.")

        elif faq == "Does the app store the uploaded data?":
            st.markdown("""The app is not connected to any database.
                        It will only process the uploaded file but the data is not saved.
                        The data will also vanish once the app is refreshed.""")

        elif faq == "What are the criteria for machine health?":
            st.markdown("<b><u>Daily Andon Count:</u></b>", unsafe_allow_html=True)
            st.markdown("Healthy<1 | 1<Warning<2 | Critical>2")
            st.markdown("<b><u>Average Waiting Time (mins):</u></b>", unsafe_allow_html=True)
            st.markdown("Healthy<8 | 8<Warning<10 | Critical>10")
            st.markdown("<b><u>Average Fixing Time (mins):</u></b>", unsafe_allow_html=True)
            st.markdown("Healthy<60 | 60<Warning<90 | Critical>90")
            st.markdown("<b><u>Average Loss Time (mins):</u></b>", unsafe_allow_html=True)
            st.markdown("Healthy<68 | 68<Warning<98 | Critical>98")

        elif faq == "What does the trend formula mean?":
            st.markdown("""The trend formula is in the algebraic form <b><i>y=mx+b</i></b>. 
                        <b><i>m</i></b> is the constant beside the variable <b><i>x</i></b>.
                        A positive <b><i>m</i></b> signifies uptrend while a negative <b><i>m</i></b>
                        signifies downtrend.""", unsafe_allow_html=True)

        else:
            st.markdown("""ARIMA (AutoRegressive Integrated Moving Average) is a time series forecasting model
                        that combines three components: autoregression (AR), which uses the relationship between
                        an observation and a number of lagged observations; differencing (I) to make the time series
                        stationary by removing trends; and moving average (MA), which models the relationship
                        between an observation and a residual error from a moving average model applied to lagged
                        observations. The model is specified by three parameters (p, d, q), where p is the order of
                        the AR part, d is the degree of differencing, and q is the order of the MA part.""")

    # --- CACHE: File loading (expensive operation) ---
    @st.cache_data(show_spinner="📂 Loading data...")
    def load_data(file):
        df = pd.read_excel(file)
        df["Adjusted Call Date Time"] = pd.to_datetime(df["Adjusted Call Date Time"], errors="coerce")
        return df

    # --- HELPER: Health logic ---
    def categorize_health(col, metric):
        def status(val):
            if metric == "Avg_Total_Defects":
                if val < 1:
                    return "Healthy"
                elif 1 <= val <= 2:
                    return "Warning"
                else:
                    return "Critical"
            elif metric == "Avg_Total_Waiting_Time":
                if val < 8:
                    return "Healthy"
                elif 8 <= val <= 10:
                    return "Warning"
                else:
                    return "Critical"
            elif metric == "Avg_Total_Fixing_Time":
                if val < 60:
                    return "Healthy"
                elif 60 <= val <= 90:
                    return "Warning"
                else:
                    return "Critical"
            elif metric == "Avg_Total_Loss_Time":
                if val < 68:
                    return "Healthy"
                elif 68 <= val <= 98:
                    return "Warning"
                else:
                    return "Critical"
            else:
                return "Unknown"

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

    # Clean sheet names in excel
    def sanitize_sheet_name(name):
        invalid_chars = ['\\', '/', '*', '?', ':', '[', ']']
        for ch in invalid_chars:
            name = name.replace(ch, "")
        return name[:31]

    # ---------------- MAIN APP ----------------

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📁 Upload & Filter", "📊 Summary & Health", "🔍 Diagnosis", "📈 Trends", "📅 ARIMA Forecast", "📦 Downloads"])

    # ---------- TAB 1: Upload & Filter ----------
    with tab1:
        uploaded_file = st.file_uploader("📂 Upload Excel file:", type="xlsx")

        if uploaded_file:
            raw = load_data(uploaded_file)

            # Validate columns
            required = ["Adjusted Call Date Time", "Line", "Machine", "Machine No.",
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
                filtered = filtered[(filtered["Adjusted Call Date Time"] >= start) & (filtered["Adjusted Call Date Time"] <= end)]
                num_days = (end - start).days + 1
                st.session_state['filtered'] = filtered
                st.session_state['num_days'] = num_days
    
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
                    df[f"{metric}_Status"] = categorize_health(df[metric], metric)
    
                st.session_state['summary_df'] = df
    
                if 'summary_df' in st.session_state:
                    st.subheader("📊 Machine Number Summary")
                    # Exclude columns that end with "_Status"
                    summary_no_status = st.session_state['summary_df'].loc[:, ~st.session_state['summary_df'].columns.str.endswith("_Status")]
                    st.dataframe(summary_no_status, use_container_width=True)
    
                    st.subheader(f"✅ Overall Machine Health Status \n ({start.date().strftime('%B %d, %Y')} to {end.date().strftime('%B %d, %Y')})")
                    health_cols = [col for col in st.session_state['summary_df'].columns if "Status" in col]
                    st.dataframe(st.session_state['summary_df'][health_cols].style.applymap(highlight_status), use_container_width=True)
                    
                    st.markdown("""
                                Disclaimer: The data above is the accumulated data for the whole selected date range.
                                Figures may look different on the daily basis as presented in the daily trend bar chart below.
                                """)
    
                    # --- New Feature: Category Distribution and Trend ---
                    st.subheader("📈 Health Status Daily Trend")
    
                    metric_display = {
                        "Avg_Total_Defects": "Total Defects",
                        "Avg_Total_Waiting_Time": "Total Waiting Time",
                        "Avg_Total_Fixing_Time": "Total Fixing Time",
                        "Avg_Total_Loss_Time": "Total Loss Time"
                    }
    
                    metric_map = {v: k for k, v in metric_display.items()}
                    selected_metric = st.selectbox("📊 Select Metric for Trend", list(metric_display.values()))
                    selected_col = metric_map[selected_metric]
    
                    # Prepare daily machine stats with daily averages
                    daily = filtered.copy()
                    daily["Date"] = pd.to_datetime(daily["Adjusted Call Date Time"]).dt.date
                    grouped = daily.groupby(["Date", "Machine No."]).agg(
                        Total_Defects=("Machine No.", "count"),
                        Total_Waiting_Time=("Waiting Time (mins.)", "sum"),
                        Total_Fixing_Time=("Fixing Time Duration (mins.)", "sum"),
                        Total_Loss_Time=("Total Loss Time", "sum")
                    ).reset_index()
    
                    # Add daily averages (1-day basis)
                    metric_internal = selected_col.replace("Avg_", "")
                    grouped[selected_col] = grouped[metric_internal]
    
                    # Categorize machine health per day
                    grouped["Health_Status"] = categorize_health(grouped[selected_col], selected_col)
    
                    # Count health status per day and collect machine numbers accurately
                    status_counts = grouped.groupby(["Date", "Health_Status"]).agg(
                        Count=("Health_Status", "size"),
                        Machines=("Machine No.", lambda x: ', '.join(map(str, sorted(x.unique()))))
                    ).reset_index()
    
                    # Pivot tables for plotting
                    pivoted = status_counts.pivot(index="Date", columns="Health_Status", values="Count").fillna(0).astype(int)
                    machines_pivoted = status_counts.pivot(index="Date", columns="Health_Status", values="Machines").fillna('')
    
                    # Ensure order of bars
                    for status in ["Healthy", "Warning", "Critical"]:
                        if status not in pivoted.columns:
                            pivoted[status] = 0
                            machines_pivoted[status] = ''
    
                    # Plotting
                    fig = go.Figure()
                    color_map = {"Healthy": "green", "Warning": "yellow", "Critical": "red"}
    
                    for status in ["Healthy", "Warning", "Critical"]:
                        fig.add_trace(go.Bar(
                            x=pivoted.index,
                            y=pivoted[status],
                            name=status,
                            marker_color=color_map[status],
                            hovertemplate=[
                                f"Date: {date}<br>"
                                f"{status}<br>"
                                f"Count: {count}<br>"
                                f"Machine No/s: {machines_pivoted.loc[date, status]}<br>"
                                "<extra></extra>"
                                for date, count in zip(pivoted.index, pivoted[status])
                            ]
                        ))
    
                    fig.update_layout(
                        barmode='stack',
                        title=dict(
                            text=f"📆 Daily Health Status Count for {selected_metric} (by daily averages)",
                            font=dict(color='black')  # Pure black title
                        ),
                        xaxis=dict(
                            title=dict(text="Date", font=dict(color='black')),
                            tickfont=dict(color='black'),
                            linecolor='black',
                            tickcolor='black'
                        ),
                        yaxis=dict(
                            title=dict(text="Machine Count", font=dict(color='black')),
                            tickfont=dict(color='black'),
                            linecolor='black',
                            tickcolor='black'
                        ),
                        legend=dict(
                            title=dict(text="Health Status", font=dict(color='black')),
                            font=dict(color='black')
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
                    st.markdown("""
                            Disclaimer: These are the data of the machines with Andon daily.
                            The total will look diffrent each day because some machines do not have Andon everyday.
                            """)
                    
                                    # --- Create Pivot Tables for Excel Export ---
                    health_metrics = {
                        "Andon Calls": "count",  # special case: count Machine No.
                        "Waiting Time": "Waiting Time (mins.)",
                        "Fixing Time": "Fixing Time Duration (mins.)",
                        "Loss Time": "Total Loss Time"
                    }
    
                    excel_pivots = {}
    
                    for sheet_name, metric in health_metrics.items():
                        daily_copy = filtered.copy()
                        daily_copy["Date"] = pd.to_datetime(daily_copy["Adjusted Call Date Time"]).dt.date
    
                        if sheet_name == "Andon Calls":
                            # Count Machine No. as Total_Defects
                            grouped_metric = daily_copy.groupby(["Date", "Machine No."]).size().reset_index(name="Value")
                            health_metric_name = "Avg_Total_Defects"
                        else:
                            grouped_metric = daily_copy.groupby(["Date", "Machine No."]).agg(Value=(metric, "sum")).reset_index()
                            metric_name_map = {
                                "Waiting Time": "Avg_Total_Waiting_Time",
                                "Fixing Time": "Avg_Total_Fixing_Time",
                                "Loss Time": "Avg_Total_Loss_Time"
                            }
                            health_metric_name = metric_name_map[sheet_name]
    
                        grouped_metric["Health_Status"] = categorize_health(grouped_metric["Value"], health_metric_name)
                        pivot_table = grouped_metric.pivot(index="Machine No.", columns="Date", values="Health_Status").fillna("No Data")
                        excel_pivots[sheet_name] = pivot_table
    
                    # Convert pivot tables to Excel in memory
                    output = BytesIO()
    
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        for sheet_name, df_pivot in excel_pivots.items():
                            # Write the DataFrame to Excel starting from the FIRST row (row=0)
                            df_pivot.to_excel(writer, sheet_name=sheet_name, startrow=0, header=True, index=True)
    
                            workbook = writer.book
                            worksheet = writer.sheets[sheet_name]
    
                            # Define formats
                            border_format = workbook.add_format({'border': 1})
                            critical_format = workbook.add_format({'bg_color': 'red', 'font_color': 'white', 'border': 1})
                            warning_format = workbook.add_format({'bg_color': 'yellow', 'font_color': 'black', 'border': 1})
                            healthy_format = workbook.add_format({'bg_color': 'green', 'font_color': 'white', 'border': 1})
    
                            # Dimensions
                            n_rows, n_cols = df_pivot.shape
                            first_data_row = 1  # Because header is at row 0, data starts at row 1
                            last_data_row = first_data_row + n_rows - 1
                            first_col = 1  # Column B (skip index column)
                            last_col = first_col + n_cols - 1
    
                            # Convert column number to Excel column letters (A, B, ..., Z, AA, AB, ...)
                            def colnum_string(n):
                                string = ""
                                while n >= 0:
                                    string = chr(n % 26 + ord('A')) + string
                                    n = n // 26 - 1
                                return string
    
                            start_cell = f"{colnum_string(first_col)}{first_data_row + 1}"  # +1 for Excel's 1-based indexing
                            end_cell = f"{colnum_string(last_col)}{last_data_row + 1}"
                            cell_range = f"{start_cell}:{end_cell}"
    
                            # Apply formatting
                            worksheet.conditional_format(cell_range, {'type': 'no_blanks', 'format': border_format})
                            worksheet.conditional_format(cell_range, {'type': 'text', 'criteria': 'containing', 'value': 'Critical', 'format': critical_format})
                            worksheet.conditional_format(cell_range, {'type': 'text', 'criteria': 'containing', 'value': 'Warning', 'format': warning_format})
                            worksheet.conditional_format(cell_range, {'type': 'text', 'criteria': 'containing', 'value': 'Healthy', 'format': healthy_format})
                            worksheet.conditional_format(cell_range, {'type': 'text', 'criteria': 'containing', 'value': 'No Data', 'format': healthy_format})
    
                    output.seek(0)
    
                    st.download_button(
                        label=f"📥 Download {line} {machine} Machine Health Status",
                        data=output,
                        file_name=f"{line}_{machine}_{start.date()}_to_{end.date()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
            else:
                st.markdown("Please select date range.")

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
            machine_list = [int(x) for x in machine_list]
            
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
            trend_data['Date'] = trend_data['Adjusted Call Date Time'].dt.date
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
                fig_loss_time.add_shape(type="line", x0=pm_date, x1=pm_date, y0=0, y1=daily_data["Total_Loss_Time"].max(), line=dict(color="Green", width=3))
                fig_andon_count.add_shape(type="line", x0=pm_date, x1=pm_date, y0=0, y1=daily_data["Andon_Count"].max(), line=dict(color="Green", width=3))
                fig_mttr.add_shape(type="line", x0=pm_date, x1=pm_date, y0=0, y1=daily_data["MTTR"].max(), line=dict(color="Green", width=3))

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
            forecast_data['Date'] = forecast_data['Adjusted Call Date Time'].dt.date
            
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
                # Display error message
                st.error(f"Cannot forecast due to: {str(e)}")

    # ---------- TAB 6: Downloads ----------
    with tab6:
        if 'raw' not in st.session_state or 'num_days' not in st.session_state:
            st.warning("⚠️ Please upload a file and select date range in Tab 2 first.")
        else:
            raw = st.session_state['raw']
            start = pd.to_datetime(st.session_state['filtered']["Adjusted Call Date Time"].min())
            end = pd.to_datetime(st.session_state['filtered']["Adjusted Call Date Time"].max())
            date_range_str = f"{start.date()}_to_{end.date()}"
    
            # Mapping of button label to health metric and data column
            download_options = {
                "Andon Logs Status": ("Avg_Total_Defects", "count"),
                "Waiting Time Status": ("Avg_Total_Waiting_Time", "Waiting Time (mins.)"),
                "Fixing Time Status": ("Avg_Total_Fixing_Time", "Fixing Time Duration (mins.)"),
                "Loss Time Status": ("Avg_Total_Loss_Time", "Total Loss Time")
            }
    
            def sanitize_sheet_name(name):
                invalid_chars = ['\\', '/', '*', '?', ':', '[', ']']
                for ch in invalid_chars:
                    name = name.replace(ch, "")
                return name[:31]
    
            def generate_zip(metric_label, metric_key, column_source):
                zip_buffer = BytesIO()
    
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for line in raw["Line"].unique():
                        line_data = raw[raw["Line"] == line]
                        excel_buffer = BytesIO()
    
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            workbook = writer.book
    
                            for machine in line_data["Machine"].unique():
                                df_machine = line_data[line_data["Machine"] == machine].copy()
                                df_machine["Date"] = pd.to_datetime(df_machine["Adjusted Call Date Time"]).dt.date
                                df_machine = df_machine[(df_machine["Date"] >= start.date()) & (df_machine["Date"] <= end.date())]
    
                                if df_machine.empty:
                                    continue  # Skip machines with no data in date range
    
                                # Group and pivot
                                if column_source == "count":
                                    grouped = df_machine.groupby(["Date", "Machine No."]).size().reset_index(name="Value")
                                else:
                                    grouped = df_machine.groupby(["Date", "Machine No."]).agg(Value=(column_source, "sum")).reset_index()
    
                                grouped["Health_Status"] = categorize_health(grouped["Value"], metric_key)
                                pivot = grouped.pivot(index="Machine No.", columns="Date", values="Health_Status").fillna("No Data")
    
                                if pivot.empty:
                                    continue  # Avoid empty pivot tables
    
                                safe_sheet_name = sanitize_sheet_name(machine)
                                pivot.to_excel(writer, sheet_name=safe_sheet_name, startrow=0)
                                worksheet = writer.sheets[safe_sheet_name]
    
                                # Determine formatting range
                                n_rows, n_cols = pivot.shape
                                first_row, first_col = 1, 1
                                last_row = first_row + n_rows - 1
                                last_col = first_col + n_cols - 1
    
                                def colnum_string(n):
                                    string = ""
                                    while n >= 0:
                                        string = chr(n % 26 + ord('A')) + string
                                        n = n // 26 - 1
                                    return string
    
                                start_cell = f"{colnum_string(first_col)}{first_row + 1}"
                                end_cell = f"{colnum_string(last_col)}{last_row + 1}"
                                cell_range = f"{start_cell}:{end_cell}"
    
                                # Conditional formatting
                                formats = {
                                    "Critical": workbook.add_format({'bg_color': 'red', 'font_color': 'white', 'border': 1}),
                                    "Warning": workbook.add_format({'bg_color': 'yellow', 'font_color': 'black', 'border': 1}),
                                    "Healthy": workbook.add_format({'bg_color': 'green', 'font_color': 'white', 'border': 1}),
                                    "No Data": workbook.add_format({'bg_color': 'white', 'border': 1}),
                                }
    
                                for status, fmt in formats.items():
                                    worksheet.conditional_format(cell_range, {
                                        'type': 'text',
                                        'criteria': 'containing',
                                        'value': status,
                                        'format': fmt
                                    })
    
                                # Auto-fit data columns
                                for col_num, col in enumerate(pivot.columns, start=1):
                                    col_width = max([len(str(v)) for v in pivot[col].values] + [len(str(col))]) + 2
                                    worksheet.set_column(col_num, col_num, col_width)
    
                                # Auto-fit index column (Machine No.)
                                if not pivot.index.empty:
                                    index_width = max([len(str(v)) for v in pivot.index] + [len("Machine No.")]) + 2
                                    worksheet.set_column(0, 0, index_width)
    
                        excel_filename = f"{line}_{metric_label.replace(' ', '_').lower()}_{date_range_str}.xlsx"
                        zip_file.writestr(excel_filename, excel_buffer.getvalue())
    
                zip_buffer.seek(0)
                return zip_buffer
    
            # UI Buttons and Handlers
            for label, (metric_key, source_col) in download_options.items():
                if st.button(label):
                    zip_bytes = generate_zip(label, metric_key, source_col)
                    st.download_button(
                        label=f"📦 Download {label} ZIP",
                        data=zip_bytes,
                        file_name=f"{label.replace(' ', '_').lower()}_{date_range_str}.zip",
                        mime="application/zip"
                    )

st.divider()

#with open("MachineDiagnosis/style.css") as f:
    #css = f.read()

#st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
