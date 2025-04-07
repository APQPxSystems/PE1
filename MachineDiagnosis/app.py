import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Page configurations
st.set_page_config(page_title="Machine Health | kentjk", layout="wide")
hide_st_style = """
                <style>
                #MainMenu {visibility:visible;}
                footer {visibility:hidden;}
                header {visibility:visible;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Remove top white space
st.markdown("""
        <style>
            .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

# App title and description
st.title("Machine Health Monitoring")

# Sidebar boxplot
st.sidebar.subheader("Status Judgment Reference")
st.sidebar.image("Boxplot.png")

# File uploader
raw = st.file_uploader("Upload Excel file here:", type='xlsx')

def highlight_status(val):
    color = ''
    if isinstance(val, str):
        if 'Critical' in val:
            color = 'background-color: red; color: white'
        elif 'Warning' in val:
            color = 'background-color: yellow; color: black'
        elif 'Healthy' in val:
            color = 'background-color: green; color: white'
    return color

if raw is not None:
    raw = pd.read_excel(raw)

    # Selecting necessary columns
    raw = raw[["Call Date Time", "Line", "Machine", "Machine No.",
            "Waiting Time (mins.)", "Fixing Time Duration (mins.)",
            "Total Loss Time"]]

    # Rename call date column and convert to datetime
    raw["Call Date Time"] = pd.to_datetime(raw["Call Date Time"], errors='coerce')

    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        line_selection = st.selectbox("Select Line:", raw["Line"].unique())
        raw = raw[raw["Line"] == line_selection]

    with filter_col2:
        machine_selection = st.selectbox("Select Machine:", raw["Machine"].unique())
        raw = raw[raw["Machine"] == machine_selection]

    with filter_col3:
        date_range = st.date_input("Select Date Range:", [])
        if len(date_range) == 2:
            start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            raw = raw[(raw["Call Date Time"] >= start_date) & (raw["Call Date Time"] <= end_date)]
            num_days = (end_date - start_date).days + 1
        else:
            num_days = 0

    # Aggregate data per machine
    machine_no_df = raw.groupby("Machine No.", as_index=True).agg(
        Total_Defects=("Machine No.", "count"),
        Total_Waiting_Time=("Waiting Time (mins.)", "sum"),
        Total_Fixing_Time=("Fixing Time Duration (mins.)", "sum"),
        Total_Loss_Time=("Total Loss Time", "sum")
    ).round(2)

    # Display overview
    st.subheader("Machine Number Summary:")
    st.dataframe(machine_no_df, use_container_width=True)

    if not machine_no_df.empty:
        if num_days > 0:
            machine_no_df["Avg_Daily_Defects"] = (machine_no_df["Total_Defects"] / num_days).round(2)
            machine_no_df["Avg_Waiting_Time"] = (machine_no_df["Total_Waiting_Time"] / num_days).round(2)
            machine_no_df["Avg_Fixing_Time"] = (machine_no_df["Total_Fixing_Time"] / num_days).round(2)
            machine_no_df["Avg_Loss_Time"] = (machine_no_df["Total_Loss_Time"] / num_days).round(2)

            health_statuses = {}
            for category in ["Avg_Daily_Defects", "Avg_Waiting_Time", "Avg_Fixing_Time", "Avg_Loss_Time"]:
                health_values = machine_no_df[category].dropna()
                if not health_values.empty:
                    q1 = np.percentile(health_values, 25)
                    q3 = np.percentile(health_values, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    def categorize(value):
                        if value < lower_bound:
                            return "Healthy"
                        elif lower_bound <= value < np.percentile(health_values, 50):
                            return "Healthy"
                        elif value >= upper_bound:
                            return "Critical"
                        else:
                            return "Warning"

                    health_statuses[category] = health_values.apply(categorize)

            for category, statuses in health_statuses.items():
                machine_no_df[f"{category}_Status"] = statuses

            machine_health_df = machine_no_df[[
                "Avg_Daily_Defects", "Avg_Daily_Defects_Status",
                "Avg_Waiting_Time", "Avg_Waiting_Time_Status",
                "Avg_Fixing_Time", "Avg_Fixing_Time_Status",
                "Avg_Loss_Time", "Avg_Loss_Time_Status"
            ]]

            st.subheader("Machine Health Status")
            status_cols = [col for col in machine_health_df.columns if col.endswith("_Status")]
            st.dataframe(machine_health_df.style.applymap(highlight_status, subset=status_cols), use_container_width=True)
        else:
            st.info("👉 Set a date range above to categorize machine health.")
    else:
        st.warning("No data available for the selected filters. Please adjust your selections.")

    # Machine Diagnosis
    selected_machine_no = st.selectbox("Select machine no. for complete diagnosis:", machine_no_df.index)

    if selected_machine_no:
        machine_data = raw[raw["Machine No."] == selected_machine_no]

        defect_count = len(machine_data)
        avg_daily_defect = round(defect_count / num_days if num_days > 0 else 0, 2)
        total_wait_time = round(machine_data["Waiting Time (mins.)"].sum(), 2)
        avg_daily_wait_time = round(total_wait_time / num_days if num_days > 0 else 0, 2)
        total_fixing_time = round(machine_data["Fixing Time Duration (mins.)"].sum(), 2)
        avg_daily_fixing_time = round(total_fixing_time / num_days if num_days > 0 else 0, 2)
        total_loss_time = round(machine_data["Total Loss Time"].sum(), 2)
        avg_daily_loss_time = round(total_loss_time / num_days if num_days > 0 else 0, 2)

        if all(col in machine_no_df.columns for col in [
            "Avg_Daily_Defects_Status", "Avg_Waiting_Time_Status",
            "Avg_Fixing_Time_Status", "Avg_Loss_Time_Status"
        ]):
            defect_category = machine_no_df.loc[selected_machine_no, "Avg_Daily_Defects_Status"]
            waiting_time_category = machine_no_df.loc[selected_machine_no, "Avg_Waiting_Time_Status"]
            fixing_time_category = machine_no_df.loc[selected_machine_no, "Avg_Fixing_Time_Status"]
            loss_time_category = machine_no_df.loc[selected_machine_no, "Avg_Loss_Time_Status"]
        else:
            st.warning("Health status not yet computed. Please ensure a date range is set.")
            defect_category = waiting_time_category = fixing_time_category = loss_time_category = "N/A"

        st.subheader(f"Machine Diagnosis for {line_selection} {machine_selection} Machine No. {selected_machine_no}")
        machine_diag_spacer, machine_diag_col1, machine_diag_col2 = st.columns([0.2, 2, 2])

        with machine_diag_col1:
            st.markdown(f"""
            - **Selected Date Range:** {num_days} days  
            - **Defect Count:** {defect_count}  
            - **Average Daily Defect:** {avg_daily_defect}  
            - **Total Caused Waiting Time:** {total_wait_time} mins  
            - **Average Daily Waiting Time:** {avg_daily_wait_time} mins  
            - **Total Caused Fixing Time:** {total_fixing_time} mins  
            - **Average Daily Fixing Time:** {avg_daily_fixing_time} mins  
            """)

        with machine_diag_col2:
            st.markdown(f"""
            - **Total Caused Loss Time:** {total_loss_time} mins  
            - **Average Daily Loss Time:** {avg_daily_loss_time} mins  
            - **Category based on Defect Count:** {defect_category}  
            - **Category based on Waiting Time:** {waiting_time_category}  
            - **Category based on Fixing Time:** {fixing_time_category}  
            - **Category based on Loss Time:** {loss_time_category}  
            """)

    st.markdown("")
    st.markdown("Improvement Suggestion/s:")
    if not machine_no_df.empty:
        if defect_category == "Critical":
            st.markdown("- Check machine process capability and conduct necessary maintenance.")
        elif defect_category == "Warning":
            st.markdown("- Strictly follow machine preventive maintenance schedule.")
        
        if waiting_time_category == "Critical":
            st.markdown("- Assign enough technicians on the line.")
        elif waiting_time_category == "Warning":
            st.markdown("- Continue monitoring technician availability.")
        
        if fixing_time_category == "Critical":
            st.markdown("- Provide enough training to the technicians as soon as possible.")
        elif fixing_time_category == "Warning":
            st.markdown("- Continue technician skill-checking.")
        
        if defect_category == "Healthy" and waiting_time_category == "Healthy" and fixing_time_category == "Healthy":
            st.markdown("- Keep machine status healthy in all categories.")
    else:
        st.warning("No data available.")

# Footer
st.write("_________________________________________________________")
st.write("PE1 | K. Katigbak | rev. 01 | 2025")

# Load and apply custom CSS
with open("PE1_apps/style.css") as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
