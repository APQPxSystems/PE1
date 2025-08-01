# Import libraries
import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import re
import io

# Page configurations
st.set_page_config(page_title='Andon Data Mod | kentjk', layout='wide')

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
    <div class="footer-center">&copy; Kent Katigbak | EE1 | rev. 1.0 | 2025</div>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)

st.title('EE1 Andon Data Cleaning and Modification')
st.divider()

# File uploader
df1 = st.file_uploader("Upload Andon Logs xlsx", type='xlsx')

if df1 is not None:
    
    with st.spinner('Processing data... Please wait...'):
    
        df1 = pd.read_excel(df1)
        df2 = pd.read_excel("Andon_Data_Cleaning/Problem_Category.xlsx")

        # Function to adjust to production day and get shift
        def get_shift_and_adjusted_date(ts):
            if ts.time() >= pd.to_datetime("06:00").time() and ts.time() < pd.to_datetime("18:00").time():
                return ts.replace(hour=6, minute=0, second=0, microsecond=0), "DS"
            elif ts.time() >= pd.to_datetime("18:00").time():
                return ts.replace(hour=18, minute=0, second=0, microsecond=0), "NS"
            else:
                adjusted_date = (ts - pd.Timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
                return adjusted_date, "NS"

        # Adjust 'Call Date Time' for shift logic
        df1[['Adjusted Call Date Time', 'Shift']] = df1['Call Date Time'].apply(
            lambda ts: pd.Series(get_shift_and_adjusted_date(ts))
        )

        # Use adjusted datetime for derived columns
        df1["Call Year"] = df1["Adjusted Call Date Time"].dt.year
        df1["Call Month"] = df1["Adjusted Call Date Time"].dt.month
        df1["Call Day"] = df1["Adjusted Call Date Time"].dt.day
        df1["Call Date No Time"] = df1["Adjusted Call Date Time"].dt.date
        df1["Call Time"] = df1["Call Date Time"].dt.time
        df1["Total Loss Time"] = df1["Waiting Time (mins.)"] + df1["Fixing Time Duration (mins.)"]
        df1["Loss Time Instances"] = 1

        # Match Columns for problem category
        col_df1 = "Problem"
        col_df2 = "problem"

        # Function to find best match
        def get_best_match(value, choices, threshold=80):
            match = process.extractOne(value, choices, scorer=fuzz.token_set_ratio)
            if match and match[1] >= threshold:
                return match[0]
            return None

        # Get the unique values in df2 for faster matching
        choices = df2[col_df2].unique()

        # Apply fuzzy match
        df1['Matched_Value'] = df1[col_df1].apply(lambda x: get_best_match(str(x), choices))

        # Merge df1 with df2 using the matched value
        merged = pd.merge(df1, df2, left_on='Matched_Value', right_on=col_df2, how='left')

        # Add hourly column
        merged["Hour"] = pd.to_datetime(merged["Call Date Time"]).dt.strftime('%H:00')

        # Some andon categories are not applicable to specific machines
        # Define affected machine names
        affected_machines = [
            "CASTING MACHINE", "CASTING",
            "TWISTING MACHINE", "TWISTING SECONDARY",
            "TWISTING PRIMARY", "SECONDARY TWISTING",
            "V-TYPE TWISTING"
        ]

        # Override problem_category where applicable
        mask = (
            merged['Machine'].str.upper().isin([m.upper() for m in affected_machines]) &
            (merged['problem_category'] == "APPLICATOR PROBLEM")
        )
        merged.loc[mask, 'problem_category'] = "MACHINE PROBLEM"

        # Add the Section column
        def get_section(line):
            if line == "HONDA TKRA-First Process":
                return "S6 Honda TKRA"
            elif line == "MAZDA J12-First Process":
                return "S3.1 Mazda J12"
            elif line == "DAIHATSU D01L-First Process":
                return "S4 Daihatsu"
            elif line == "HONDA TKRA-Secondary Process":
                return "S6 Honda TKRA"
            elif line == "MAZDA MERGE-Secondary Process":
                return "S3 Mazda Merge"
            elif line == "HONDA 3TOA-First Process":
                return "S7 Honda 3T0A"
            elif line == "SUZUKI-First Process":
                return "S1 Suzuki"
            elif line == "SUBARU-Secondary Process":
                return "S5 Subaru"
            elif line == "TUBE CUTTING---":
                return "VO Cutting"
            elif line == "INITIAL BATTERY---":
                return "INITIAL BATTERY"
            elif line == "DAIHATSU D01L-Secondary Process":
                return "S4 Daihatsu"
            elif line == "SUBARU-First Process":
                return "S5 Subaru"
            elif line == "HONDA 3TOA-Secondary Process":
                return "S7 Honda 3T0A"
            elif line == "MAZDA MERGE-First Process":
                return "S3 Mazda Merge"
            elif line == "SUZUKI-Secondary Process":
                return "S1 Suzuki"
            elif line == "Honda T20A-Secondary Process":
                return "S8 Honda T20A"
            elif line == "Honda T20A-First Process":
                return "S8 Honda T20A"
            elif line == "TUBE MAKING---":
                return "VO Making"
            elif line == "PRACTICE TRAINING CENTER-Secondary Process":
                return "Practice Training"
            elif line == "MAZDA J12-Secondary Process":
                return "S3.1 Mazda J12"
            elif line == "TOYOTA-Secondary Process":
                return "S2 Toyota"
            elif line == "TOYOTA-First Process":
                return "S2 Toyota"
            elif line == "SUZUKI-Suzuki- NEW TRD":
                return "S1 Suzuki"
            elif line == "HONDA OLD-First Process":
                return "S7 Honda 3T0A"
            elif line == "PRACTICE TRAINING CENTER-First Process":
                return "Practice Training"
            else:
                return None  # Default value if no match

        # Create the new Section column
        merged['Section'] = merged['Line'].apply(get_section)

        # Drop Practice Training, VO Cutting, VO Making in Section Column
        merged = merged[~merged['Section'].isin(['Practice Training', 'VO Cutting', 'VO Making'])]

        # Drop Machine Accessories (IT concern) in Problem Column
        merged = merged[~merged['Problem'].str.contains('Machine Accessories', case=False, na=False)]

        # Drop forget/ forgot in Solution Column
        keywords = ['forgot', 'forget']

        def contains_forgot(text, keywords):
            cleaned = re.sub(r'\W+', '', str(text)).lower()
            return any(kw in cleaned for kw in keywords)

        merged = merged[~merged['Solution'].apply(lambda x: contains_forgot(x, keywords))]
            
        # Drop duplicates based on selected columns only
        columns_to_check = [
            'Call Date Time', 'Start Time', 'End Time', 'Technician'
        ]
        merged_final = merged.drop_duplicates(subset=columns_to_check).reset_index(drop=True)

        # Drop Sundays
        # Convert column to datetime (safe even if already converted)
        merged_final['Call Date No Time'] = pd.to_datetime(merged_final['Call Date No Time'], errors='coerce')

        # Drop Sundays (dayofweek == 6)
        merged_final = merged_final[merged_final['Call Date No Time'].dt.dayofweek != 6]

        # Convert the DataFrame to Excel in-memory
        def convert_df_to_excel(df):
            # Create an in-memory bytes buffer
            output = io.BytesIO()
            # Write the DataFrame to the buffer in Excel format
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            # Get the byte content of the buffer
            return output.getvalue()

        # Streamlit download button
        excel_file = convert_df_to_excel(merged_final)
        st.download_button(
            label="Download as Excel",
            data=excel_file,
            file_name="output_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
with open("Andon_Data_Cleaning/style.css") as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
