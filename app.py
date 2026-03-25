import re
import streamlit as st
import pandas as pd
import numpy as np
from itertools import zip_longest
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Beauhurst Funding Explorer", page_icon="💷", layout="wide")

# ==========================================
# DATA PROCESSING ENGINE (The "Brain")
# ==========================================
def get_direct_gdrive_link(url):
    """Converts a standard Google Drive or Sheets share link into a direct CSV download link."""
    # Match a Google Sheets link
    sheets_match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url)
    if sheets_match:
        return f"https://docs.google.com/spreadsheets/d/{sheets_match.group(1)}/export?format=csv"
    
    # Match a standard Google Drive file link
    drive_match = re.search(r'/file/d/([a-zA-Z0-9-_]+)', url)
    if drive_match:
        return f"https://drive.google.com/uc?id={drive_match.group(1)}&export=download"
        
    # If it doesn't match Google's format, just return the raw URL and hope for the best
    return url

def process_beauhurst_file(file):
    """Reads a raw Beauhurst file, detects its format, and unpivots it cleanly."""
    raw = pd.read_csv(file, low_memory=False)
    
    # Standardize Column Names
    COL = {
        "company_name": "Company name", "ch_id": "(Company) Companies House ID",
        "inc_date": "(Company) Incorporation date (Companies House)",
        "region_head": "(Company) Head Office Address - Region", "region_reg": "(Company) Registered Address - Region",
        "local_auth_head": "(Company) Head Office Address - Local Authority", "local_auth_reg": "(Company) Registered Address - Local Authority",
        "round_date": "Deal date", "round_amount_gbp": "Amount raised (converted to GBP)",
        "advisors": "Advisors - Advisor Name", "purpose": "Purpose",
        "is_equity_round": "Form(s) of funding - Equity", "funding_forms": "Form(s) of funding",
        "company_url": "Beauhurst company URL", "deal_url": "Beauhurst deal URL"
    }

    raw_cols_lower = {str(c).strip().lower(): c for c in raw.columns}

    # Safely inject missing columns
    for key, c in COL.items():
        if c not in raw.columns:
            # Check if there's a close match in lowercase first
            if str(c).lower() in raw_cols_lower:
                raw = raw.rename(columns={raw_cols_lower[str(c).lower()]: c})
            else:
                raw[c] = False if key == "is_equity_round" else ""

    # Clean Dates & Numbers
    raw[COL["inc_date"]] = pd.to_datetime(raw[COL["inc_date"]], dayfirst=True, format='mixed', errors="coerce")
    raw[COL["round_date"]] = pd.to_datetime(raw[COL["round_date"]], dayfirst=True, format='mixed', errors="coerce")
    raw[COL["round_amount_gbp"]] = pd.to_numeric(raw[COL["round_amount_gbp"]], errors="coerce")

    # Consolidate Region and Local Authority
    raw["Company Region"] = raw[COL["region_head"]].fillna(raw[COL["region_reg"]])
    raw["Company Local Authority"] = raw[COL["local_auth_head"]].fillna(raw[COL["local_auth_reg"]])

    # Equity Flag & Form of Funding
    if "Form(s) of funding" in raw.columns and not raw["Form(s) of funding"].astype(str).str.strip().eq("").all():
        raw['IsEquityRound'] = raw[COL["funding_forms"]].astype(str).str.lower().str.contains('equity', na=False)
    else:
        raw['IsEquityRound'] = raw[COL["is_equity_round"]].astype(str).str.lower().str.strip().replace({'true': True, 'false': False, 'nan': False, 'yes': True, 'no': False}).fillna(False).astype(bool)
    
    raw['FormOfFunding'] = raw[COL["funding_forms"]].astype(str).str.strip()

    # Generate Keys
    raw = raw.reset_index(names="__rowid")
    raw["RoundIDKey"] = raw[COL["company_name"]].astype(str).str.strip().str.lower() + "|" + raw[COL["round_date"]].astype(str) + "|" + raw[COL["round_amount_gbp"]].astype(str)

    # DETECT FORMAT: Grouped (Extras) vs Standard (Primary/Pipeline)
    is_grouped = "fundraising investors - name" in raw_cols_lower
    
    long_rows = []
    
    if is_grouped:
        # --- GROUPED UNPIVOT LOGIC ---
        inv_name_col = raw_cols_lower.get("fundraising investors - name", "")
        inv_type_col = raw_cols_lower.get("fundraising investors - fund type", "")
        inv_mgr_col  = raw_cols_lower.get("fundraising investors - fund manager", "")
        inv_cnty_col = raw_cols_lower.get("fundraising investors - head office country", "")
        inv_amt_col  = raw_cols_lower.get("fundraising investors - amount contributed (converted to gbp)", "")

        def clean_split(val):
            return [x.strip() for x in str(val).split(',')] if str(val).lower() not in ['nan', 'none', ''] else []

        for index, row in raw.iterrows():
            base_data = {
                "CompanyName": row[COL["company_name"]], "CompaniesHouseID": row[COL["ch_id"]],
                "IncorporationDate": row[COL["inc_date"]], "RoundDate": row[COL["round_date"]],
                "RoundAmountGBP_total": row[COL["round_amount_gbp"]], "Company Region": row["Company Region"],
                "Company Local Authority": row["Company Local Authority"], "RoundIDKey": row["RoundIDKey"],
                "Advisors": row[COL["advisors"]], "Purpose": row[COL["purpose"]],
                "FormOfFunding": row["FormOfFunding"], "IsEquityRound": row["IsEquityRound"],
                "BeauhurstCompanyURL": row.get(COL["company_url"], ""), "BeauhurstDealURL": row.get(COL["deal_url"], "")
            }

            names, types, mgrs, cntys, amts = (clean_split(row.get(c, "")) if c else [] for c in [inv_name_col, inv_type_col, inv_mgr_col, inv_cnty_col, inv_amt_col])

            if not names and not types and not mgrs:
                base_data.update({"InvestorName": np.nan, "InvestorType": "", "InvestorManager": "", "InvestorCountry": "", "InvestorAmountGBP": np.nan})
                long_rows.append(base_data)
                continue

            for n, t, m, c, a in zip_longest(names, types, mgrs, cntys, amts, fillvalue=""):
                row_data = base_data.copy()
                row_data.update({
                    "InvestorName": n if n.lower() != '(no value)' and n else np.nan,
                    "InvestorType": t if t.lower() != '(no value)' else "",
                    "InvestorManager": m if m.lower() != '(no value)' else "",
                    "InvestorCountry": c if c.lower() != '(no value)' else "",
                    "InvestorAmountGBP": pd.to_numeric(a, errors="coerce") if a and a.lower() != '(no value)' else np.nan
                })
                long_rows.append(row_data)
        long = pd.DataFrame(long_rows)
        
    else:
        # --- STANDARD UNPIVOT LOGIC ---
        long_frames = []
        base_cols = [COL["company_name"], COL["ch_id"], COL["inc_date"], COL["round_date"], COL["round_amount_gbp"], "Company Region", "Company Local Authority", "RoundIDKey", COL["advisors"], COL["purpose"], "IsEquityRound", "FormOfFunding", COL["company_url"], COL["deal_url"]]
        
        for i in range(1, 51):
            expected_cols_lower = {
                "InvestorName": f"fundraising investors {i} - name",
                "InvestorType": f"fundraising investors {i} - fund type",
                "InvestorCountry": f"fundraising investors {i} - head office country",
                "InvestorAmountGBP": f"fundraising investors {i} - amount contributed (converted to gbp)",
                "InvestorManager": f"fundraising investors {i} - fund manager"
            }
            cols_i = {key: raw_cols_lower[exp] for key, exp in expected_cols_lower.items() if exp in raw_cols_lower}
            if not cols_i: continue

            part = raw[base_cols + list(cols_i.values())].copy()
            part = part.rename(columns={v: k for k, v in cols_i.items()})
            
            for k in ["InvestorName", "InvestorType", "InvestorCountry", "InvestorAmountGBP", "InvestorManager"]:
                if k not in part.columns: part[k] = np.nan
            for str_col in ["InvestorName", "InvestorType", "InvestorCountry", "InvestorManager"]:
                part[str_col] = part[str_col].astype(str).str.strip().replace({"nan": "", "NaN": "", "None": ""})
            
            part["InvestorAmountGBP"] = pd.to_numeric(part["InvestorAmountGBP"], errors="coerce")
            
            mask_valid = part["InvestorName"].ne("") | part["InvestorType"].ne("") | part["InvestorCountry"].ne("") | part["InvestorManager"].ne("") | part["InvestorAmountGBP"].fillna(0).gt(0)
            if not part[mask_valid].empty:
                long_frames.append(part[mask_valid])
        
        if long_frames:
            long = pd.concat(long_frames, ignore_index=True)
            long = long.rename(columns={COL["company_name"]: "CompanyName", COL["ch_id"]: "CompaniesHouseID", COL["inc_date"]: "IncorporationDate", COL["round_date"]: "RoundDate", COL["round_amount_gbp"]: "RoundAmountGBP_total", COL["advisors"]: "Advisors", COL["purpose"]: "Purpose", COL["company_url"]: "BeauhurstCompanyURL", COL["deal_url"]: "BeauhurstDealURL"})
        else:
            long = pd.DataFrame(columns=["CompanyName","CompaniesHouseID","IncorporationDate","RoundDate","RoundAmountGBP_total","Company Region","Company Local Authority","RoundIDKey","InvestorName","InvestorType","InvestorCountry","InvestorAmountGBP","Advisors","Purpose","FormOfFunding","IsEquityRound","BeauhurstCompanyURL","BeauhurstDealURL","InvestorManager"])

    # Final cleanup before return
    long["InvestorAmountGBP"] = pd.to_numeric(long["InvestorAmountGBP"], errors="coerce")
    has_signal = long["RoundDate"].notna() | long["RoundAmountGBP_total"].fillna(0).gt(0)
    long.loc[has_signal & (long["InvestorName"].isna() | (long["InvestorName"] == "")), "InvestorName"] = "Unknown"
    
    return long

# ==========================================
# APP UI & STATE MANAGEMENT
# ==========================================
st.title("💷 Ultimate Funding Explorer")
st.markdown("Upload raw Beauhurst exports. We'll automatically clean, unpivot, and merge them into an interactive dashboard.")

# Initialize session state for our dataset
if 'master_df' not in st.session_state:
    st.session_state.master_df = None

# --- SIDEBAR: DATA UPLOAD & PROCESSING ---
with st.sidebar:
    st.header("📂 1. Upload Data")
    
    # Option A: File Upload
    uploaded_files = st.file_uploader("Upload Raw Beauhurst CSVs", type=['csv'], accept_multiple_files=True)
    
    st.markdown("**OR**")
    
    # Option B: URL Paste
    gdrive_url = st.text_input("Paste a Google Drive / Sheets Link", placeholder="https://docs.google.com/spreadsheets/...")
    st.caption("⚠️ *Note: The Google file must be set to 'Anyone with the link can view'.*")
    
    if st.button("Process & Merge Files", type="primary"):
        if uploaded_files or gdrive_url:
            with st.spinner("Processing files... this might take a minute."):
                all_dfs = []
                
                # 1. Process directly uploaded files
                if uploaded_files:
                    for file in uploaded_files:
                        all_dfs.append(process_beauhurst_file(file))
                
                # 2. Process the Google Drive URL
                if gdrive_url:
                    try:
                        direct_url = get_direct_gdrive_link(gdrive_url)
                        # process_beauhurst_file accepts URLs just as easily as local files!
                        all_dfs.append(process_beauhurst_file(direct_url)) 
                    except Exception as e:
                        st.error(f"Could not read from URL. Ensure it's a public CSV/Sheet. Error: {e}")
                        st.stop()
                
                # 3. Merge and deduplicate
                if all_dfs:
                    master_df = pd.concat(all_dfs, ignore_index=True)
                    master_df = master_df.drop_duplicates(subset=["RoundIDKey", "InvestorName"], keep="first")
                    
                    # Consolidate Scottish Regions
                    master_df['Region_Clean'] = master_df['Company Region'].astype(str).str.lower().str.strip()
                    master_df.loc[master_df['Region_Clean'].isin(['aberdeen', 'east of scotland', 'highlands and islands', 'south of scotland', 'tayside', 'west of scotland']), 'Region_Clean'] = 'scotland'
                    master_df['Region_Display'] = master_df['Region_Clean'].str.title()
                    
                    # Save to state
                    st.session_state.master_df = master_df
                    st.success("✅ Data processed successfully!")
        else:
            st.warning("Please upload a file or paste a link to begin.")

# Stop execution if no data is loaded yet
if st.session_state.master_df is None:
    st.info("👈 Upload your raw Beauhurst CSV files in the sidebar and click 'Process & Merge Files' to begin.")
    st.stop()

df = st.session_state.master_df.copy()

# --- SIDEBAR: FILTERS ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 2. Filter Dashboard")

# Search Bar
search_query = st.sidebar.text_input("Search (Company, Investor, or Advisor)", "")

# Region Filter
all_regions = sorted([r for r in df['Region_Display'].unique() if r != 'Nan' and r != 'None'])
selected_region = st.sidebar.selectbox("Select Region", ["All Regions"] + all_regions)

# Equity Filter
funding_type = st.sidebar.radio("Funding Type", ["All Funding", "Equity Only", "Non-Equity (Grants, Debt, etc.)"])

# APPLY FILTERS
if selected_region != "All Regions":
    df = df[df['Region_Display'] == selected_region]

if funding_type == "Equity Only":
    df = df[df['IsEquityRound'] == True]
elif funding_type == "Non-Equity (Grants, Debt, etc.)":
    df = df[df['IsEquityRound'] == False]

if search_query:
    search_lower = search_query.lower()
    df = df[
        df['CompanyName'].astype(str).str.lower().str.contains(search_lower) |
        df['InvestorName'].astype(str).str.lower().str.contains(search_lower) |
        df['Advisors'].astype(str).str.lower().str.contains(search_lower)
    ]

# ==========================================
# MAIN DASHBOARD: KPIs & CHARTS
# ==========================================
st.markdown("---")
col1, col2, col3 = st.columns(3)

df_unique_rounds = df.drop_duplicates('RoundIDKey')
total_inv = df_unique_rounds['RoundAmountGBP_total'].sum()
total_deals = df_unique_rounds['RoundIDKey'].nunique()
total_comps = df_unique_rounds['CompanyName'].nunique()

col1.metric("Total Investment", f"£{total_inv:,.0f}")
col2.metric("Total Deals", f"{total_deals:,}")
col3.metric("Companies Funded", f"{total_comps:,}")

# -- INTERACTIVE TIMELINE --
st.markdown("### 📈 Investment Over Time")
# Create Year-Quarter column for clean charting
df_unique_rounds['Quarter'] = df_unique_rounds['RoundDate'].dt.to_period('Q').astype(str)
timeline_data = df_unique_rounds.groupby('Quarter')['RoundAmountGBP_total'].sum().reset_index()

if not timeline_data.empty:
    fig = px.bar(timeline_data, x='Quarter', y='RoundAmountGBP_total', 
                 labels={'RoundAmountGBP_total': 'Total Investment (£)', 'Quarter': 'Quarter'},
                 color_discrete_sequence=['#1f77b4'])
    fig.update_layout(xaxis_tickangle=-45, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough dated round data to plot timeline.")

# ==========================================
# LEADERBOARDS: FUNDERS & ADVISORS
# ==========================================
st.markdown("---")
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🏆 Top Funders")
    valid_inv = df[~df['InvestorName'].isin(['Unknown', ''])].dropna(subset=['InvestorName']).copy()
    if not valid_inv.empty:
        valid_inv['InvestorManager'] = valid_inv.get('InvestorManager', valid_inv['InvestorName']).fillna('').astype(str).str.strip()
        mask_no_manager = valid_inv['InvestorManager'] == ''
        valid_inv.loc[mask_no_manager, 'InvestorManager'] = valid_inv.loc[mask_no_manager, 'InvestorName']

        top_funders = valid_inv.groupby('InvestorManager').agg(
            Deals=('RoundIDKey', 'nunique'),
            Total_Capital=('InvestorAmountGBP', 'sum')
        ).reset_index().sort_values(by='Deals', ascending=False).head(20)
        
        st.dataframe(top_funders, use_container_width=True, hide_index=True)
    else:
        st.info("No investor data available.")

with col_right:
    st.subheader("🥇 Top Advisors")
    if 'Advisors' in df.columns:
        df_adv = df[['RoundIDKey', 'CompanyName', 'Advisors']].drop_duplicates().dropna(subset=['Advisors']).copy()
        df_adv = df_adv[df_adv['Advisors'].astype(str).str.strip() != '']

        if not df_adv.empty:
            df_adv['Advisors'] = df_adv['Advisors'].astype(str).str.replace(';', ',').str.split(',')
            df_adv = df_adv.explode('Advisors')
            df_adv['Advisors'] = df_adv['Advisors'].str.strip()
            df_adv = df_adv[df_adv['Advisors'] != '']

            top_advisors = df_adv.groupby('Advisors').agg(
                Deals=('RoundIDKey', 'nunique'),
                Companies=('CompanyName', 'nunique')
            ).reset_index().sort_values(by=['Deals', 'Companies'], ascending=[False, False]).head(10)
            
            st.dataframe(top_advisors, use_container_width=True, hide_index=True)
        else:
            st.info("No advisor data available.")

# ==========================================
# DATA DOWNLOAD
# ==========================================
st.markdown("---")
st.subheader("📄 Filtered Data")

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df)

st.download_button(
    label="⬇️ Download Processed Dataset (CSV)",
    data=csv,
    file_name='processed_beauhurst_rounds.csv',
    mime='text/csv',
)

st.dataframe(df[['CompanyName', 'RoundDate', 'RoundAmountGBP_total', 'Company Region', 'FormOfFunding', 'InvestorName']].head(50), use_container_width=True)
