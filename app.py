import streamlit as st
import pandas as pd

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Beauhurst Funding Explorer", page_icon="💷", layout="wide")
st.title("💷 Regional Funding Explorer")
st.markdown("Explore funding rounds, top investors, and leading advisors across regions.")

# ==========================================
# DATA LOADING & CACHING
# ==========================================
# @st.cache_data ensures the massive CSV is only loaded once, making the app lightning fast
@st.cache_data
def load_data():
    # Update this path if running locally
    df = pd.read_csv('Full Cohort Funding Rounds.csv', low_memory=False)
    
    # Standardize the Equity flag
    if 'IsEquityRound' in df.columns:
        df['IsEquityRound'] = df['IsEquityRound'].astype(str).str.lower().replace({'true': True, 'false': False, 'nan': False})
        df['IsEquityRound'] = df['IsEquityRound'].fillna(False).astype(bool)
    
    # Consolidate Scottish regions
    df['Region_Clean'] = df['Company Region'].astype(str).str.lower().str.strip()
    scottish_regions = ['aberdeen', 'east of scotland', 'highlands and islands', 'south of scotland', 'tayside', 'west of scotland']
    df.loc[df['Region_Clean'].isin(scottish_regions), 'Region_Clean'] = 'scotland'
    
    # Title-case the regions for the dropdown menu
    df['Region_Display'] = df['Region_Clean'].str.title()
    
    return df

df = load_data()

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.header("🔍 Filter Data")

# 1. Region Filter
all_regions = sorted([r for r in df['Region_Display'].unique() if r != 'Nan' and r != 'None'])
selected_region = st.sidebar.selectbox("Select Region", ["All Regions"] + all_regions)

# 2. Equity Filter
funding_type = st.sidebar.radio("Funding Type", ["All Funding", "Equity Only", "Non-Equity (Grants, Debt, etc.)"])

# Apply Filters
df_filtered = df.copy()

if selected_region != "All Regions":
    df_filtered = df_filtered[df_filtered['Region_Display'] == selected_region]

if funding_type == "Equity Only":
    df_filtered = df_filtered[df_filtered['IsEquityRound'] == True]
elif funding_type == "Non-Equity (Grants, Debt, etc.)":
    df_filtered = df_filtered[df_filtered['IsEquityRound'] == False]

# ==========================================
# MAIN DASHBOARD: KPIs
# ==========================================
st.markdown("---")
col1, col2, col3 = st.columns(3)

total_investment = df_filtered.drop_duplicates('RoundIDKey')['RoundAmountGBP_total'].sum()
total_deals = df_filtered['RoundIDKey'].nunique()
total_companies = df_filtered['CompanyName'].nunique()

col1.metric("Total Investment", f"£{total_investment:,.0f}")
col2.metric("Total Deals", f"{total_deals:,}")
col3.metric("Companies Funded", f"{total_companies:,}")

st.markdown("---")

# ==========================================
# TWO-COLUMN LAYOUT: FUNDERS & ADVISORS
# ==========================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🏆 Top Funders (By Deals)")
    
    valid_inv = df_filtered[~df_filtered['InvestorName'].isin(['Unknown', ''])].dropna(subset=['InvestorName']).copy()
    if not valid_inv.empty:
        if 'InvestorManager' in valid_inv.columns:
            valid_inv['InvestorManager'] = valid_inv['InvestorManager'].fillna('').astype(str).str.strip()
            mask_no_manager = valid_inv['InvestorManager'] == ''
            valid_inv.loc[mask_no_manager, 'InvestorManager'] = valid_inv.loc[mask_no_manager, 'InvestorName']
        else:
            valid_inv['InvestorManager'] = valid_inv['InvestorName']

        top_funders = valid_inv.groupby('InvestorManager').agg(
            NumberOfDeals=('RoundIDKey', 'nunique'),
            TotalAmountInvested=('InvestorAmountGBP', 'sum')
        ).reset_index()

        top_funders = top_funders.sort_values(by='NumberOfDeals', ascending=False).head(20)
        
        # Display as an interactive dataframe
        st.dataframe(top_funders, use_container_width=True, hide_index=True)
    else:
        st.info("No investor data available for these filters.")

with col_right:
    st.subheader("🥇 Top Advisors")
    
    if 'Advisors' in df_filtered.columns:
        df_adv = df_filtered[['RoundIDKey', 'CompanyName', 'Advisors']].drop_duplicates().dropna(subset=['Advisors']).copy()
        df_adv = df_adv[df_adv['Advisors'].astype(str).str.strip() != '']

        if not df_adv.empty:
            df_adv['Advisors'] = df_adv['Advisors'].astype(str).str.replace(';', ',').str.split(',')
            df_adv = df_adv.explode('Advisors')
            df_adv['Advisors'] = df_adv['Advisors'].str.strip()
            df_adv = df_adv[df_adv['Advisors'] != '']

            top_advisors = df_adv.groupby('Advisors').agg(
                NumberOfDeals=('RoundIDKey', 'nunique'),
                UniqueCompanies=('CompanyName', 'nunique')
            ).reset_index()

            top_advisors = top_advisors.sort_values(by=['NumberOfDeals', 'UniqueCompanies'], ascending=[False, False]).head(10)
            
            st.dataframe(top_advisors, use_container_width=True, hide_index=True)
        else:
            st.info("No advisor data available for these filters.")
    else:
        st.info("Advisors column not found.")

# ==========================================
# RAW DATA VIEWER
# ==========================================
st.markdown("---")
st.subheader("📄 Raw Data Explorer")
with st.expander("Click to view and download the filtered data"):
    # Select a few key columns to keep it readable
    display_cols = ['CompanyName', 'RoundDate', 'RoundAmountGBP_total', 'Company Region', 'FormOfFunding', 'InvestorName']
    available_cols = [c for c in display_cols if c in df_filtered.columns]
    
    st.dataframe(df_filtered[available_cols].head(100), use_container_width=True, hide_index=True)
    st.caption("Showing top 100 rows. Use the filter menu to drill down further.")
