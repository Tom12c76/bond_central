import streamlit as st
import pandas as pd


def load_feather():
    # Your existing asset class filtering logic
    asset_class = 'Funds'
    df = st.session_state.df_filtered.copy()
    df = df.loc[df['L1'] == asset_class]
    # df['Peso'] = df['Controvalore EUR'] / df['Controvalore EUR'].sum()

    # Group by ISIN and Descrizione and sum Controvalore EUR and Peso
    df = df.groupby(['AC', 'Module', 'ISIN', 'Descrizione'], as_index=False) \
        .agg({'Controvalore EUR':'sum', 'Rebate':'median'})

    # Sort the DataFrame in descending order of Controvalore EUR
    df = df.sort_values(by='Controvalore EUR', ascending=False)
    return df

# Initialize session state DataFrames if not already present
if 'df_ric' not in st.session_state:
    st.session_state.df_ric = pd.DataFrame(columns=['Instrument', 'RIC'])
if 'df_rectype' not in st.session_state:
    st.session_state.df_rectype = pd.DataFrame(columns=['Instrument', 'RECORDTYPE'])
if 'df_hist' not in st.session_state:
    st.session_state.df_hist = pd.DataFrame()


# Radio button to select data source
data_source = st.sidebar.radio("Choose data source", ("Use Feather DB", "Load Excel file"))

# Load data based on the selected source
if data_source == "Use Feather DB":
    df = load_feather()
else:
    df = load_excel()

if df is None or df.empty:
    st.warning('Please load valid file')
