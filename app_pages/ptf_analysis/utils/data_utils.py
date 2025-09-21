import streamlit as st
import pandas as pd
import eikon as ek

# @st.cache_data
def load_feather_data(df_filtered):
    asset_class = 'Funds'
    df = df_filtered.copy()
    df = df.loc[df['L1'] == asset_class]
    df = df.groupby(['AC', 'Module', 'ISIN', 'Descrizione'], as_index=False) \
        .agg({'Controvalore EUR':'sum', 'Rebate':'median', 'Quant':'sum'})
    df = df.sort_values(by='Controvalore EUR', ascending=False)
    return df

# @st.cache_data
def load_excel_data(uploaded_file, df_full):
    df = pd.read_excel(uploaded_file)
    df = df.drop('Descrizione', axis=1)
    fund_info = df_full.copy()
    fund_info = fund_info.loc[
        fund_info['L1'] == 'Funds', ['AC', 'Module', 'ISIN', 'Descrizione', 'Rebate']].drop_duplicates()
    df = df.merge(fund_info, on='ISIN')
    return df

# @st.cache_data
def get_fund_hist_cached(rics, start, end):
    fund_hist, err = ek.get_data(rics,
                                 ['TR.FundNav.Date', 'TR.FundNav'],
                                 {'SDate': start.strftime('%Y-%m-%d'),
                                  'EDate': end.strftime('%Y-%m-%d'),
                                  'Curn': 'EUR'})
    fund_hist['Date'] = pd.to_datetime(fund_hist['Date']).dt.tz_localize(None)
    fund_hist = fund_hist.pivot(index='Date', columns=['Instrument'])
    fund_hist = fund_hist.droplevel(0, axis=1)
    fund_hist = fund_hist.sort_index(ascending=True)
    fund_hist = fund_hist.astype(float)
    return fund_hist
