import streamlit as st
import eikon as ek
import numpy as np
import pandas as pd
import plotly.express as px


st.set_page_config(layout='wide')

ek.set_app_key('cf2eaf5e3b3c42adba08b3c5c2002b6ced1e77d7')
cols = px.colors.qualitative.G10 * 10


# Fetch or use cached historical data
@st.cache_data
def get_fund_hist(rics, start, end):
    placeholder = st.empty()
    placeholder.info(f"Fetching data ***for {len(rics)} RIC's*** from Eikon...")
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
    placeholder.empty()
    return fund_hist


def get_calc():
    ptf_val = df_hist * df_funds['Quant']
    ptf_val['Ptf'] = ptf_val.sum(axis=1)

    logret = np.log(ptf_val / ptf_val.shift(1))

    return ptf_val, logret


def load_feather():
    # Your existing asset class filtering logic
    asset_class = 'Funds'
    df = st.session_state.df_filtered.copy()
    df = df.loc[df['L1'] == asset_class]

    # Group by ISIN and Descrizione and sum Controvalore EUR and Peso
    df = df.groupby(['AC', 'Module', 'ISIN', 'Descrizione'], as_index=False) \
        .agg({'Controvalore EUR': 'sum', 'Rebate': 'median', 'Quant': 'sum'})

    # Sort the DataFrame in descending order of Controvalore EUR
    df = df.sort_values(by='Controvalore EUR', ascending=False)
    return df


def load_excel():
    uploaded_file = st.sidebar.file_uploader("ISIN, (Descrizione), Ctv EUR", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df = df.drop('Descrizione', axis=1)
        # df['Peso'] = df['Controvalore EUR'] / df['Controvalore EUR'].sum()
        fund_info = st.session_state.df_full.copy()
        fund_info = fund_info.loc[
            fund_info['L1'] == 'Funds', ['AC', 'Module', 'ISIN', 'Descrizione', 'Rebate']].drop_duplicates()
        df = df.merge(fund_info, on='ISIN')
        return df


# Initialize session state DataFrames if not already present
if 'df_ric' not in st.session_state:
    st.session_state.df_ric = pd.DataFrame(columns=['Instrument', 'RIC'])
if 'df_rectype' not in st.session_state:
    st.session_state.df_rectype = pd.DataFrame(columns=['Instrument', 'RECORDTYPE'])
if 'df_hist' not in st.session_state:
    st.session_state.df_hist = pd.DataFrame()

end = pd.to_datetime('today').normalize().tz_localize(None) - pd.tseries.offsets.BusinessDay(n=1)
start1 = end - pd.DateOffset(years=1)
start2 = end - pd.DateOffset(years=2) - pd.DateOffset(days=50)

model_ptf = pd.read_csv("lookup/model_ptf.csv", sep=";")
model_ptf['Weight'] = model_ptf['Weight'].astype(str).str.rstrip('%').str.replace(',', '.').astype('float') / 100
models = model_ptf.Model.unique().tolist()
model_hist = get_fund_hist(model_ptf.ISIN.unique().tolist(), start2, end)
model_logret = np.log(model_hist / model_hist.shift(1))
for m in models:
    w = model_ptf.loc[model_ptf['Model'] == m, ['ISIN', 'Weight']].set_index('ISIN')
    model_logret[m] = model_logret.mul(w['Weight']).sum(axis=1)

model_hist = 100 + 100 * model_logret[models].cumsum()
st.line_chart(model_hist)

existing_columns = set(st.session_state.df_hist.columns)
new_columns = set(model_hist.columns)
columns_to_add = list(new_columns - existing_columns)  # Columns in model_hist that aren't in df_hist

if columns_to_add:
    model_hist_filtered = model_hist[columns_to_add]  # Only keep the new columns
    st.session_state.df_hist = pd.concat([st.session_state.df_hist, model_hist_filtered], axis=1)

st.write(st.session_state.df_hist)

# Radio button to select data source
data_source = st.sidebar.radio("Choose data source", ("Use Feather DB", "Load Excel file"))

# Load data based on the selected source
if data_source == "Use Feather DB":
    df = load_feather()
else:
    df = load_excel()

if df is None or df.empty:
    st.warning('Please load valid file')
    st.stop()

# Slice by asset class and module
col1, col2 = st.columns([2, 3])

unique_ac = ['All'] + df['AC'].unique().tolist()
selected_ac = col1.multiselect('Select Asset Class(es)', unique_ac, default='All', key='selected_ac')
df = df if 'All' in selected_ac or not selected_ac else df.loc[df['AC'].isin(selected_ac)]

unique_mod = ['All'] + df['Module'].unique().tolist()
selected_mod = col2.multiselect('Select Module(s)', unique_mod, default='All', key='selected_mod')
df = df if 'All' in selected_mod or not selected_mod else df.loc[df['Module'].isin(selected_mod)]

# Generate slider steps including the final length
steps = list(range(0, len(df), 10))
if len(df) not in steps:
    steps.append(len(df))

# Limit the number of funds based on slider selection
num_funds = st.select_slider('Select number of funds for analysis:', options=steps, value=min(10, len(df)))
df = df.head(num_funds)

# Count of all ISINs in the selected portfolio
isin_list = df.ISIN.tolist()

# Get RICs from Eikon
missing_isins = [isin for isin in isin_list if isin not in st.session_state.df_ric['Instrument'].tolist()]

if missing_isins:
    df_ric_new, err = ek.get_data(missing_isins, fields=['TR.RIC'])
    df_ric_new['RIC'] = df_ric_new['RIC'].replace('', np.nan)
    st.session_state.df_ric = pd.concat([st.session_state.df_ric, df_ric_new], ignore_index=True)

df = df.merge(st.session_state.df_ric, left_on='ISIN', right_on='Instrument', how='left').drop(columns=['Instrument'])

if df['RIC'].isna().sum() > 0:
    ric_not_found = df.loc[df.RIC.isna(), 'ISIN'].tolist()
    st.write('Instruments not found on Eikon that will be dropped:',
             ", ".join(df.loc[df['ISIN'].isin(ric_not_found), 'Descrizione'].tolist()))

# Get RecordType from Eikon using RICs and merge
ric_list = df.RIC.dropna().tolist()
missing_rics = [ric for ric in ric_list if ric not in st.session_state.df_rectype['Instrument'].tolist()]

if missing_rics:
    df_rectype_new, err = ek.get_data(missing_rics, fields=['RECORDTYPE'])
    st.session_state.df_rectype = pd.concat([st.session_state.df_rectype, df_rectype_new], ignore_index=True)

df = df.merge(st.session_state.df_rectype, left_on='RIC', right_on='Instrument', how='left').drop(
    columns=['Instrument'])

rectype_funds = [96]
if len(df[~df.RECORDTYPE.isin(rectype_funds)]) > 0:
    st.write('Assets not mapped as a fund that will be dropped:')
    st.write(df.loc[~df.RECORDTYPE.isin(rectype_funds), 'Descrizione'])

df_funds = df[df.RECORDTYPE.isin(rectype_funds)].copy()
df_funds = df_funds.drop('RECORDTYPE', axis=1)
df_funds = df_funds.set_index('RIC', drop=True)

missing_rics = [ric for ric in df_funds.index if ric not in st.session_state.df_hist.columns]

if missing_rics:
    batch_size = 100
    for i in range(0, len(missing_rics), batch_size):
        batch_rics = missing_rics[i:i + batch_size]
        new_hist = get_fund_hist(rics=batch_rics, start=start2.to_pydatetime(), end=end.to_pydatetime())
        st.session_state.df_hist = pd.concat([st.session_state.df_hist, new_hist], axis=1)
        st.rerun()

df_hist = st.session_state.df_hist[df_funds.index]

### Perform calculations from here ########################################
win = 252
sma = 21

hist_too_short_rics = (df_hist.loc[:, df_hist.count() < win + 2 * sma]).columns.tolist()
if len(hist_too_short_rics) > 0:
    st.write("Funds with insufficient history that will be dropped:")
    st.write(df_funds.loc[hist_too_short_rics, 'Descrizione'])
    df_hist = df_hist.drop(hist_too_short_rics, axis=1)

ptf_val, logret = get_calc()

df_funds.loc['Ptf', 'Descrizione'] = 'Portfolio'
df_funds['Initial'] = ptf_val[start1:].iloc[0]
df_funds['Final'] = ptf_val.iloc[-1]
df_funds['Peso'] = df_funds['Final'] / df_funds['Final'].drop('Ptf').sum()
df_funds['PNL'] = df_funds['Final'] - df_funds['Initial']
df_funds['12M Rtn'] = df_funds['Final'] / df_funds['Initial'] - 1
df_funds['12M Contrib'] = df_funds['PNL'] / df_funds.loc['Ptf', 'Initial']
# CAREFUL IF LESS THAN ONE YEAR... pro-rata
df_funds['Rebate 12M EUR'] = df_funds['Rebate'] * ptf_val.loc[start1:].mean()
df_funds.loc['Ptf', 'Rebate 12M EUR'] = df_funds['Rebate 12M EUR'].drop('Ptf').sum()
df_funds.loc[:, 'contributor detractor'] = np.where(df_funds['12M Rtn'] >= 0, 'contributor', 'detractor')
df_funds.loc['Ptf', 'contributor detractor'] = np.nan

df_funds.loc[:, 'Last Yr Rtn'] = np.exp(logret.loc[logret.index.year == logret.index.year.max() - 1].sum()) - 1
df_funds.loc[:, 'YTD Rtn'] = np.exp(logret.loc[logret.index.year == logret.index.year.max()].sum()) - 1
df_funds['color'] = df_funds['Descrizione'].apply(lambda x: 'Portfolio' if x == 'Portfolio' else 'Other')

df_funds = df_funds.sort_values('12M Rtn', ascending=True)

st.write(df_funds)
df_funds.to_excel('df_funds.xlsx')

fund_names = df_funds['Descrizione'].to_dict()
fund_names['Ptf'] = 'Portfolio'


