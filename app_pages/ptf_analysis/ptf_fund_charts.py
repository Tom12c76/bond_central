import os
import streamlit as st
import eikon as ek
import numpy as np
import pandas as pd
import scipy.interpolate as inter
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import subprocess
from PyPDF2 import PdfMerger

from app_pages.ptf_analysis.utils.calculations import get_calc
from app_pages.ptf_analysis.utils.charts import get_fig, get_fig_treemap, get_fig_trellis, get_fig_scatter, get_fig_contrib, get_fig_dendrogram, get_fig_rebate, get_fig_returns
from app_pages.ptf_analysis.utils.data_utils import load_feather_data, load_excel_data, get_fund_hist_cached

st.set_page_config(page_title="Ptf Fund Charts", page_icon="📈", layout='wide')

ek.set_app_key('cf2eaf5e3b3c42adba08b3c5c2002b6ced1e77d7')
cols = px.colors.qualitative.G10 * 10


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
    df = load_feather_data(st.session_state.df_filtered)
else:
    uploaded_file = st.sidebar.file_uploader("ISIN, (Descrizione), Ctv EUR", type="xlsx")
    if uploaded_file:
        df = load_excel_data(uploaded_file, st.session_state.df_full)
    else:
        df = None

if df is None or df.empty:
    st.warning('Please load valid file')
    st.stop()

# Display the filtered DataFrame in an expandable container
with st.expander('Filtered Portfolio Data', expanded=False):
    st.dataframe(
        df,
        column_config={
            "Controvalore EUR": st.column_config.NumberColumn(
                format="%.0f",
            ),
            "Rebate": st.column_config.NumberColumn(
                format="%.4f",
            ),
            "Quant": st.column_config.NumberColumn(
                format="%.0f"
            ),
        },
        hide_index=True,
        width='stretch'
    )

col1, col2 = st.columns([2,3])

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
    dfs_to_concat = [st.session_state.df_ric, df_ric_new]
    dfs_to_concat = [df for df in dfs_to_concat if not df.empty]
    if dfs_to_concat:
        st.session_state.df_ric = pd.concat(dfs_to_concat, ignore_index=True)

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
    dfs_to_concat = [st.session_state.df_rectype, df_rectype_new]
    dfs_to_concat = [df for df in dfs_to_concat if not df.empty]
    if dfs_to_concat:
        st.session_state.df_rectype = pd.concat(dfs_to_concat, ignore_index=True)

df = df.merge(st.session_state.df_rectype, left_on='RIC', right_on='Instrument', how='left').drop(columns=['Instrument'])

rectype_funds = [96]
if len(df[~df.RECORDTYPE.isin(rectype_funds)]) > 0:
    st.write('Assets not mapped as a fund that will be dropped:')
    st.write(df.loc[~df.RECORDTYPE.isin(rectype_funds), 'Descrizione'])

df_funds = df[df.RECORDTYPE.isin(rectype_funds)].copy()
df_funds = df_funds.drop('RECORDTYPE', axis=1)
df_funds = df_funds.set_index('RIC', drop=True)

end = pd.to_datetime('today').normalize().tz_localize(None) - pd.tseries.offsets.BusinessDay(n=1)
start1 = end - pd.DateOffset(years=1)
start2 = end - pd.DateOffset(years=2) - pd.DateOffset(days=50)

missing_rics = [ric for ric in df_funds.index if ric not in st.session_state.df_hist.columns]

if missing_rics:
    batch_size = 100
    for i in range(0, len(missing_rics), batch_size):
        batch_rics = missing_rics[i:i + batch_size]
        new_hist = get_fund_hist_cached(batch_rics, start2.to_pydatetime(), end.to_pydatetime())
        dfs_to_concat = [st.session_state.df_hist, new_hist]
        dfs_to_concat = [df for df in dfs_to_concat if not df.empty]
        if dfs_to_concat:
            st.session_state.df_hist = pd.concat(dfs_to_concat, axis=1)
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

ptf_val, logret, cumret, cumret_xs, rolret, rolret_sma, rolvol, rolrar, rolrar_sma, rolret_sma_spl, rolrar_sma_spl, rolrar_sma_acc, rarcdf = get_calc(df_hist, df_funds, start1, win, sma)

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
df_funds.loc[:, '12M XS Rtn'] = cumret_xs.iloc[-1]
df_funds['12M Attrib'] = df_funds['Peso'] * df_funds['12M XS Rtn']
df_funds.loc[:, 'leader laggard'] = np.where(df_funds['12M XS Rtn'] > 0, 'leader', 'laggard')
df_funds.loc['Ptf', 'leader laggard'] = np.nan

df_funds.loc[:, 'Last Yr Rtn'] = np.exp(logret.loc[logret.index.year == logret.index.year.max()-1].sum())-1
df_funds.loc[:, 'YTD Rtn'] = np.exp(logret.loc[logret.index.year == logret.index.year.max()].sum())-1

df_funds['color'] = df_funds['Descrizione'].apply(lambda x: 'Portfolio' if x == 'Portfolio' else 'Other')

df_funds = df_funds.sort_values('12M Rtn', ascending=True)

sorted_rics = cumret.iloc[-1].sort_values(ascending=False).index.tolist()
cumret = cumret[sorted_rics]
df_funds = df_funds.reindex(sorted_rics)

st.write(df_funds)
# df_funds.to_excel('df_funds.xlsx')

color_map_mod = {'GIG':'#e57200',
                'Thematic':'#016388',
                'Chiuso':'#e4002b',
                'Core Discretionary':'#1d2758',
                'High Conviction':'#04a7e0',
                'Complementary':'#293895',
                'Strategic Liquidity':'#609bc7',
                'not found':'#a8a8aa',
                'None':'#d7dee2'}

db_colors = ['#0c2340', '#4ac9e3', '#8794a1', '#ffc845', '#07792b', '#e4002b', '#00a3e0', '#671e75', '#cedc00',
             '#e57200', '#57646c', '#99dcf3', '#a4bcc2', '#c9b7d1', '#f29e97', '#a7d6cd', '#d7dee2']

fund_names = df_funds['Descrizione'].to_dict()
fund_names['Ptf'] = 'Portfolio'

sorted_rics = cumret.iloc[-1].sort_values(ascending=False).drop('Ptf').index.tolist()

st.write(df_funds.loc[sorted_rics, 'Descrizione'])

st.plotly_chart(get_fig(1, sorted_rics[0], df_funds, cumret_xs, cumret, logret, rolvol, sorted_rics, rolret, rolret_sma_spl, cols))

# Streamlit call to display the combined figure
fig_treemap = get_fig_treemap(df_funds, df, color_map_mod)
st.plotly_chart(fig_treemap)

fig_returns = get_fig_returns(df_funds)
st.plotly_chart(fig_returns)


fig_trellis = get_fig_trellis(cumret, df_funds)
st.plotly_chart(fig_trellis)

fig_scatter = get_fig_scatter(df_funds)
st.plotly_chart(fig_scatter)

# df_funds = df_funds.sort_values('12M Contrib', ascending=True)

fig_contrib = get_fig_contrib(df_funds)
st.plotly_chart(fig_contrib)

corr_matrix = logret.drop('Ptf', axis=1).rename(columns=fund_names).corr()
st.plotly_chart(
    px.imshow(corr_matrix,
              text_auto='.0%',
              width=800,
              height=800,
              labels={'x': '', 'y': ''},
              title='Correlation Matrix'
              ).update_layout(coloraxis_showscale=False)
)

import seaborn as sns

# Assuming logret is your DataFrame containing the returns, and 'Portfolio' column needs to be excluded
correlation_matrix = logret.rename(columns=fund_names).drop('Portfolio', axis=1).corr()

# Create a clustermap from the correlation matrix
# Clean correlation matrix: replace inf with nan, then fill nan with 0
correlation_matrix_clean = correlation_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)

sns.set_theme(font_scale=1)

clustermap_fig = sns.clustermap(
    correlation_matrix_clean*100,
    method='average',
    metric='euclidean',
    cmap='vlag',
    linewidths=0.5,
    figsize=(10, 8),
    annot=True,
    fmt=".0f"
)

# Save the clustermap to a file
clustermap_fig.savefig("clustermap.png")

# Display the saved image in Streamlit
st.image("clustermap.png", caption="Clustered Correlation Matrix of Funds", width='stretch')

# Clean logret before dendrogram: replace inf with nan, then fill nan with 0
logret_clean = logret.replace([np.inf, -np.inf], np.nan).fillna(0)

# Ensure only numeric columns
logret_clean = logret_clean.select_dtypes(include=[np.number])

# Check for finite values
if not np.isfinite(logret_clean.values).all():
    raise ValueError("Non-finite values remain in logret_clean after cleaning!")

# Check shape (should be 2D)
if logret_clean.ndim != 2:
    raise ValueError(f"logret_clean must be 2D, got shape {logret_clean.shape}")


fig_dendrogram = get_fig_dendrogram(logret_clean, fund_names)
st.plotly_chart(fig_dendrogram)

fig_rebate = get_fig_rebate(df_funds)
st.plotly_chart(fig_rebate)

def generate_and_merge_pdfs(fig_treemap, fig_returns, fig_trellis, fig_scatter, fig_contrib, fig_rebate, df_hist, df_funds, sorted_rics):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(script_dir) # go up one level
    output_dir = os.path.join(script_dir, "temp_data", "fund_charts") # create a temporary directory

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    merger = PdfMerger()
    try:
        #Efficiently merge files
        figures = [fig_treemap, fig_returns, fig_trellis, fig_scatter, fig_contrib, fig_rebate]
        filenames = ['fig_treemap', 'fig_returns', 'fig_trellis', 'fig_scatter', 'fig_contrib', 'fig_rebate']
        for fig, filename in zip(figures, filenames):
            filepath = os.path.join(output_dir, f"{filename}")
            st.write("Working on: " + filepath)
            fig.write_image(filepath + ".pdf") # Only save as PDF.
            # fig.write_image(filepath + ".png") # Only save as PNG.
            merger.append(filepath + ".pdf")

        for i, RIC in enumerate(sorted_rics):
            filename_pdf = os.path.join(output_dir, f"{df_hist.index[-1].date()}_{RIC}_{df_funds.loc[RIC, 'Descrizione'].replace('/', '')}.pdf")
            st.write("Working on: " + filename_pdf)
            fig = get_fig(i, RIC, df_funds, cumret_xs, cumret, logret, rolvol, sorted_rics, rolret, rolret_sma_spl, cols)
            fig.write_image(filename_pdf)
            merger.append(filename_pdf)

        timestamp = pd.to_datetime("today").strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = os.path.join(output_dir, f"{timestamp}_result.pdf")
        merger.write(output_filename)
        merger.close()
        #Safely open PDF
        subprocess.Popen([output_filename], shell=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

if st.button('Generate Charts'):
    with st.spinner('Generating charts...'):
        generate_and_merge_pdfs(fig_treemap, fig_returns, fig_trellis, fig_scatter, fig_contrib, fig_rebate, df_hist, df_funds, sorted_rics)