import os
import streamlit as st
import eikon as ek
import numpy as np
import pandas as pd
import plotly.express as px
import warnings
from PyPDF2 import PdfMerger
import subprocess
import seaborn as sns

from app_pages.ptf_analysis.utils.calculations import get_calc
from app_pages.ptf_analysis.utils.charts import get_fig, get_fig_treemap, get_fig_trellis, get_fig_scatter, get_fig_contrib, get_fig_dendrogram, get_fig_rebate, get_fig_returns
from app_pages.ptf_analysis.utils.data_utils import load_feather_data, load_excel_data, get_fund_hist_cached

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Constants ---
EIKON_APP_KEY = 'cf2eaf5e3b3c42adba08b3c5c2002b6ced1e77d7'
COLOR_MAP_MOD = {
    'GIG': '#e57200', 'Thematic': '#016388', 'Chiuso': '#e4002b',
    'Core Discretionary': '#1d2758', 'High Conviction': '#04a7e0',
    'Complementary': '#293895', 'Strategic Liquidity': '#609bc7',
    'not found': '#a8a8aa', 'None': '#d7dee2'
}
DB_COLORS = [
    '#0c2340', '#4ac9e3', '#8794a1', '#ffc845', '#07792b', '#e4002b',
    '#00a3e0', '#671e75', '#cedc00', '#e57200', '#57646c', '#99dcf3',
    '#a4bcc2', '#c9b7d1', '#f29e97', '#a7d6cd', '#d7dee2'
]
COLS = px.colors.qualitative.G10 * 10

# --- Eikon API Initialization ---
def initialize_eikon():
    """Initializes the Eikon API connection."""
    try:
        ek.set_app_key(EIKON_APP_KEY)
    except ek.EikonError as e:
        st.error(f"Failed to connect to Eikon: {e}")
        st.stop()

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes Streamlit session state variables."""
    if 'df_ric' not in st.session_state:
        st.session_state.df_ric = pd.DataFrame(columns=['Instrument', 'RIC'])
    if 'df_rectype' not in st.session_state:
        st.session_state.df_rectype = pd.DataFrame(columns=['Instrument', 'RECORDTYPE'])
    if 'df_hist' not in st.session_state:
        st.session_state.df_hist = pd.DataFrame()

# --- UI Components ---
def setup_sidebar():
    """Sets up the Streamlit sidebar for user input."""
    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio("Choose data source", ("Use Feather DB", "Load Excel file"))
    
    uploaded_file = None
    if data_source == "Load Excel file":
        uploaded_file = st.sidebar.file_uploader("ISIN, (Descrizione), Ctv EUR", type="xlsx")
        
    return data_source, uploaded_file

def setup_control_panel(df):
    """Sets up the control panel for filtering data."""
    st.header("Control Panel")
    col1, col2 = st.columns([2, 3])

    unique_ac = ['All'] + df['AC'].unique().tolist()
    selected_ac = col1.multiselect('Select Asset Class(es)', unique_ac, default='All')

    unique_mod = ['All'] + df['Module'].unique().tolist()
    selected_mod = col2.multiselect('Select Module(s)', unique_mod, default='All')

    steps = list(range(0, len(df), 10))
    if len(df) not in steps:
        steps.append(len(df))
    num_funds = st.select_slider('Select number of funds for analysis:', options=steps, value=min(10, len(df)))
    
    return selected_ac, selected_mod, num_funds

# --- Data Loading and Processing ---
def load_initial_data(data_source, uploaded_file):
    """Loads the initial portfolio data based on user selection."""
    if data_source == "Use Feather DB":
        if 'df_filtered' in st.session_state:
            return load_feather_data(st.session_state.df_filtered)
        else:
            st.warning("Feather DB selected, but no initial data found in session state.")
            return None
    elif uploaded_file:
        if 'df_full' in st.session_state:
            return load_excel_data(uploaded_file, st.session_state.df_full)
        else:
            st.warning("Excel file uploaded, but no full dataframe found in session state for merging.")
            return None
    return None

def enrich_data(df):
    """Enriches the DataFrame with RICs and historical data from Eikon."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    isin_list = df.ISIN.tolist()
    missing_isins = [isin for isin in isin_list if isin not in st.session_state.df_ric['Instrument'].tolist()]

    if missing_isins:
        df_ric_new, err = ek.get_data(missing_isins, fields=['TR.RIC'])
        if df_ric_new is not None and not df_ric_new.empty:
            df_ric_new['RIC'] = df_ric_new['RIC'].replace('', np.nan)
            st.session_state.df_ric = pd.concat([st.session_state.df_ric, df_ric_new], ignore_index=True)

    df = df.merge(st.session_state.df_ric, left_on='ISIN', right_on='Instrument', how='left').drop(columns=['Instrument'])
    
    df_funds = df[df.RECORDTYPE.isin([96])].copy() if 'RECORDTYPE' in df.columns else df.copy()
    if 'RECORDTYPE' in df_funds.columns:
        df_funds = df_funds.drop('RECORDTYPE', axis=1)
    if 'RIC' in df_funds.columns:
        df_funds = df_funds.set_index('RIC', drop=True)

    end = pd.to_datetime('today').normalize().tz_localize(None) - pd.tseries.offsets.BusinessDay(n=1)
    start2 = end - pd.DateOffset(years=2) - pd.DateOffset(days=50)
    
    valid_rics = [ric for ric in df_funds.index if isinstance(ric, str) and pd.notna(ric) and ric.strip() != ""]
    missing_rics = [ric for ric in valid_rics if ric not in st.session_state.df_hist.columns]

    if missing_rics:
        batch_size = 100
        for i in range(0, len(missing_rics), batch_size):
            batch_rics = [ric for ric in missing_rics[i:i + batch_size] if isinstance(ric, str) and pd.notna(ric) and ric.strip() != ""]
            if not batch_rics:
                continue
            new_hist = get_fund_hist_cached(batch_rics, start2.to_pydatetime(), end.to_pydatetime())
            if new_hist is not None and not new_hist.empty:
                st.session_state.df_hist = pd.concat([st.session_state.df_hist, new_hist], axis=1)
        st.rerun()

    df_hist = pd.DataFrame()
    if not df_funds.empty and not st.session_state.df_hist.empty:
        existing_rics = [ric for ric in valid_rics if ric in st.session_state.df_hist.columns]
        if existing_rics:
            df_hist = st.session_state.df_hist[existing_rics]
        
    return df, df_funds, df_hist

def perform_calculations(df_hist, df_funds):
    """Performs quantitative calculations on the portfolio data."""
    if df_hist.empty or df_funds.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    end = pd.to_datetime('today').normalize().tz_localize(None) - pd.tseries.offsets.BusinessDay(n=1)
    start1 = end - pd.DateOffset(years=1)
    win = 252
    sma = 21

    ptf_val, logret, cumret, cumret_xs, rolret, rolret_sma, rolvol, rolrar, rolrar_sma, rolret_sma_spl, rolrar_sma_spl, rolrar_sma_acc, rarcdf = get_calc(df_hist, df_funds, start1, win, sma)

    df_funds.loc['Ptf', 'Descrizione'] = 'Portfolio'
    if not ptf_val[start1:].empty:
        df_funds['Initial'] = ptf_val[start1:].iloc[0]
    else:
        df_funds['Initial'] = np.nan
        
    if not ptf_val.empty:
        df_funds['Final'] = ptf_val.iloc[-1]
    else:
        df_funds['Final'] = np.nan

    df_funds['Peso'] = df_funds['Final'] / df_funds['Final'].drop('Ptf', errors='ignore').sum()
    df_funds['PNL'] = df_funds['Final'] - df_funds['Initial']
    df_funds['12M Rtn'] = (df_funds['Final'] / df_funds['Initial']) - 1
    if 'Ptf' in df_funds.index and df_funds.loc['Ptf', 'Initial'] != 0:
        df_funds['12M Contrib'] = df_funds['PNL'] / df_funds.loc['Ptf', 'Initial']
    else:
        df_funds['12M Contrib'] = 0

    df_funds['Rebate 12M EUR'] = df_funds['Rebate'] * ptf_val.loc[start1:].mean()
    df_funds.loc['Ptf', 'Rebate 12M EUR'] = df_funds['Rebate 12M EUR'].drop('Ptf', errors='ignore').sum()
    df_funds['contributor detractor'] = np.where(df_funds['12M Rtn'] >= 0, 'contributor', 'detractor')
    df_funds.loc['Ptf', 'contributor detractor'] = np.nan
    
    if not cumret_xs.empty:
        df_funds['12M XS Rtn'] = cumret_xs.iloc[-1]
    else:
        df_funds['12M XS Rtn'] = np.nan

    df_funds['12M Attrib'] = df_funds['Peso'] * df_funds['12M XS Rtn']
    df_funds['leader laggard'] = np.where(df_funds['12M XS Rtn'] > 0, 'leader', 'laggard')
    df_funds.loc['Ptf', 'leader laggard'] = np.nan
    df_funds['Last Yr Rtn'] = np.exp(logret.loc[logret.index.year == logret.index.year.max()-1].sum()) - 1
    df_funds['YTD Rtn'] = np.exp(logret.loc[logret.index.year == logret.index.year.max()].sum()) - 1
    df_funds['color'] = df_funds['Descrizione'].apply(lambda x: 'Portfolio' if x == 'Portfolio' else 'Other')

    df_funds = df_funds.sort_values('12M Rtn', ascending=True)
    
    sorted_rics = []
    if not cumret.empty:
        sorted_rics = cumret.iloc[-1].sort_values(ascending=False).index.tolist()
        cumret = cumret[sorted_rics]
        df_funds = df_funds.reindex(sorted_rics)

    return df_funds, cumret, cumret_xs, logret, rolvol, rolret, rolret_sma_spl, sorted_rics

# --- Charting and Reporting ---
def display_warnings(df, df_hist, df_funds):
    """Displays warnings for dropped funds."""
    ric_not_found = df.loc[df.RIC.isna(), 'ISIN'].tolist() if 'RIC' in df.columns and 'ISIN' in df.columns else []
    not_found_descr = df.loc[df['ISIN'].isin(ric_not_found), 'Descrizione'].tolist() if ric_not_found else []

    rectype_funds = [96]
    not_fund_descr = df.loc[(df['RECORDTYPE'].notna()) & (~df['RECORDTYPE'].isin(rectype_funds)), 'Descrizione'].tolist() if 'RECORDTYPE' in df.columns else []

    win = 252
    sma = 21
    hist_too_short_rics = []
    if not df_hist.empty:
        hist_too_short_rics = (df_hist.loc[:, df_hist.count() < win + 2 * sma]).columns.tolist()
    
    too_short_descr = []
    if hist_too_short_rics and not df_funds.empty:
        valid_rics_in_funds = [ric for ric in hist_too_short_rics if ric in df_funds.index]
        if valid_rics_in_funds:
            too_short_descr = df_funds.loc[valid_rics_in_funds, 'Descrizione'].tolist()

    dropped_msgs = []
    if not_found_descr:
        dropped_msgs.append("**Not found on Eikon:** " + ", ".join(not_found_descr))
    if not_fund_descr:
        dropped_msgs.append("**Not mapped as fund:** " + ", ".join(not_fund_descr))
    if too_short_descr:
        dropped_msgs.append("**Insufficient history:** " + ", ".join(too_short_descr))
    if dropped_msgs:
        st.markdown("*The following funds will be dropped from the quant analysis:*<br>" + "<br>".join(dropped_msgs), unsafe_allow_html=True)

def display_dataframes(df, df_funds, df_hist):
    """Displays the DataFrames in expandable sections."""
    with st.expander('Filtered Portfolio Data', expanded=False):
        if df is not None and not df.empty:
            st.dataframe(df.style.format({"Controvalore EUR": "{:,.0f}", "Rebate": "{:.2%}", "Quant": "{:,.2f}"}))
        else:
            st.write("No portfolio data to display.")
    
    with st.expander("Historical Fund NAV's", expanded=False):
        if df_hist is not None and not df_hist.empty:
            st.dataframe(df_hist.style.format("{:,.2f}"))
        else:
            st.write("No historical data available.")

    with st.expander('Portfolio Calculations', expanded=False):
        if df_funds is not None and not df_funds.empty:
            st.dataframe(df_funds.style.format({
                "Controvalore EUR": "{:,.0f}", "Initial": "{:,.0f}", "Final": "{:,.0f}", "PNL": "{:,.0f}",
                "Rebate 12M EUR": "{:,.0f}", "Quant": "{:,.2f}", "Peso": "{:.2%}", "Rebate": "{:.2%}",
                "12M Rtn": "{:.2%}", "12M Contrib": "{:.2%}", "12M XS Rtn": "{:.2%}", "12M Attrib": "{:.2%}",
                "Last Yr Rtn": "{:.2%}", "YTD Rtn": "{:.2%}"
            }))
        else:
            st.write("No portfolio calculations to display.")

def display_charts(df, df_funds, cumret, cumret_xs, logret, rolvol, rolret, rolret_sma_spl, sorted_rics):
    """Generates and displays all the charts."""
    if not sorted_rics:
        st.info("Not enough data to display charts.")
        return None

    fund_names = df_funds['Descrizione'].to_dict()
    fund_names['Ptf'] = 'Portfolio'

    st.plotly_chart(get_fig(1, sorted_rics[0], df_funds, cumret_xs, cumret, logret, rolvol, sorted_rics, rolret, rolret_sma_spl, COLS))
    
    fig_treemap = get_fig_treemap(df_funds, df, COLOR_MAP_MOD)
    st.plotly_chart(fig_treemap)

    fig_returns = get_fig_returns(df_funds)
    st.plotly_chart(fig_returns)

    fig_trellis = get_fig_trellis(cumret, df_funds)
    st.plotly_chart(fig_trellis)

    fig_scatter = get_fig_scatter(df_funds)
    st.plotly_chart(fig_scatter)

    fig_contrib = get_fig_contrib(df_funds)
    st.plotly_chart(fig_contrib)

    corr_matrix = logret.drop('Ptf', axis=1, errors='ignore').rename(columns=fund_names).corr()
    st.plotly_chart(px.imshow(corr_matrix, text_auto='.0%', width=800, height=800, labels={'x': '', 'y': ''}, title='Correlation Matrix').update_layout(coloraxis_showscale=False))

    correlation_matrix_clean = corr_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
    sns.set_theme(font_scale=1)
    try:
        clustermap_fig = sns.clustermap(correlation_matrix_clean * 100, method='average', metric='euclidean', cmap='vlag', linewidths=0.5, figsize=(10, 8), annot=True, fmt=".0f")
        clustermap_fig.savefig("clustermap.png")
        st.image("clustermap.png", caption="Clustered Correlation Matrix of Funds", use_column_width=True)
    except Exception as e:
        st.error(f"Could not generate clustermap: {e}")


    logret_clean = logret.replace([np.inf, -np.inf], np.nan).fillna(0).select_dtypes(include=[np.number])
    if not logret_clean.empty:
        fig_dendrogram = get_fig_dendrogram(logret_clean, fund_names)
        st.plotly_chart(fig_dendrogram)

    fig_rebate = get_fig_rebate(df_funds)
    st.plotly_chart(fig_rebate)
    
    return fig_treemap, fig_returns, fig_trellis, fig_scatter, fig_contrib, fig_rebate

def generate_pdf_report(figures, df_hist, df_funds, sorted_rics, cumret_xs, cumret, logret, rolvol, rolret, rolret_sma_spl):
    """Generates and merges PDFs for the report."""
    if not figures:
        st.error("Cannot generate report because no charts were created.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(script_dir), "temp_data", "fund_charts")
    os.makedirs(output_dir, exist_ok=True)

    merger = PdfMerger()
    try:
        fig_treemap, fig_returns, fig_trellis, fig_scatter, fig_contrib, fig_rebate = figures
        
        figure_files = {
            'treemap': fig_treemap, 'returns': fig_returns, 'trellis': fig_trellis,
            'scatter': fig_scatter, 'contrib': fig_contrib, 'rebate': fig_rebate
        }

        for name, fig in figure_files.items():
            filepath = os.path.join(output_dir, f"fig_{name}.pdf")
            fig.write_image(filepath)
            merger.append(filepath)

        for i, ric in enumerate(sorted_rics):
            description = df_funds.loc[ric, 'Descrizione'].replace('/', '').replace('\\', '')
            filename_pdf = os.path.join(output_dir, f"{df_hist.index[-1].date()}_{ric}_{description}.pdf")
            fig = get_fig(i, ric, df_funds, cumret_xs, cumret, logret, rolvol, sorted_rics, rolret, rolret_sma_spl, COLS)
            fig.write_image(filename_pdf)
            merger.append(filename_pdf)

        timestamp = pd.to_datetime("today").strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = os.path.join(output_dir, f"{timestamp}_result.pdf")
        merger.write(output_filename)
        merger.close()
        
        try:
            subprocess.Popen([output_filename], shell=True)
            st.success(f"Successfully generated and opened {output_filename}")
        except Exception as e:
            st.success(f"Successfully generated {output_filename}")
            st.warning(f"Could not automatically open the PDF. Please find it at: {output_filename}. Error: {e}")

    except Exception as e:
        st.error(f"An error occurred during PDF generation: {e}")

# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Ptf Fund Charts", page_icon="📈", layout='wide')
    
    initialize_eikon()
    initialize_session_state()
    
    data_source, uploaded_file = setup_sidebar()
    df = load_initial_data(data_source, uploaded_file)

    if df is None or df.empty:
        st.info('Please load a valid file to begin analysis.')
        st.stop()

    selected_ac, selected_mod, num_funds = setup_control_panel(df)
    
    df_filtered = df.copy()
    if 'All' not in selected_ac and selected_ac:
        df_filtered = df_filtered[df_filtered['AC'].isin(selected_ac)]
    if 'All' not in selected_mod and selected_mod:
        df_filtered = df_filtered[df_filtered['Module'].isin(selected_mod)]
    df_filtered = df_filtered.head(num_funds)

    df_enriched, df_funds_pre_calc, df_hist = enrich_data(df_filtered)
    
    display_warnings(df_enriched, df_hist, df_funds_pre_calc)
    
    df_funds, cumret, cumret_xs, logret, rolvol, rolret, rolret_sma_spl, sorted_rics = perform_calculations(df_hist, df_funds_pre_calc)
    
    display_dataframes(df_filtered, df_funds, df_hist)
    
    figures_for_pdf = display_charts(df_filtered, df_funds, cumret, cumret_xs, logret, rolvol, rolret, rolret_sma_spl, sorted_rics)

    if st.button('Generate PDF Report'):
        with st.spinner('Generating PDF report...'):
            generate_pdf_report(figures_for_pdf, df_hist, df_funds, sorted_rics, cumret_xs, cumret, logret, rolvol, rolret, rolret_sma_spl)

if __name__ == "__main__":
    main()