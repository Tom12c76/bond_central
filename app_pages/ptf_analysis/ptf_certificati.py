import streamlit as st
import numpy as np
import pandas as pd

import eikon as ek
from eikon.tools import get_date_from_today
import xlwings as xw
from PyPDF2 import PdfMerger
import subprocess
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Ptf Certificati", page_icon="🏦", layout='wide')

# import plotly.express as px
# cols = px.colors.qualitative.G10*10

cols = [
    '#0c2340', '#4ac9e3', '#ffc845', '#e4002b', '#07792b',
    '#00a3e0', '#a7d6cd', '#8794a1', '#0018a8', '#c9b7d1',
    '#99dcf3', '#f29e97', '#e57200', '#cedc00', '#57646c',
    '#671e75', '#d7dee2', '#a4bcc2']

# Eikon API key and set page configuration
ek.set_app_key('cf2eaf5e3b3c42adba08b3c5c2002b6ced1e77d7')
st.set_page_config(layout='wide')

# Initialize session state variables
if 'df_filtered' not in st.session_state:
    st.warning('HOUSTON: df_filtered not found in session state memory')
if 'bonds_ek' not in st.session_state:
    st.session_state.bonds_ek = pd.DataFrame(columns=['ISIN'])


def get_hist():
    df_hist = ek.get_timeseries(df_basket_isin['RIC'].tolist(),
                                start_date=(dt_strike - pd.Timedelta(days=90)).strftime('%Y-%m-%d'),
                                end_date=get_date_from_today(1),
                                corax='adjusted',
                                interval="daily")

    # spesso ci sono dei -1 nelle serie storiche... non so perchè. mercato chiuso?
    # get rid of -1
    df_hist = df_hist.replace(-1, np.nan).fillna(method='ffill')

    ### Single or Multi underlying ###
    # sarebbe meglio: if len(df_basket_isin['RIC']) >1:
    if len(df_hist.columns) > 6:
        df_hist = df_hist.loc[:, (slice(None), 'CLOSE')].droplevel(axis=1, level=1)
    else:
        df_hist = df_hist[['CLOSE']]
        df_hist = df_hist.rename(columns={'CLOSE': df_hist.columns.name})

    perf_recap = []
    df_chg = df_hist.copy()
    for ric in df_chg.columns:
        strike = df_basket[(df_basket['ISIN'] == isin) & (df_basket['RIC'] == ric)]['STRIKE'].item()
        df_chg[ric] = df_chg[ric] / strike
        perf_recap.append([ric, strike, df_hist[ric].iloc[-1].item(), df_chg[ric].iloc[-1].item() - 1])

    df_last = pd.DataFrame(perf_recap, columns=['RIC', 'Strike', 'Last', 'Chg'])

    if how == 'BO':
        df_ref = df_chg.max(axis=1)
    elif how == 'WO':
        df_ref = df_chg.min(axis=1)
    else:
        df_ref = df_chg.mean(axis=1)

    ### Get historical prices of Certificate ###
    df_hist_cert = pd.DataFrame()
    try:
        df_hist_cert = ek.get_timeseries(isin + '.TX',
                                         start_date=(dt_strike - pd.Timedelta(days=90)).strftime('%Y-%m-%d'),
                                         end_date=get_date_from_today(1),
                                         corax='adjusted',
                                         interval="daily")[['CLOSE']] / nominale
    except:
        df_hist_cert = pd.DataFrame()
    return df_hist, df_chg, df_last, df_ref, df_hist_cert


def get_cedole():
    cedole_isin = df_cedole.loc[isin]
    cedole_isin = cedole_isin.reset_index().set_index('DATA RILEVAMENTO').merge(df_ref.rename('ref'), left_index=True,
                                                                                right_index=True, how='left')
    cedole_isin['is_known'] = ~np.isnan(cedole_isin['ref'])
    cedole_isin['is_known'] = cedole_isin['is_known'].fillna(False)
    cedole_isin['TRIGGER CEDOLA'] = cedole_isin['TRIGGER CEDOLA'].fillna(0)
    cedole_isin['cedola_chk'] = np.greater_equal(cedole_isin['ref'], cedole_isin['TRIGGER CEDOLA'])

    marker_symbol = 'arrow-bar-up'

    ced_max = 1
    ced = 1
    mem = 0

    for d in cedole_isin.index:
        ced_max = ced_max + cedole_isin.loc[d, 'CEDOLA']
        cedole_isin.loc[d, 'ced_max'] = ced_max

        if cedole_isin.loc[d, 'is_known']:

            if np.isnan(cedole_isin.loc[d, 'TRIGGER AUTOCALLABLE']) or cedole_isin.loc[d]['ref'] < cedole_isin.loc[d][
                'TRIGGER AUTOCALLABLE']:

                if cedole_isin.loc[d, 'tipo_cedola'] == 'memoria':
                    if cedole_isin.loc[d, 'ref'] >= cedole_isin.loc[d]['TRIGGER CEDOLA']:
                        ced = ced + mem + cedole_isin.loc[d, 'CEDOLA']
                        mem = 0
                        cedole_isin.loc[d, 'ced_pagate'] = ced
                        cedole_isin.loc[d, 'ced_mem'] = mem
                    else:
                        mem = mem + cedole_isin.loc[d]['CEDOLA']
                        cedole_isin.loc[d, 'ced_pagate'] = ced
                        cedole_isin.loc[d, 'ced_mem'] = mem
                else:
                    if cedole_isin.loc[d, 'ref'] >= cedole_isin.loc[d, 'TRIGGER CEDOLA']:
                        ced = ced + cedole_isin.loc[d, 'CEDOLA']
                        mem = 0
                        cedole_isin.loc[d, 'ced_pagate'] = ced
                        cedole_isin.loc[d, 'ced_mem'] = mem

    cedole_isin = pd.concat(
        [cedole_isin, pd.DataFrame(data={'ced_max': [1], 'ced_pagate': [1]}, index=[dt_strike])]).sort_index()

    return cedole_isin, marker_symbol


def get_fig():
    fig = make_subplots(rows=2, cols=1, row_heights=[4, 1], vertical_spacing=None,
                        specs=[[{"type": "scatter"}],
                               [{"type": "table"}]])

    fig.add_trace(
        go.Scatter(x=df_ref.index, y=df_ref, mode='lines', name='Reference', line=dict(color=cols[7], width=0.75)))
    fig.add_trace(go.Scatter(x=[df_ref.index[-1]], y=[df_ref[-1]], mode='text', text=f'   Ref = {df_ref[-1] - 1:.1%}',
                             textposition='top right', showlegend=False))
    if len(df_hist_cert) > 0:
        fig.add_trace(
            go.Scatter(x=df_hist_cert.index, y=df_hist_cert.CLOSE, mode='lines', line=dict(color=cols[1], width=2),
                       opacity=1, name=f'Prezzo TLX'))
        fig.add_trace(go.Scatter(x=[df_hist_cert.index[-1]], y=[df_hist_cert.CLOSE[-1]], mode='text',
                                 text=f'   Perf Prz = {df_hist_cert.CLOSE[-1] - 1:.1%}', textposition='top right',
                                 showlegend=False))

    if cert_type in ['Phoenix']:

        fig.add_trace(
            go.Scatter(x=cedole_isin.index, y=cedole_isin['TRIGGER CEDOLA'], mode='markers', name='Trig Cedola',
                       marker=dict(color=cols[3], size=10, line=dict(color=cols[3], width=2)),
                       marker_symbol=marker_symbol))

        if len(cedole_isin['ced_max'].dropna()) > 0:
            fig.add_trace(go.Scatter(x=cedole_isin.index, y=cedole_isin['ced_max'],
                                     text=[f'{x - 1:.1%}' for x in cedole_isin['ced_max'].dropna()],
                                     textposition='top left',
                                     mode='lines+markers+text', name='Cedole Ideal',
                                     line=dict(color=cols[7], width=0.75), showlegend=False))
            fig.add_trace(
                go.Scatter(x=cedole_isin.index, y=cedole_isin['ced_pagate'], mode='lines+markers', name='Cedole Pagate',
                           marker=dict(color=cols[0]), marker_symbol='circle', line_shape='hv'))
            fig.add_trace(go.Scatter(x=[cedole_isin['ced_pagate'].dropna().index[-1]],
                                     y=[cedole_isin['ced_pagate'].dropna()[-1]], mode='text',
                                     text=[f'   Ced = {cedole_isin.ced_pagate.dropna()[-1] - 1: 0.1%}'],
                                     textposition='top right',
                                     showlegend=False))

    if cert_type in ['Express']:
        fig.add_trace(go.Scatter(x=cedole_isin.index, y=cedole_isin['PREMIO PER IL RIMBORSO'] + 1,
                                 mode='lines+markers', name='Cedole Ideal', line=dict(color=cols[7], width=0.75),
                                 showlegend=False))
        fig.add_trace(
            go.Scatter(x=cedole_isin.index, y=cedole_isin['TRIGGER AUTOCALLABLE'], mode='markers', name='Trig Autocall',
                       marker=dict(color=cols[4], size=10, line=dict(color=cols[4], width=2)),
                       marker_symbol=marker_symbol))

    if cert_type in ['Bonus']:

        df_bonus = df_chg.copy()

        if not np.isnan(df_info.loc[isin, 'Partecipazione up']):
            scale = df_info.loc[isin, 'Partecipazione up']
        else:
            scale = 1
        df_bonus['upside_scaled'] = np.maximum((df_ref - 1) * scale, 0) + 1

        if not np.isnan(df_info.loc[isin, 'Cap']):
            cap = df_info.loc[isin, 'Cap']
            fig.add_hline(y=cap, line=dict(color=cols[3], dash='dash', width=1), annotation_text=f'Bonus Cap {cap:.0%}',
                          annotation_position="top left", row=None, col=None)
            df_bonus['cap'] = cap - 1
        else:
            df_bonus['cap'] = np.nan

        df_bonus['upside_scaled_capped'] = df_bonus[['upside_scaled', 'cap']].max(axis=1)
        df_bonus = df_bonus.loc[dt_strike:]
        fig.add_trace(go.Scatter(x=df_bonus.index, y=df_bonus['upside_scaled_capped'],
                                 mode='lines', line=dict(color=cols[0], width=2),
                                 name=f'Intrinsic<br>(part. up {scale:.0%})'))

    fig.add_trace(
        go.Table(
            header=dict(
                values=df_last.columns,
                font=dict(),
                fill_color='whitesmoke',
                align="left"),
            cells=dict(
                values=[df_last[k].tolist() for k in df_last.columns],
                font=dict(),
                format=["", ".4f", ".4f", ".1%"],
                align="right")),
        row=2, col=1)

    fig.add_hline(y=barriera, line=dict(color=cols[3], dash='dash', width=1), annotation_text=f'Barriera {barriera:.0%}',
                  annotation_position="top left", row=None, col=None)
    fig.add_vline(x=dt_strike, line=dict(color='grey', dash='dash', width=1), row=None, col=None)
    fig.add_vline(x=dt_finale, line=dict(color='grey', dash='dash', width=1), row=None, col=None)

    fig.update_layout(height=210 * 5, width=297 * 5)
    fig.update_layout(yaxis={'tickformat': ',.1%'}, template='presentation',
                      showlegend=True, font_family="Deutsche Bank Text",
                      margin=dict(l=80, r=30, t=100, b=100), paper_bgcolor=None, plot_bgcolor=None,
                      title=df_descr.loc[isin]['description'] + "  [" + isin + "]")
    return fig


##########################
### Main Streamlit App ###
##########################

asset_class = 'Certificati'
cert = st.session_state.df_filtered.copy()
cert = cert[cert['is_in_opc'] == 'Certificato']
cert = cert.dropna(subset=['Controvalore EUR'])
cert = cert.groupby(['ISIN', 'Descrizione', 'Divisa', 'Tipologia', 'Sub AC'])[['Quant', 'Controvalore EUR']] \
             .sum().reset_index().sort_values('Controvalore EUR', ascending=False)

st.write(cert)
my_isins = cert.ISIN.tolist()

cert_db_xlsx = r'C:\Users\A9T230\OneDrive - Deutsche Bank AG\_Certificati\cert_db.xlsx'
df_descr = xw.Book(cert_db_xlsx).sheets('description').range('tbl_description[#All]').options(pd.DataFrame).value
df_info = xw.Book(cert_db_xlsx).sheets('info').range('tbl_info[#All]').options(pd.DataFrame).value
df_basket = xw.Book(cert_db_xlsx).sheets('basket').range('tbl_basket[#All]').options(pd.DataFrame).value.reset_index()
df_cedole = xw.Book(cert_db_xlsx).sheets('cedole').range('tbl_cedole[#All]').options(pd.DataFrame).value

mapped=[]
not_mapped=[]
st.write('List of ISINs not yet mapped:')
for isin in my_isins:
    if isin in df_info.index:
        mapped.append(isin)
    else:
        not_mapped.append(isin)
        st.write(isin)

st.write(cert.loc[cert['ISIN'].isin(not_mapped)])
cert.loc[cert['ISIN'].isin(not_mapped)].to_excel('not_mapped.xlsx')

st.write('')
st.write('No of ISINS already mapped: ' + f'{len(mapped)}')

if st.button('Generate Charts -->'):
    st.write('Generating Charts & PDF...')
    merger = PdfMerger()
    isin_err = []

    # for isin in ['XS2313844305']:
    for isin in mapped:
        try:
            cert_type = df_descr.loc[isin, 'cert_type']
            st.write(isin, cert_type)
            how = df_descr.loc[isin, 'how']
            dt_strike = df_info.loc[isin, 'Data strike']
            barriera = df_info.loc[isin, 'Barriera']
            nominale = df_info.loc[isin, 'Nominale']
            dt_finale = df_info.loc[isin, 'Data Valutazione finale']
            df_basket_isin = df_basket[df_basket['ISIN'] == isin]
            df_hist, df_chg, df_last, df_ref, df_hist_cert = get_hist()
            if cert_type in ['Phoenix', 'Express']:
                cedole_isin, marker_symbol = get_cedole()
            fig = get_fig()
            # st.plotly_chart(fig)
            fig.write_image('temp_data/certificati/' + isin + '.pdf')
            fig.write_image('temp_data/certificati/' + isin + '.png')
            merger.append('temp_data/certificati/' + isin + '.pdf')

        except Exception as e:
            st.warning('error with isin = ' + isin + ': ')
            isin_err.append(isin)

    filename = pd.to_datetime("today").strftime("%Y-%m-%d %H-%M-%S") + ' Certificati Result.pdf'
    merger.write(filename)
    merger.close()
    subprocess.Popen([filename], shell=True)