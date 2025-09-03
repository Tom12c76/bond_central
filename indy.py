import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import scipy.interpolate as inter

import plotly.express as px

from pathlib import Path
import eikon as ek

df_funds_cols = ['Unit', 'NDG Banker', 'Banker', 'NDG Cliente', 'Intestazione', 'Rapporto', 'L1', 'ISIN',
 'Codice DB', 'Descrizione', 'Divisa', 'Quant', 'Prezzo', 'Cambio', 'Controvalore EUR', 'Abs Ctv EUR',
 'Data Rif', 'Prezzo carico', 'Perf prezzo', 'Cambio carico', 'Perf cambio', 'PNL Tot EUR', 'PNL Gain EUR',
 'PNL Loss EUR', 'Perf Tot', 'action', 'Perf EOY', 'Perf YTD', 'Data carico', 'Tipologia', 'AC', 'Module',
 'Rebate', 'Tunnel', 'Aliquota', 'Zainetto', 'Plus Fondi, ETF', 'DB Group']

ek.set_app_key('cf2eaf5e3b3c42adba08b3c5c2002b6ced1e77d7')

st.set_page_config(layout='wide')

file_path = Path('lookup') / 'ucits_model_ptf.csv'
models = pd.read_csv(file_path, sep=';', thousands=',')
del file_path

file_path = Path('lookup') / 'suore.xlsx'
df = pd.read_excel(file_path)
del file_path

df_funds = df[df_funds_cols]
st.write(df_funds)

isins = list(set(df.ISIN.tolist() + models.ISIN.tolist()))

# fund_names = df_funds.set_index('ISIN')['Descrizione'].to_dict()
# st.write(fund_names)

def get_hist(isins):
    # Load df_hist from file or fetch data if it doesn't exist
    file_path = Path('lookup') / 'df_hist.xlsx'
    if file_path.exists():
        df_hist = pd.read_excel(file_path, index_col=0)
    else:
        df_hist, err = ek.get_data(isins,
                                   ['TR.FundNav.Date', 'TR.FundNav'],
                                   {'SDate': '2024-10-10',
                                    'EDate': '2022-10-10',
                                    'Curn': 'EUR'})
        df_hist.to_excel(file_path)

    df_hist['Date'] = pd.to_datetime(df_hist['Date']).dt.date
    del file_path

    df_hist = df_hist.pivot(index='Date', columns=['Instrument'])
    df_hist = df_hist.droplevel(0, axis=1)
    df_hist = df_hist.sort_index(ascending=True)
    df_hist = df_hist.astype(float)

    return df_hist

df_hist = get_hist(isins)

def get_calc(df_hist, win=252, sma=21):

    end = df_hist.index.max()
    # start1 = end - pd.DateOffset(years=1)
    start1 = (end - pd.DateOffset(years=1)).date()
    # start2 = end - pd.DateOffset(years=2) - pd.DateOffset(days=50)

    logret = np.log(df_hist / df_hist.shift(1))
    logret['Ptf'] = logret.mean(axis=1)
    # st.write(df_funds.loc[logret.columns,"Peso"])

    # logret['Ptf'] = logret.mul(df_funds.loc[logret.columns, "Peso"], axis=1).sum(axis=1)

    cumret = logret[start1:].copy()
    cumret.iloc[0] = 0
    cumret = cumret.ffill()
    cumret = cumret.cumsum()
    cumret = np.exp(cumret) - 1

    cumret_xs = cumret.subtract(cumret['Ptf'], axis=0)

    rolret = (np.exp(logret.rolling(win).sum()) - 1).dropna(how='all')
    rolret_sma = rolret.rolling(sma).mean()

    rolvol = (logret.rolling(win).std() * np.sqrt(win)).dropna(how='all')
    rolrar = rolret / rolvol
    rolrar_sma = rolrar.rolling(sma).mean()

    rolret = rolret[start1:]
    rolret_sma = rolret_sma[start1:]

    spl_app = []
    for isin in rolret_sma:
        try:
            x = rolret_sma[isin].dropna().index.values.astype('datetime64[D]').astype(int)
            y = rolret_sma[isin].dropna().values
            spl = inter.UnivariateSpline(x, y, s=0.005)
            spl_app.append(pd.DataFrame(spl(x), index=rolret_sma[isin].dropna().index, columns=[isin]))
        except Exception as e:
            st.write(f"Error with ISIN {isin}: {e}")
            st.write(rolret_sma[isin].dropna())
    rolret_sma_spl = pd.concat(spl_app, axis=1) if spl_app else pd.DataFrame()

    rolrar = rolrar[start1:]
    rolrar_sma = rolrar_sma[start1:]

    spl_app = []
    acc_app = []
    for isin in rolrar_sma:
        if len(rolrar_sma[isin].dropna()) > 0:
            try:
                x = rolrar_sma[isin].dropna().index.values.astype('datetime64[D]').astype(int)
                y = rolrar_sma[isin].dropna().values
                spl = inter.UnivariateSpline(x, y, s=0.05)
                acc = spl.derivative()
                spl_app.append(pd.DataFrame(spl(x), index=rolrar_sma[isin].dropna().index, columns=[isin]))
                acc_app.append(pd.DataFrame(acc(x), index=rolrar_sma[isin].dropna().index, columns=[isin]))
            except Exception as e:
                st.write(f"Error with ISIN {isin}: {e}")
                st.write(rolrar_sma[isin].dropna())
    rolrar_sma_spl = pd.concat(spl_app, axis=1) if spl_app else pd.DataFrame()
    rolrar_sma_acc = pd.concat(acc_app, axis=1) if acc_app else pd.DataFrame()

    rarcdf = pd.DataFrame(stats.norm.cdf(rolrar), index=rolrar.index, columns=rolrar.columns)

    #### NEW FROM HERE ####

    # List of dataframes and their names
    metrics = {
        'close': df_hist,
        'logret': logret,
        'cumret': cumret,
        'cumret_xs': cumret_xs,
        'rolret': rolret,
        'rolret_sma': rolret_sma,
        'rolvol': rolvol,
        'rolrar': rolrar,
        'rolrar_sma': rolrar_sma,
        'rolret_sma_spl': rolret_sma_spl,
        'rolrar_sma_spl': rolrar_sma_spl,
        'rolrar_sma_acc': rolrar_sma_acc,
        'rarcdf': rarcdf
    }

    # Initialize an empty list to hold the melted dataframes
    melted_dfs = []

    # Loop through each dataframe and melt it
    for metric_name, df in metrics.items():
        if df is not None and not df.empty:
            # Reset index to have the date as a column
            df_reset = df.reset_index()

            # Melt the dataframe, keeping 'Date' as id_var and ISIN codes as values
            melted_df = df_reset.melt(id_vars=['Date'], var_name='ISIN', value_name='Value')

            # Add a column for the metric name
            melted_df['Metric'] = metric_name

            # Append to the list
            melted_dfs.append(melted_df)
            del df, df_reset

    # Concatenate all melted dataframes into one tall dataframe
    df_tall = pd.concat(melted_dfs, ignore_index=True)

    # Set the index to be a multi-index with ISIN first and Date second
    df_tall.set_index(['ISIN', 'Date'], inplace=True)

    # Pivot the dataframe to get metrics as columns
    df_tall = df_tall.pivot(columns='Metric', values='Value')

    # Reset the column names to remove the multi-level index created by `pivot()`
    df_tall.columns.name = None
    df_tall.reset_index(inplace=False)

    # REst the column order
    df_tall = df_tall[list(metrics.keys())]

    return df_tall

df_tall = get_calc(df_hist)

def get_fig_returns(df_funds, df_tall):

    max_date = df_tall.index.get_level_values(1).max()
    df_returns = df_tall.loc[df_tall.index.get_level_values(1) == max_date, ['cumret']]
    df_returns = df_returns.reset_index(level=1, drop=True)
    df_returns = df_returns.merge(df_funds[['ISIN', 'Descrizione']], on='ISIN', how='outer')
    df_returns.loc[df_returns['ISIN'] == 'Ptf', 'Descrizione'] = 'Portfolio'
    df_returns['color'] = df_returns['Descrizione'].apply(lambda x: 'Portfolio' if x == 'Portfolio' else 'Other')
    st.write(df_returns)

    fig_returns = px.bar(
        df_returns,
        y = 'Descrizione',
        x = 'cumret',
        orientation = 'h',
        text_auto = '.1%',
        title = 'Returns 12M',
        color = 'color',
        color_discrete_map = {'Portfolio':'#00a3e0', 'Other':'#0c2340'},
        category_orders = {'Descrizione':df_funds['Descrizione'].tolist()[::-1]}
    )

    fig_returns.add_vline(
        x=df_returns.loc[df_returns['ISIN'] != 'Ptf', 'cumret'].mean(),
        line_width=1.5, line_color='#ffc845', line_dash='dash',
        annotation=dict(text='Average', showarrow=True)
    )

    fig_returns.update_xaxes(tickformat='.0%', title='', showgrid=False)
    fig_returns.update_yaxes(title='', showgrid=True, mirror=False)
    fig_returns.update_layout(
                      height=210 * 5, width=297 * 5, template='presentation',
                      showlegend=False, font_family="Deutsche Bank Text",
                      margin=dict(l=80, r=30, t=150, b=50), plot_bgcolor=None)

    return fig_returns

st.plotly_chart(get_fig_returns(df_funds, df_tall))



