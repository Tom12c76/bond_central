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

st.set_page_config(page_title="Ptf Fund Charts", page_icon="📈", layout='wide')

ek.set_app_key('cf2eaf5e3b3c42adba08b3c5c2002b6ced1e77d7')
cols = px.colors.qualitative.G10 * 10

# Fetch or use cached historical data
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

    rolret_sma_spl = None  # Initialize to None
    try:
        spl_app = []
        for isin in rolret_sma:
            x = rolret_sma[isin].dropna().index.values.astype('datetime64[D]').astype(int)
            y = rolret_sma[isin].dropna().values

            if len(x) > 3:  # Minimum data points
                spl = inter.UnivariateSpline(x, y, s=0.005)
                spl_app.append(pd.DataFrame(spl(x), index=rolret_sma[isin].dropna().index, columns=[isin]))
            else:
                print(f"Skipping spline (rolret_sma): ISIN {isin} has too few data points.")
                spl_app.append(pd.DataFrame(index=rolret_sma[isin].dropna().index, columns=[isin]))  # Empty DF

        rolret_sma_spl = pd.concat(spl_app, axis=1)
    except Exception as e:
        print(f"Error calculating rolret_sma_spl: {e}")
        # Optionally set rolret_sma_spl to an empty dataframe or a dataframe filled with NaN,
        # depending on how you want to handle the error downstream.
        # rolret_sma_spl = pd.DataFrame(index=rolret_sma.index, columns=rolret_sma.columns)
        rolret_sma_spl = rolret_sma.copy()
        rolret_sma_spl[:] = np.nan


    rolrar = rolrar[start1:]
    rolrar_sma = rolrar_sma[start1:]

    rolrar_sma_spl = None
    rolrar_sma_acc = None

    try:
        spl_app = []
        acc_app = []
        for isin in rolrar_sma:
            x = rolrar_sma[isin].dropna().index.values.astype('datetime64[D]').astype(int)
            y = rolrar_sma[isin].dropna().values

            if len(x) > 3: # Minimum Data Points
                spl = inter.UnivariateSpline(x, y, s=0.05)
                acc = spl.derivative()
                spl_app.append(pd.DataFrame(spl(x), index=rolrar_sma[isin].dropna().index, columns=[isin]))
                acc_app.append(pd.DataFrame(acc(x), index=rolrar_sma[isin].dropna().index, columns=[isin]))
            else:
                print(f"Skipping spline (rolrar_sma): ISIN {isin} has too few data points.")
                spl_app.append(pd.DataFrame(index=rolrar_sma[isin].dropna().index, columns=[isin]))
                acc_app.append(pd.DataFrame(index=rolrar_sma[isin].dropna().index, columns=[isin]))


        rolrar_sma_spl = pd.concat(spl_app, axis=1)
        rolrar_sma_acc = pd.concat(acc_app, axis=1)
    except Exception as e:
        print(f"Error calculating rolrar_sma_spl/acc: {e}")
        rolrar_sma_spl = rolrar_sma.copy()
        rolrar_sma_spl[:] = np.nan
        rolrar_sma_acc = rolrar_sma.copy()
        rolrar_sma_acc[:] = np.nan



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
    tall_df = pd.concat(melted_dfs, ignore_index=True)

    # Set the index to be a multi-index with ISIN first and Date second
    # Check if tall_df is empty before attempting to set the index
    if not tall_df.empty:
        tall_df.set_index(['ISIN', 'Date'], inplace=True)

        # Pivot the dataframe to get metrics as columns
        tall_df = tall_df.pivot(columns='Metric', values='Value')

        # Reset the column names to remove the multi-level index created by `pivot()`
        tall_df.columns.name = None
        tall_df.reset_index(inplace=False)

        # REst the column order
        tall_df = tall_df[list(metrics.keys())]

    else:
        print("Warning: tall_df is empty.  Returning empty DataFrame.")
        tall_df = pd.DataFrame() # Or some other default.

    # Display the resulting tall dataframe
    # st.write(tall_df)

    return ptf_val, logret, cumret, cumret_xs, rolret, rolret_sma, rolvol, rolrar, rolrar_sma, rolret_sma_spl, rolrar_sma_spl, rolrar_sma_acc, rarcdf


def get_fig(i, RIC):
    # def get_fig(
    #     i=0,
    #     RIC='ABC123',
    #     df_funds=df_funds,
    #     cumret_xs=cumret_xs,
    #     cumret=cumret,
    #     logret=logret,
    #     rolvol=rolvol,
    #     sorted_rics=sorted_rics,
    #     rolret=rolret,
    #     rolret_sma_spl=rolret_sma_spl,
    #     cols=cols
    # )

    fund_name = df_funds.loc[RIC, 'Descrizione']
    fund_names = df_funds['Descrizione'].to_dict()
    fund_names['Ptf'] = 'Portfolio'

    fig = go.Figure()

    fig = make_subplots(rows=4, cols=3,
                        shared_xaxes=True,
                        shared_yaxes=True,
                        subplot_titles=['<b>1 Year Excess Return', '', '',
                                        '<b>1 Year Total Return', '', '<b>Risk-Total Return',
                                        '', '', '<b>Volatility',
                                        '<b>1 Year Rolling Return (20 day SMA)'],
                        column_widths=(8, 0.75, 3), row_heights=(1, 3, 0.5, 1),
                        horizontal_spacing=0.02,
                        vertical_spacing=0.04)

    ### ROW 1 ###############################
    ### Row 1, Col1: Excess Return Line chart
    fig.add_trace(go.Scatter(
        x=cumret_xs.index,
        y=cumret_xs[RIC],
        name="Tom",  # This sets the name for the series
        line=dict(color=cols[i]),
        hovertemplate='<b>%{name}</b><br>Date: %{x|%d-%b-%Y}<br>XS Ret: %{y:.2%}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=[cumret_xs[RIC].idxmin()],
                             y=[cumret_xs[RIC].min()],
                             mode='markers+text',
                             name=fund_name,
                             showlegend=False,
                             text=f'{cumret_xs[RIC].idxmin():%d-%b}',
                             textposition='middle right',
                             textfont_size=10,
                             marker=dict(color='red')),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=[cumret_xs[RIC].idxmax()],
                             y=[cumret_xs[RIC].max()],
                             mode='markers+text',
                             name=fund_name,
                             text=f'{cumret_xs[RIC].idxmax():%d-%b}',
                             textposition='middle right',
                             textfont_size=10,
                             showlegend=False,
                             marker=dict(color='blue')), row=1, col=1)

    ### Row 1, Col2: Excess Return Bar chart
    fig.add_trace(go.Bar(x=[fund_name],
                         y=[cumret_xs[RIC].iloc[-1]],
                         text=[f'{cumret_xs[RIC].iloc[-1]:.1%}'],
                         showlegend=True, width=0.75,
                         marker=dict(color=cols[i])), row=1, col=2)

    ### ROW 2 ##################################
    ### Row 2, Col 1: Cumlative Return Line plot
    fig.add_trace(go.Scatter(x=cumret.index,
                             y=cumret.max(axis=1),
                             name='Max',
                             opacity=0.2,
                             line=dict(color='grey', width=2),
                             visible=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=cumret.index,
                             y=cumret['Ptf'],
                             name='Portfolio',
                             line=dict(color='grey')), row=2, col=1)

    fig.add_trace(go.Scatter(x=cumret.index,
                             y=cumret.min(axis=1),
                             name='Min',
                             opacity=0.2,
                             line=dict(color='grey', width=0)), row=2, col=1)

    fig.add_trace(go.Scatter(x=cumret.index,
                             y=cumret[RIC],
                             name=fund_name,
                             line=dict(color=cols[i])), row=2, col=1)

    ### Row 2, Col 2: Cumlative Return Bar plot
    fig.add_trace(go.Bar(x=[fund_name, 'Portfolio'],
                         y=[cumret[RIC].iloc[-1], cumret['Ptf'].iloc[-1]],
                         text=[f'{cumret[RIC].iloc[-1]:.1%}', f'{cumret["Ptf"].iloc[-1]:.1%}'],
                         width=0.75,
                         showlegend=True,
                         marker_color=[cols[i], 'grey']), row=2, col=2)

    # Cumative Return versus Risk Scatter chart

    corr = logret['Ptf'].corr(logret[RIC])
    ef = pd.DataFrame(np.linspace(0, 1, 21), columns=['wa'])
    ef['wb'] = 1 - ef['wa']
    ef['va'] = rolvol[RIC].iloc[-1]
    ef['vb'] = rolvol['Ptf'].iloc[-1]

    # REMINDER:
    # Portfolio variance = w1σ1^2 + w2σ2^2 + 2w1w2Cov
    # Cov = p(1,2)σ1σ2, where p(1,2)=correlation
    # Portfolio variance = w1σ1^2 + w2σ2^2 + 2w1w2p(1,2)σ1σ2

    ef['vp'] = np.sqrt(ef.wa * ef.va ** 2 + ef.wb * ef.vb ** 2 + 2 * ef.wa * ef.wb * corr * ef.va * ef.vb)
    ef['rp'] = ef.wa * cumret[RIC].iloc[-1] + ef.wb * cumret['Ptf'].iloc[-1]
    ef['rar'] = ef['rp'] / ef['vp']

    # Handling NaN values for marker size
    # Create a list to store marker sizes, replacing NaN with 0
    marker_sizes = []
    names = []

    for ric in sorted_rics:
        if not pd.isna(rolvol[ric].iloc[-1]) and not pd.isna(df_funds.loc[ric, 'Peso']):  # Both values should be valid
            marker_size = df_funds.loc[ric, 'Peso'] * 200
            marker_sizes.append(marker_size)
            names.append(ric)
        else:
            # If NaN, append 0 or another default value
            marker_sizes.append(0)  # Or some default size, like 10
            names.append(ric)

    fig.add_trace(go.Scatter(x=rolvol[sorted_rics].iloc[-1],
                             y=cumret[sorted_rics].iloc[-1],
                             mode='markers',
                             # name=df_funds.loc[sorted_rics, 'Descrizione'].tolist(),
                             text=df_funds.loc[sorted_rics, 'Descrizione'].tolist(),
                             textposition='bottom right',
                             marker=dict(color='lightgrey',
                                         size=marker_sizes),  # marker size based on if statement above
                             ), row=2, col=3)

    #  Handling the individual RIC
    if not pd.isna(rolvol[RIC].iloc[-1]) and not pd.isna(df_funds.loc[RIC, 'Peso']):  # Check for NaN
        marker_size = df_funds.loc[RIC, 'Peso'] * 200
    else:
        marker_size = 0  # default size

    fig.add_trace(go.Scatter(x=[rolvol[RIC].iloc[-1]],
                             y=[cumret[RIC].iloc[-1]],
                             name=fund_name,
                             marker=dict(color=cols[i],
                                         size=marker_size)), row=2, col=3) # Using the defined marker_size

    fig.add_trace(go.Scatter(x=ef['vp'],
                             y=ef['rp'],
                             name='Frontier',
                             line=dict(color=cols[i], dash=None, width=1)), row=2, col=3)

    ### Horizontal Bar Chart Volatility ##############

    fig.add_trace(go.Bar(y=['Portfolio', fund_name],
                         x=[rolvol['Ptf'].iloc[-1], rolvol[RIC].iloc[-1]],
                         showlegend=False, orientation='h', width=0.75,
                         text=[f'{rolvol["Ptf"].iloc[-1]:.1%}', f'{rolvol[RIC].iloc[-1]:.1%}'],
                         hovertemplate='<b>%{y}</b><br>Vol: %{text}<extra></extra>',
                         marker_color=['grey', cols[i]]), row=3, col=3)

    # Adding an SRRI axis to the plot
    srri_labels = ["Conservativo\u25B2 2%", "Bilanciato\u25B2 5%", "Crescita\u25B2 10%", "Dinamico\u25B2 15%",
                   "Max Crescita\u25B2 20%"]
    srri_ranges = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]  # Using upper bounds for labels

    fig.update_xaxes(
        row=3, col=3,
        tickvals=srri_ranges,
        ticktext=srri_labels,
        tickangle=270,
    )

    # Rolling Return Fund

    fig.add_trace(go.Scatter(x=rolret.index,
                             y=rolret[RIC],
                             name=fund_name,
                             showlegend=False,
                             mode='markers',
                             marker=dict(color=cols[i], size=3, opacity=0.2)), row=4, col=1)

    fig.add_trace(go.Scatter(x=rolret_sma_spl.index,
                             y=rolret_sma_spl[RIC],
                             name=fund_name,
                             showlegend=False,
                             line=dict(color=cols[i])), row=4, col=1)

    # Rolling Return BM

    fig.add_trace(go.Scatter(x=rolret.index,
                             y=rolret['Ptf'],
                             name='Portfolio',
                             showlegend=False,
                             mode='markers',
                             marker=dict(color='grey', size=3, opacity=0.2)), row=4, col=1)

    fig.add_trace(go.Scatter(x=rolret_sma_spl['Ptf'].dropna().index,
                             y=rolret_sma_spl['Ptf'],
                             name='Core Equity Fund Avg',
                             showlegend=False,
                             line=dict(color='grey')), row=4, col=1)

    # Lines and annotations

    fig.add_vline(x=cumret_xs[RIC].idxmin(), line=dict(color='red', dash='dash', width=1), col=1, row=(1))
    fig.add_vline(x=cumret_xs[RIC].idxmax(), line=dict(color='blue', dash='dash', width=1), col=1, row=(1))

    fig.add_hline(y=cumret['Ptf'].iloc[-1], line=dict(color='grey', dash='dash', width=1), col=3, row=(2))
    fig.add_vline(x=rolvol['Ptf'].iloc[-1], line=dict(color='grey', dash='dash', width=1), col=3, row=(2))

    # formatting

    fig.update_xaxes(row=2, col=3, autorange='reversed', rangemode='tozero')
    fig.update_xaxes(row=3, col=3, showticklabels=True)
    fig.update_yaxes(tickformat=',.0%')
    fig.update_layout(title=fund_name,
                      height=210 * 5, width=297 * 5, template='presentation',
                      showlegend=False, font_family="Deutsche Bank Text",
                      margin=dict(l=80, r=30, t=150, b=50), plot_bgcolor='whitesmoke')

    return fig


def get_fig_treemap():
    # First Treemap: Breakdown by AC
    df_ac = df_funds.groupby(['AC', 'Descrizione'])['Controvalore EUR'].sum().reset_index()
    df_ac['Peso'] = df['Controvalore EUR'] / df['Controvalore EUR'].sum()
    fig_ac = px.treemap(
        df_ac,
        path=[px.Constant('Breakdown by AC'), 'AC', 'Descrizione'],
        values='Controvalore EUR',
        custom_data=['Controvalore EUR', 'Peso'],
        title=None
    ).update_traces(
        texttemplate='%{label}<br>%{customdata[0]:,.0f} EUR<br>%{customdata[1]:.1%}',
        hovertemplate='<b>%{label}</b><br>Ctv EUR: %{customdata[0]:,.0f}<br>Peso: %{customdata[1]:.1%}<extra></extra>'
    )

    # Second Treemap: Breakdown by Module
    df_module = df_funds.groupby(['Module', 'Descrizione'])['Controvalore EUR'].sum().reset_index()
    df_module['Peso'] = df_module['Controvalore EUR'] / df_module['Controvalore EUR'].sum()
    fig_module = px.treemap(
        df_module,
        path=[px.Constant('Breakdown by Module'), 'Module', 'Descrizione'],
        values='Controvalore EUR',
        custom_data=['Controvalore EUR', 'Peso'],
        title=None,
        color='Module',
        color_discrete_map={'(?)': '#FFFFFF', **color_map_mod}
    ).update_traces(
        hovertemplate='<b>%{label}</b><br>Ctv EUR: %{customdata[0]:,.0f}<br>Peso: %{customdata[1]:.1%}',
        texttemplate='%{label}<br>%{customdata[0]:,.0f} EUR<br>%{customdata[1]:.1%}',
    )

    # Create a figure with subplots with proper specs for treemaps
    fig_treemap = make_subplots(
        rows=2, cols=1,
        subplot_titles=[None, None],
        specs=[[{"type": "domain"}], [{"type": "domain"}]],
        vertical_spacing=0.05
    )

    # Add both treemaps to the subplots
    for trace in fig_ac.data:
        fig_treemap.add_trace(trace, row=1, col=1)

    for trace in fig_module.data:
        fig_treemap.add_trace(trace, row=2, col=1)

    # Update the layout for the combined figure
    fig_treemap.update_layout(
        title_text='Portfolio Breakdown Analysis',
        height=210 * 5, width=297 * 5, template='presentation',
        showlegend=False, font_family='Deutsche Bank Text',
        margin=dict(l=30, r=30, t=150, b=50), plot_bgcolor=None
    )

    return fig_treemap


def get_fig_returns():
    # def get_fig_returns(df_funds):

    fig_returns = px.bar(
        df_funds[['Descrizione', '12M Rtn', 'color']],
        y = 'Descrizione',
        x = '12M Rtn',
        orientation = 'h',
        text_auto = '.1%',
        title = 'Returns 12M',
        color = 'color',
        color_discrete_map = {'Portfolio':'#00a3e0', 'Other':'#0c2340'},
        category_orders = {'Descrizione':df_funds['Descrizione'].tolist()[::-1]}
    )

    fig_returns.add_vline(
        x=df_funds['12M Rtn'].drop('Ptf').mean(), line_width=1.5, line_color='#ffc845', line_dash='dash',
        annotation=dict(text='Average', showarrow=True)
    )

    fig_returns.update_xaxes(tickformat='.0%', title='', showgrid=False)
    fig_returns.update_yaxes(title='', showgrid=True, mirror=False)
    fig_returns.update_layout(
                      height=210 * 5, width=297 * 5, template='presentation',
                      showlegend=False, font_family="Deutsche Bank Text",
                      margin=dict(l=300, r=50, t=150, b=100))

    return fig_returns


def get_fig_trellis():
    # def get_fig_trellis(cumret, df_funds):

    cumret_long = cumret.rename(columns=df_funds['Descrizione'].to_dict())
    cumret_long = cumret_long.reset_index()
    cumret_long = cumret_long.melt(id_vars='Date', var_name='Fund', value_name='cumret')

    # Plotting the multi-fund line chart over columns and rows using Plotly Express
    fig_trellis = px.line(
        cumret_long,
        x='Date',
        y='cumret',
        facet_col='Fund',
        facet_col_wrap=5,  # Adjust this to control the number of columns per row
        color_discrete_sequence=['#0c2340'],
        title='Cumulative Returns 12M'
    )

    # Adding the 'Portfolio' line to each subplot
    funds = cumret_long['Fund'].unique()
    num_funds = len(funds)
    num_rows = (num_funds + 4) // 5  # +4 to round up for any remainder
    for i, fund in enumerate(funds):
        fig_trellis.add_trace(
            go.Scatter(
                x=cumret_long.set_index('Fund').loc['Portfolio', 'Date'],
                y=cumret_long.set_index('Fund').loc['Portfolio', 'cumret'],
                mode='lines',
                name='Portfolio',
                line=dict(color='#00a3e0'),
                showlegend=False
            ),
            row=num_rows - (i // 5),
            col=(i % 5) + 1
        )

    for annotation in fig_trellis.layout.annotations:
        annotation.text = annotation.text.split('=')[1]

    fig_trellis.update_yaxes(tickformat='.0%', title='', showgrid=False, zeroline=True, showline=False,
                             minor=dict(showgrid=False))
    fig_trellis.update_xaxes(showgrid=False, showticklabels=False, title="", showline=False)


    fig_trellis.update_layout(
        height=210 * 5, width=297 * 5, template='presentation',
        showlegend=False, font_family="Deutsche Bank Text",
        margin=dict(l=80, r=30, t=150, b=50), plot_bgcolor=None)

    return fig_trellis


def get_fig_scatter():
    # def get_fig_scatter(df_funds):

    df_filtered = df_funds[['Descrizione', '12M Rtn', 'Peso', '12M Contrib']].dropna()

    # Create lollipop chart using Scatter and Line traces
    fig_scatter = go.Figure()

    # Add lines from y=0 to the y-value (lollipop sticks)
    for idx, row in df_filtered.drop('Ptf').iterrows():
        fig_scatter.add_trace(go.Scatter(
            x=[row['Peso'], row['Peso']],
            y=[0, row['12M Rtn']],
            mode='lines',
            line=dict(color='#0c2340', width=3),
            opacity=0.6,
            showlegend=False
        ))

    # Add circles at the end of each line (lollipop heads)
    fig_scatter.add_trace(go.Scatter(
        x=df_filtered['Peso'].drop('Ptf'),
        y=df_filtered['12M Rtn'].drop('Ptf'),
        mode='markers+text',
        marker=dict(size=20, color='#0c2340'),
        text=df_filtered['Descrizione'].drop('Ptf'),
        textposition='middle right',
        showlegend=False
    ))

    fig_scatter.add_vline(x=df_funds['Peso'].drop('Ptf').mean(),
                          line=dict(color='#ffc845', dash='dash', width=2),
                          annotation=dict(text='Avg Weight'))

    fig_scatter.add_hline(y=df_funds['12M Rtn'].drop('Ptf').mean(),
                          line=dict(color='#ffc845', dash='dash', width=2),
                          annotation=dict(text='Avg Rtn'))

    fig_scatter.add_hline(y=df_funds.loc['Ptf', '12M Rtn'],
                          line=dict(color='#00a3e0', dash='dash', width=2),
                          annotation=dict(text='Portfolio'))

    fig_scatter.update_traces(textposition='middle right')
    fig_scatter.update_xaxes(tickformat='.0%', title='Portfolio Weight', showline=False, mirror=False, rangemode='tozero')
    fig_scatter.update_yaxes(tickformat='.0%', title='Return 12M', showline=False, mirror=False)

    fig_scatter.update_layout(
                      title='Contribution Scatterplot',
                      height=210 * 5, width=297 * 5, template='presentation',
                      showlegend=False, font_family="Deutsche Bank Text",
                      margin=dict(l=100, r=80, t=150, b=80), plot_bgcolor=None)

    return fig_scatter


def get_fig_contrib(df_funds):

    temp = df_funds[['Descrizione', '12M Contrib']].sort_values('12M Contrib', ascending=False)
    x_values = temp['12M Contrib'].drop('Ptf').tolist() + [0]
    y_labels = temp['Descrizione'].drop('Ptf').tolist() + ['Portafoglio']
    new_index = temp.drop('Ptf').index.tolist() + ['Ptf']
    reordered_series = temp['12M Contrib'].loc[new_index]
    text_labels = [f'{x:.2%}' for x in reordered_series]

    # Creating the Waterfall figure with specified colors.
    fig_contrib = go.Figure(go.Waterfall(
        name="Perf contrib",
        orientation="h",
        measure=["relative"] * (len(df_funds) - 1) + ['total'],
        y=y_labels,
        textposition="outside",
        text=text_labels,
        x=x_values,
        increasing= dict(marker=dict(color='#0c2340')),
        decreasing = dict(marker=dict(color='#8794a1')),
        totals=dict(marker=dict(color='#00a3e0')),
        connector=dict(visible=False)
    ))

    fig_contrib.add_vline(
        x=temp.loc['Ptf','12M Contrib'], line_width=1.5, line_color='#00a3e0', line_dash='dash',
        # annotation=dict(text='Average', showarrow=True)
    )

    # Updating axes and layout.
    fig_contrib.update_xaxes(tickformat='.0%', title='12M Contribution', showgrid=False)
    fig_contrib.update_yaxes(title='', showgrid=True, autorange='reversed', side='right')
    fig_contrib.update_layout(
        title='Portfolio Contribution Waterfall',
        height=210 * 5, width=297 * 5, template='presentation',
        showlegend=False, font_family="Deutsche Bank Text",
        margin=dict(l=50, r=300, t=150, b=100), plot_bgcolor=None
    )

    return fig_contrib


def get_fig_dendrogram(logret, fund_names, user_defined_colors=None):

    # Default user-defined colors if none are provided
    if user_defined_colors is None:
        user_defined_colors = ['#0c2340', '#4ac9e3', '#8794a1', '#ffc845', '#07792b', '#e4002b', '#00a3e0', '#671e75',
                               '#cedc00', '#e57200', '#57646c', '#99dcf3', '#a4bcc2', '#c9b7d1', '#f29e97', '#a7d6cd',
                               '#d7dee2']

    # Assuming you have the correlation matrix named corr_matrix
    corr_matrix = logret.rename(columns=fund_names).drop('Portfolio', axis=1).corr()

    # Convert the correlation matrix to a distance matrix
    distance_matrix = 1 - corr_matrix

    dendrogram = ff.create_dendrogram(
        distance_matrix,
        orientation='right',
        labels=corr_matrix.columns.tolist())

    # If user_defined_colors is provided, substitute the dendrogram colors
    color_map = {}  # Map original colors to user-defined colors
    unique_colors = list(set(scatter['marker']['color'] for scatter in dendrogram['data']))
    for i, original_color in enumerate(unique_colors):
        color_map[original_color] = user_defined_colors[i % len(user_defined_colors)]

    # Apply the user-defined colors to the dendrogram traces
    for scatter in dendrogram['data']:
        scatter['marker']['color'] = color_map[scatter['marker']['color']]

    # Extracting (x, y, color) triplets
    triplets = []
    for scatter in dendrogram['data']:
        color = scatter['marker']['color']
        x_coords = scatter['x']
        y_coords = scatter['y']

        # Creating (x, y, color) triplets
        for x, y in zip(x_coords, y_coords):
            triplets.append((x, y, color))

    # Creating a DataFrame from the triplets
    color_codes_df = pd.DataFrame(triplets, columns=['x', 'y', 'color'])
    # Filter and sort unique colors based on 'y' coordinate
    color_codes = color_codes_df.query('x == 0')[['y', 'color']].drop_duplicates().sort_values('y')['color']

    # Step 4: Calculate cumulative returns
    cumulative_returns = np.exp(logret.sum()) - 1

    # Assigning colors based on the dendrogram colors for consistency
    color_codes_list = color_codes.values.tolist()

    # Step 5: Create the horizontal bar chart
    bar_chart = go.Bar(
        x=cumulative_returns.values,
        y=np.array(dendrogram['layout']['yaxis']['tickvals']),
        marker=dict(color=color_codes_list),
        orientation='h'
    )

    # Step 6: Combine dendrogram and bar chart into subplots with shared y-axis
    fig_dendrogram = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        column_widths=[0.3, 0.7],
        horizontal_spacing=0.02,
        subplot_titles=("Dendrogram", "Cumulative Returns")
    )

    # Add dendrogram to the first subplot
    for trace in dendrogram['data']:
        fig_dendrogram.add_trace(trace, row=1, col=1)

    # Add bar chart to the second subplot
    fig_dendrogram.add_trace(bar_chart, row=1, col=2)

    # Update layout
    fig_dendrogram.update_layout(
        title='Mutual Funds Analysis: Dendrogram and Cumulative Returns',
        showlegend=False,
        height=800
    )

    # Update y-axis to use the dendrogram labels
    fig_dendrogram.update_xaxes(title_text='Distance (1 - Correlation)', row=1, col=1)
    fig_dendrogram.update_xaxes(title_text='Cumulative Return', row=1, col=2)

    fig_dendrogram.update_yaxes(showticklabels=True, row=1, col=2, ticktext=distance_matrix.columns,
                                tickvals=np.array(dendrogram['layout']['yaxis']['tickvals']), side='right',
                                showgrid=True)
    fig_dendrogram.update_yaxes(showticklabels=False, row=1, col=1)

    # Debug: Print distance matrix to check for inconsistencies
    st.write("Distance Matrix:\n", distance_matrix)

    # Debug: Print dendrogram tick values
    st.write("Dendrogram Tickvals:\n", dendrogram['layout']['yaxis']['tickvals'])

    # Debug: Print cumulative returns calculation
    cumulative_returns = np.exp(logret.sum()) - 1
    st.write("Cumulative Returns:\n", cumulative_returns)

    return fig_dendrogram


def get_fig_rebate():
    df_filtered = df_funds.drop('Ptf').sort_values('Rebate 12M EUR', ascending=False)

    # Create lollipop chart using Scatter and Line traces
    fig_rebate = go.Figure()

    # Add lines from y=0 to the y-value (lollipop sticks)
    for idx, row in df_filtered.iterrows():
        fig_rebate.add_trace(go.Scatter(
            x=[row['Peso'], row['Peso']],
            y=[0, row['Rebate 12M EUR']],
            mode='lines',
            line=dict(color='#0c2340', width=3),
            opacity=0.6,
            showlegend=False
        ))

    # Add circles at the end of each line (lollipop heads)
    fig_rebate.add_trace(go.Scatter(
        x=df_filtered['Peso'],
        y=df_filtered['Rebate 12M EUR'],
        mode='markers+text',
        marker=dict(size=20, color='#0c2340'),
        text=df_filtered['Descrizione'],
        textposition='middle right',
        showlegend=False
    ))

    # Add a trend line with the average rebate
    max_peso = df_funds['Peso'].drop('Ptf').max()
    total_rebate = df_funds['Rebate 12M EUR'].drop('Ptf').sum()
    fig_rebate.add_trace(go.Scatter(
        x=[0, max_peso],
        y=[0, max_peso * total_rebate],
        mode='lines',
        line=dict(color='#ffc845', width=3, dash='dash'),
        name='Avg Rebate'
    ))

    fig_rebate.update_xaxes(tickformat='.2%', title='Fund Weight (%)')
    fig_rebate.update_yaxes(tickformat=',.0f', title='Total Rebate (EUR)')
    fig_rebate.update_layout(
        title='Rebates',
        height=210 * 5, width=297 * 5, template='presentation',
        font_family="Deutsche Bank Text",
        margin = dict(l=100, r=80, t=150, b=80), plot_bgcolor = None
    )

    return fig_rebate


def load_feather():
    # Your existing asset class filtering logic
    asset_class = 'Funds'
    df = st.session_state.df_filtered.copy()
    df = df.loc[df['L1'] == asset_class]

    # Group by ISIN and Descrizione and sum Controvalore EUR and Peso
    df = df.groupby(['AC', 'Module', 'ISIN', 'Descrizione'], as_index=False) \
        .agg({'Controvalore EUR':'sum', 'Rebate':'median', 'Quant':'sum'})

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


# Radio button to select data source
data_source = st.sidebar.radio("Choose data source", ("Use Feather DB", "Load Excel file"))

# Load data based on the selected source
if data_source == "Use Feather DB":
    df = load_feather()
else:
    df = load_excel()

if df is None or df.empty:
    st.warning('Please load valid file')

# else:
col1, col2 = st.columns([2,3])

unique_ac = ['All'] + df['AC'].unique().tolist()
selected_ac = col1.multiselect('Select Asset Class(es)', unique_ac, default='All', key='selected_ac')
df = df if 'All' in selected_ac or not selected_ac else df.loc[df['AC'].isin(selected_ac)]

unique_mod = ['All'] + df['Module'].unique().tolist()
selected_mod = col2.multiselect('Select Module(s)', unique_mod, default='All', key='selected_mod')
df = df if 'All' in selected_mod or not selected_mod else df.loc[df['Module'].isin(selected_mod)]

# Display the filtered DataFrame
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
    use_container_width=True
)

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

ptf_val, logret, cumret, cumret_xs, rolret, rolret_sma, rolvol, rolrar, rolrar_sma, rolret_sma_spl, rolrar_sma_spl, rolrar_sma_acc, rarcdf = get_calc()

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

# lastday_lastyear = rolret.loc[rolret.index.year == rolret.index[-1].year - 1].index.max()

df_funds.loc[:, 'Last Yr Rtn'] = np.exp(logret.loc[logret.index.year == logret.index.year.max()-1].sum())-1
df_funds.loc[:, 'YTD Rtn'] = np.exp(logret.loc[logret.index.year == logret.index.year.max()].sum())-1

df_funds['color'] = df_funds['Descrizione'].apply(lambda x: 'Portfolio' if x == 'Portfolio' else 'Other')

df_funds = df_funds.sort_values('12M Rtn', ascending=True)

sorted_rics = cumret.iloc[-1].sort_values(ascending=False).index.tolist()
cumret = cumret[sorted_rics]
df_funds = df_funds.reindex(sorted_rics)

st.write(df_funds)
df_funds.to_excel('df_funds.xlsx')

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

st.plotly_chart(get_fig(1, sorted_rics[0]))

# Streamlit call to display the combined figure
fig_treemap = get_fig_treemap()
st.plotly_chart(fig_treemap)

fig_returns = get_fig_returns()
st.plotly_chart(fig_returns)

fig_trellis = get_fig_trellis()
st.plotly_chart(fig_trellis)

fig_scatter = get_fig_scatter()
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
import numpy as np
correlation_matrix_clean = correlation_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)

sns.set(font_scale=1)
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
st.image("clustermap.png", caption="Clustered Correlation Matrix of Funds", use_container_width=True)






# Clean logret before dendrogram: replace inf with nan, then fill nan with 0
import numpy as np
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

fig_rebate = get_fig_rebate()
st.plotly_chart(fig_rebate)

# if st.button('Generate Charts'):
#     with st.spinner('On it ...'):
#         merger = PdfMerger()
#
#         # Exporting charts in both PDF and PNG formats
#         fig_treemap.write_image('fig_treemap.pdf')
#         fig_treemap.write_image('fig_treemap.png')
#         merger.append('fig_treemap.pdf')
#
#         fig_returns.write_image('fig_returns.pdf')
#         fig_returns.write_image('fig_returns.png')
#         merger.append('fig_returns.pdf')
#
#         fig_trellis.write_image('fig_trellis.pdf')
#         fig_trellis.write_image('fig_trellis.png')
#         merger.append('fig_trellis.pdf')
#
#         fig_scatter.write_image('fig_scatter.pdf')
#         fig_scatter.write_image('fig_scatter.png')
#         merger.append('fig_scatter.pdf')
#
#         fig_contrib.write_image('fig_contrib.pdf')
#         fig_contrib.write_image('fig_contrib.png')
#         merger.append('fig_contrib.pdf')
#
#         fig_rebate.write_image('fig_rebate.pdf')
#         fig_rebate.write_image('fig_rebate.png')
#         merger.append('fig_rebate.pdf')
#
#         for i, RIC in enumerate(sorted_rics):
#             filename_pdf = f'temp_data/fund_charts/{df_hist.index[-1].date()} ' \
#                            f'{RIC} ' \
#                            f'{df_funds.loc[RIC, "Descrizione"].replace("/", "")}.pdf'
#             filename_png = f'temp_data/fund_charts/{df_hist.index[-1].date()} ' \
#                            f'{RIC} ' \
#                            f'{df_funds.loc[RIC, "Descrizione"].replace("/", "")}.png'
#
#             st.write(filename_pdf)
#             fig = get_fig(i, RIC)
#
#             # Save as PDF and PNG
#             fig.write_image(filename_pdf)
#             fig.write_image(filename_png)
#             merger.append(filename_pdf)
#
#         st.write('merging PDF...')
#         filename = pd.to_datetime("today").strftime("%Y-%m-%d %H-%M-%S") + ' result.pdf'
#         merger.write(filename)
#         merger.close()
#         subprocess.Popen([filename], shell=True)

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
            fig.write_image(filepath + ".pdf") #Only save as PDF.
            fig.write_image(filepath + ".png")  # Only save as PDF.
            merger.append(filepath + ".pdf")

        for i, RIC in enumerate(sorted_rics):
            filename_pdf = os.path.join(output_dir, f"{df_hist.index[-1].date()}_{RIC}_{df_funds.loc[RIC, 'Descrizione'].replace('/', '')}.pdf")
            st.write("Working on: " + filename_pdf)
            fig = get_fig(i, RIC) # Assuming get_fig function exists
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