import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_fig(i, RIC, df_funds, cumret_xs, cumret, logret, rolvol, sorted_rics, rolret, rolret_sma_spl, cols):
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
