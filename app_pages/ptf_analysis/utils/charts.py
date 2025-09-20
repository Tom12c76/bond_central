import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.subplots import make_subplots

def get_fig_returns(df_funds):
    fig_returns = px.bar(
        df_funds[['Descrizione', '12M Rtn', 'color']],
        y='Descrizione',
        x='12M Rtn',
        orientation='h',
        text_auto='.1%',
        title='Returns 12M',
        color='color',
        color_discrete_map={'Portfolio': '#00a3e0', 'Other': '#0c2340'},
        category_orders={'Descrizione': df_funds['Descrizione'].tolist()[::-1]}
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
        margin=dict(l=300, r=50, t=150, b=100)
    )
    return fig_returns

def get_fig_trellis(cumret, df_funds):
    cumret_long = cumret.rename(columns=df_funds['Descrizione'].to_dict())
    cumret_long = cumret_long.reset_index()
    cumret_long = cumret_long.melt(id_vars='Date', var_name='Fund', value_name='cumret')
    fig_trellis = px.line(
        cumret_long,
        x='Date',
        y='cumret',
        facet_col='Fund',
        facet_col_wrap=5,
        color_discrete_sequence=['#0c2340'],
        title='Cumulative Returns 12M'
    )
    funds = cumret_long['Fund'].unique()
    num_funds = len(funds)
    num_rows = (num_funds + 4) // 5
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

def get_fig_scatter(df_funds):
    df_filtered = df_funds[['Descrizione', '12M Rtn', 'Peso', '12M Contrib']].dropna()
    fig_scatter = go.Figure()
    for idx, row in df_filtered.drop('Ptf').iterrows():
        fig_scatter.add_trace(go.Scatter(
            x=[row['Peso'], row['Peso']],
            y=[0, row['12M Rtn']],
            mode='lines',
            line=dict(color='#0c2340', width=3),
            opacity=0.6,
            showlegend=False
        ))
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
    )
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
    import plotly.figure_factory as ff
    import streamlit as st
    if user_defined_colors is None:
        user_defined_colors = ['#0c2340', '#4ac9e3', '#8794a1', '#ffc845', '#07792b', '#e4002b', '#00a3e0', '#671e75',
                               '#cedc00', '#e57200', '#57646c', '#99dcf3', '#a4bcc2', '#c9b7d1', '#f29e97', '#a7d6cd',
                               '#d7dee2']
    corr_matrix = logret.rename(columns=fund_names).drop('Portfolio', axis=1).corr()
    distance_matrix = 1 - corr_matrix
    dendrogram = ff.create_dendrogram(
        distance_matrix,
        orientation='right',
        labels=corr_matrix.columns.tolist())
    color_map = {}
    unique_colors = list(set(scatter['marker']['color'] for scatter in dendrogram['data']))
    for i, original_color in enumerate(unique_colors):
        color_map[original_color] = user_defined_colors[i % len(user_defined_colors)]
    for scatter in dendrogram['data']:
        scatter['marker']['color'] = color_map[scatter['marker']['color']]
    triplets = []
    for scatter in dendrogram['data']:
        color = scatter['marker']['color']
        x_coords = scatter['x']
        y_coords = scatter['y']
        for x, y in zip(x_coords, y_coords):
            triplets.append((x, y, color))
    color_codes_df = pd.DataFrame(triplets, columns=['x', 'y', 'color'])
    color_codes = color_codes_df.query('x == 0')[['y', 'color']].drop_duplicates().sort_values('y')['color']
    cumulative_returns = np.exp(logret.sum()) - 1
    color_codes_list = color_codes.values.tolist()
    bar_chart = go.Bar(
        x=cumulative_returns.values,
        y=np.array(dendrogram['layout']['yaxis']['tickvals']),
        marker=dict(color=color_codes_list),
        orientation='h'
    )
    fig_dendrogram = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        column_widths=[0.3, 0.7],
        horizontal_spacing=0.02,
        subplot_titles=("Dendrogram", "Cumulative Returns")
    )
    for trace in dendrogram['data']:
        fig_dendrogram.add_trace(trace, row=1, col=1)
    fig_dendrogram.add_trace(bar_chart, row=1, col=2)
    fig_dendrogram.update_layout(
        title='Mutual Funds Analysis: Dendrogram and Cumulative Returns',
        showlegend=False,
        height=800
    )
    fig_dendrogram.update_xaxes(title_text='Distance (1 - Correlation)', row=1, col=1)
    fig_dendrogram.update_xaxes(title_text='Cumulative Return', row=1, col=2)
    fig_dendrogram.update_yaxes(showticklabels=True, row=1, col=2, ticktext=distance_matrix.columns,
                                tickvals=np.array(dendrogram['layout']['yaxis']['tickvals']), side='right',
                                showgrid=True)
    fig_dendrogram.update_yaxes(showticklabels=False, row=1, col=1)
    st.write("Distance Matrix:\n", distance_matrix)
    st.write("Dendrogram Tickvals:\n", dendrogram['layout']['yaxis']['tickvals'])
    cumulative_returns = np.exp(logret.sum()) - 1
    st.write("Cumulative Returns:\n", cumulative_returns)
    return fig_dendrogram

def get_fig_rebate(df_funds):
    import plotly.graph_objects as go
    df_filtered = df_funds.drop('Ptf').sort_values('Rebate 12M EUR', ascending=False)
    fig_rebate = go.Figure()
    for idx, row in df_filtered.iterrows():
        fig_rebate.add_trace(go.Scatter(
            x=[row['Peso'], row['Peso']],
            y=[0, row['Rebate 12M EUR']],
            mode='lines',
            line=dict(color='#0c2340', width=3),
            opacity=0.6,
            showlegend=False
        ))
    fig_rebate.add_trace(go.Scatter(
        x=df_filtered['Peso'],
        y=df_filtered['Rebate 12M EUR'],
        mode='markers+text',
        marker=dict(size=20, color='#0c2340'),
        text=df_filtered['Descrizione'],
        textposition='middle right',
        showlegend=False
    ))
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

def get_fig_treemap(df_funds, df, color_map_mod):
    """
    Create a treemap figure showing breakdown by AC and Module.
    Args:
        df_funds (pd.DataFrame): DataFrame with fund data.
        df (pd.DataFrame): DataFrame for overall portfolio (for AC weights).
        color_map_mod (dict): Color map for modules.
    Returns:
        plotly.graph_objs._figure.Figure: The treemap figure.
    """
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
