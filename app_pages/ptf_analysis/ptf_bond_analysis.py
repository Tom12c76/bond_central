import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import eikon as ek
import io
import xlwings as xw
import datetime

st.set_page_config(page_title="Ptf Bond Analysis", page_icon="📊")

# Eikon API key and set page configuration
ek.set_app_key('cf2eaf5e3b3c42adba08b3c5c2002b6ced1e77d7')
st.set_page_config(layout='wide')

ek_bond_fields = ['TR.ISIN', 'TR.FiIssuerISOLongCountryCode',  'TR.TRBCEconomicSector', 'TR.FiIssuerShortName',
 'TR.FiSeniorityType', 'TR.CouponRate', 'TR.FiCouponClass', 'TR.FiWorstYearsToRedem', 'TR.FiWorstRedemDate',
 'TR.FiWorstRedemEvent', 'TR.FICurrency', 'TR.FiPricingAskPrice', 'TR.FiPricingAskYield', 'TR.FiSPRating',
 'TR.FiMoodysRating', 'TR.FiFitchsRating', 'TR.FiWorstCorpModDuration', 'OAS_BID', 'TR.FIFaceOutstanding',
 'TR.FiDenominationMinimum', 'TR.FiDenominationIncrement', 'TR.FRNFORMULA', 'TR.FiDebtTypeDescription',
 'TR.FiGovernmentBondTypeDescription', 'TR.FiAccruedInterest', 'TR.PRICEFACT', 'TR.INDEXRATIO',
 'TR.FiIsIndexLinkedInterest', 'TR.FiIsIndexLinkedPrincipal']

cols_out = ['ISIN',
 'Quant',
 'Ask Price',
 'Accrued Interest',
 'MV',
 'Currency',
 'Issuer ISO Long Country Code',
 'TRBC Economic Sector Name',
 'Issuer Short Name',
 'Seniority Type',
 'Coupon Rate',
 'Coupon Class',
 'Worst Redem Date',
 'Worst Redem Event',
 'Worst Years To Redem',
 'Ask Yield',
 'Worst Corp Mod Duration',
 'OAS_BID',
 'SP Rating',
 'Moodys Rating',
 'Face Outstanding',
 'Denomination Minimum',
 'Denomination Increment',
 'FRN Formula',
 'Index Ratio',
 'Instrument']

# Initialize session state variables
if 'df_filtered' not in st.session_state:
    st.warning('HOUSTON: df_filtered not found in session state memory')
if 'bonds_ek' not in st.session_state:
    st.session_state.bonds_ek = pd.DataFrame(columns=['ISIN'])

# Function to fetch bond data from Eikon
def get_bonds_ek(bondlist):
    st.write(bondlist)
    pref_ric_dict, _ = ek.get_data(bondlist.ISIN.unique().tolist(), fields='TR.RICS(Contributor=RRPS)')
    # pref_ric_dict, _ = ek.get_data(bondlist.ISIN.unique().tolist(), fields='TR.PreferredRics')
    # pref_ric_dict, _ = ek.get_data(bondlist.ISIN.unique().tolist(), fields='TR.RICS(Contributor=1M)')
    # pref_ric_dict, _ = ek.get_data(bondlist.ISIN.unique().tolist(), fields='TR.RICS(Contributor=NGB)')
    # pref_ric_dict, _ = ek.get_data(bondlist.ISIN.unique().tolist(), fields='TR.RICS(Contributor=2M)')
    # pref_ric_dict, _ = ek.get_data(bondlist.ISIN.unique().tolist(), fields='TR.RICS(Contributor=MI)')
    # pref_ric_dict, _ = ek.get_data(bondlist.ISIN.unique().tolist(), fields='TR.RICS(Contributor=TT)')
    pref_ric_dict = pref_ric_dict.set_index('Instrument')['RICS'].to_dict()
    pref_isin_list = bondlist.ISIN.map(pref_ric_dict).fillna(bondlist.ISIN).unique().tolist()
    st.write(pref_isin_list)
    ek_data, err = ek.get_data(pref_isin_list, fields=ek_bond_fields)
    st.write('errors:')
    st.write(err)
    return ek_data, err

# Function to create the plot
def get_fig(bonds_ek):
    bonds_ek = bonds_ek.dropna(subset=['Worst Years To Redem', 'Ask Yield'])
    fig1 = px.scatter(
        bonds_ek.dropna(subset=['Worst Years To Redem', 'Ask Yield']),
        x='Worst Years To Redem',
        y='Ask Yield',
        color_discrete_sequence=['#4ac9e3'] * len(bonds_ek),  # Specify the color as red
        symbol='Divisa',
        size='Controvalore EUR',
        opacity=0.66,
        hover_name='Descrizione',
        hover_data={
            'Ask Yield': ':.2f',
            'Worst Years To Redem': ':.2f',
            'Issuer Short Name': True,
            'Seniority Type': True,
            'Coupon Rate': ':.2f',
            'Coupon Class': True,
            'Issuer ISO Long Country Code': True,
            'TRBC Economic Sector Name': True,
            'Worst Redem Date': True,
            'Worst Redem Event': True,
        },
        title='Bond Yields by Years to Maturity'
    )

    df2 = bonds_ek.sort_values('Worst Years To Redem').dropna(subset='Worst Years To Redem')
    fig2 = px.line(x=[0] + df2['Worst Years To Redem'].tolist(),
                   y=[0] + df2['Cum Redem'].tolist(),
                   line_shape='hv',
                   color_discrete_sequence=['#4ac9e3'] * len(bonds_ek))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.67, 0.33],
                        vertical_spacing=0.02)

    for trace in fig1['data']:
        fig.add_trace(trace, row=1, col=1)

    for trace in fig2['data']:
        fig.add_trace(trace, row=2, col=1)

    fig.update_layout(template='gridon',
                      xaxis = dict(range=[0, max(bonds_ek['Worst Years To Redem'])*1.05]),
                      yaxis = dict(range=[0, max(bonds_ek['Ask Yield'])*1.05]),
                      height=800
                      )

    return fig


##########################
### Main Streamlit App ###
##########################

asset_class = 'Bonds'
bonds = st.session_state.df_filtered.copy()
bonds = bonds[bonds['L1'] == asset_class]
bonds = bonds.dropna(subset=['Controvalore EUR'])

st.dataframe(bonds)

bonds = bonds.groupby(['ISIN', 'Descrizione', 'Divisa', 'Tipologia Titolo', 'Sub AC'])[['Quant', 'Controvalore EUR']] \
             .sum().reset_index().sort_values('Controvalore EUR', ascending=False)

# Filter bonds using Streamlit widgets
col1, col2, col3 = st.columns([1, 2, 3])
divisa = col1.selectbox('Divisa', options=bonds['Divisa'].unique().tolist(), index=0)
bonds_div = bonds[bonds['Divisa'] == divisa]

tipologia = col2.multiselect('Tipologia', options=bonds_div['Tipologia Titolo'].unique().tolist(),
                             default=bonds_div['Tipologia Titolo'].unique().tolist())
bonds_div_tipo = bonds_div[bonds_div['Tipologia Titolo'].isin(tipologia)]

sub_ac = col3.multiselect('Sub AC', options=bonds_div_tipo['Sub AC'].unique().tolist(),
                          default=bonds_div_tipo['Sub AC'].unique().tolist())
bonds_f = bonds_div_tipo[bonds_div_tipo['Sub AC'].isin(sub_ac)]

# Add a slider to keep only the top N bonds
if not bonds_f.empty:
    steps = list(range(0, len(bonds_f), 10))
    if len(bonds_f) not in steps:
        steps.append(len(bonds_f))
    top_n = st.select_slider('Select top N bonds to display',
                      options=steps, value=min(len(bonds_f), 10))

bonds_top_n = bonds_f.nlargest(top_n, 'Controvalore EUR')
bonds_top_n_isins = bonds_top_n.ISIN.unique()

# Determine items not yet mapped
unmapped_bonds = bonds_top_n
if not st.session_state.bonds_ek.empty:
    unmapped_bonds = bonds_top_n[~bonds_top_n['ISIN'].isin(st.session_state.bonds_ek['ISIN'])]

col11, col12 = st.columns(2)
col11.dataframe(
    bonds_top_n[['Descrizione', 'Controvalore EUR']].sort_values('Controvalore EUR', ascending=False).reset_index(
        drop=True))
col12.metric('Bonds selected (-unmapped)', f'{len(bonds_top_n):,.0f}', -len(unmapped_bonds))
col12.metric('Total Market Value', f'{bonds_top_n["Controvalore EUR"].sum():,.0f} EUR',
             f'{bonds_top_n["Controvalore EUR"].sum() / bonds_f["Controvalore EUR"].sum():,.0%}')

# Fetch new bond data in batches
if not unmapped_bonds.empty:
    if st.button('Get Eikon Bond Data'):
        ek_data, err = get_bonds_ek(unmapped_bonds)
        new_data = unmapped_bonds.merge(ek_data, on='ISIN', how='outer')
        new_data['MV'] = new_data['Quant'] * (new_data['Ask Price'] + new_data['Accrued Interest']) / 100
        # Append new data to session state without duplicates
        st.session_state.bonds_ek = pd.concat([st.session_state.bonds_ek, new_data]).drop_duplicates(
            subset='ISIN').reset_index(drop=True)
        st.rerun()

# Plot the data if available
if not st.session_state.bonds_ek.empty:
    bonds_to_plot = st.session_state.bonds_ek.loc[st.session_state.bonds_ek.ISIN.isin(bonds_top_n_isins)]
    bonds_to_plot['Cum Redem'] = bonds_to_plot.loc[:, ['Controvalore EUR', 'Worst Years To Redem']].sort_values(
        'Worst Years To Redem', ascending=True).drop('Worst Years To Redem',axis=1).cumsum()
    # fig = get_fig(bonds_to_plot)
    # st.plotly_chart(fig, theme=None)
    # fig.write_html('bonds_to_plot.html')
    st.dataframe(bonds_to_plot[cols_out])
    bonds_to_plot[cols_out].to_excel('bonds_to_plot.xlsx', index=False)

    st.write(bonds_to_plot[['Worst Years To Redem', 'Cum Redem']].sort_values('Worst Years To Redem'))

    if st.button('Save to Excel'):
        template_path = 'Lookup/output_template.xlsx'
        sheet_name = 'Advisory'
        range_name = 'advisory'

        # If the file is not open, open the template
        app = xw.App(visible=True)
        wb = app.books.open(template_path)

        # Get the specified sheet and range
        sheet = wb.sheets[sheet_name]

        # Clear any existing filters and data
        try:
            sheet.range(range_name).api.ListObject.AutoFilter.ShowAllData()
        except:
            pass
        try:
            sheet.range(range_name).api.ListObject.DataBodyRange.Delete()
        except:
            pass

        # Populate the range with DataFrame values
        sheet.range(range_name).value = bonds_to_plot[cols_out].values

        # Refresh all data connections and formulas
        wb.api.RefreshAll()

        # Get the current date and time
        current_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H_%M")

        # Save the workbook with the current date and time in the filename
        wb.save(f"Lookup/{current_date_time} Bond List.xlsx")