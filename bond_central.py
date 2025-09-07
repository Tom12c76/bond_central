import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import time

# Set a default page config for the app
st.set_page_config(page_title="Bond Central", page_icon=":bar_chart:")

def home_page():
    def load_feather():
        # Define default and home feather directories
        default_dir = Path(r'C:\Users\A9T230\OneDrive - Deutsche Bank AG\_STAR\Feather')
        home_dir = Path(r'C:\Users\thoma\OneDrive\Lebenslauf\_DB\_base_case\feather')

        # Try default location first
        feather_files = [file.name for file in default_dir.iterdir() if file.suffix == '.feather'] if default_dir.exists() else []
        feather_dir = default_dir

        # If no files found, try home location
        if not feather_files:
            feather_files = [file.name for file in home_dir.iterdir() if file.suffix == '.feather'] if home_dir.exists() else []
            feather_dir = home_dir

        st.subheader("Select a feather database file")
        if feather_files:
            file_selection = st.selectbox("Select a Feather file to load:", feather_files, index=len(feather_files) - 1)

            if st.button("Load File"):
                file_path = feather_dir / file_selection
                df_full = pd.read_feather(file_path)
                st.session_state.df_full = df_full
                st.session_state.file_loaded = True
                st.success(f"File {file_selection} loaded successfully!")
                time.sleep(1)  # Wait for one second before hiding the widget
                st.rerun()  # Use rerun to refresh the app
        else:
            st.warning("No feather files found in either location.")

    def create_waterfall_chart(df):
        grouped_df = df.groupby('L1')['Abs Ctv EUR'].sum().sort_values(ascending=False).reset_index()
        grouped_df['Cumulative'] = grouped_df['Abs Ctv EUR'].cumsum()
        grouped_df['Start'] = grouped_df['Cumulative'] - grouped_df['Abs Ctv EUR']
        grouped_df['End'] = grouped_df['Cumulative']

        sorted_l1_names = grouped_df['L1'].tolist()

        chart = alt.Chart(grouped_df).mark_bar(color='#4ac9e3').encode(
            x=alt.X('Start:Q', title=None),
            x2='End:Q',
            y=alt.Y('L1:N', sort=sorted_l1_names, title=None),
            tooltip=[alt.Tooltip('L1:N', title='L1'),
                     alt.Tooltip('Abs Ctv EUR:Q', title='Abs Ctv EUR', format=',.0f'),
                     alt.Tooltip('End:Q', title='Total EUR', format=',.0f')]
        ).properties(
            title='Total Assets Overview',
            width=700,
            height=400
        )

        return chart

    # Initialize session state variables if not already set
    if 'file_loaded' not in st.session_state:
        st.session_state.file_loaded = False

    if not st.session_state.get('file_loaded'):
        load_feather()
    else:
        st.subheader("Choose which data to analyze")
        df_full = st.session_state.df_full

        unique_units = ['All'] + df_full['Unit'].unique().tolist()
        selected_unit = st.selectbox('Select Unit', unique_units, index=0, key='selected_unit')

        df_units = df_full if selected_unit == 'All' else df_full[df_full['Unit'] == selected_unit]

        unique_bankers = ['All'] + df_units['Banker'].unique().tolist()
        selected_banker = st.selectbox('Select Banker', unique_bankers, index=0, key='selected_banker')

        df_units_bankers = df_units if selected_banker == 'All' else df_units[
            df_units['Banker'] == selected_banker]

        unique_intestazioni = ['All'] + df_units_bankers.groupby('Intestazione')[
            'Abs Ctv EUR'].sum().sort_values(ascending=False).index.tolist()
        selected_intestazioni = st.multiselect('Select Intestazioni', unique_intestazioni, default=['All'], key='selected_intestazioni')

        df_filtered = df_units_bankers if 'All' in selected_intestazioni or not selected_intestazioni else \
            df_units_bankers[df_units_bankers['Intestazione'].isin(selected_intestazioni)]

        if not df_filtered.empty:
            waterfall_chart = create_waterfall_chart(df_filtered)
            st.write('')  # some spacing
            st.altair_chart(waterfall_chart, use_container_width=True)
            # Save filtered df to Session State
            st.session_state.df_filtered = df_filtered
        else:
            st.write("No data loaded yet.")

# Modern multipage navigation with sections and icons
pg = st.navigation({
    "Home": [st.Page(home_page, title="Home", icon="🏠", default=True)],
    "Ptf Analysis": [
        st.Page("app_pages/ptf_analysis/ptf_bond_analysis.py", title="Bond Analysis", icon="📊"),
        st.Page("app_pages/ptf_analysis/ptf_fund_charts.py", title="Fund Charts", icon="📈"),
        st.Page("app_pages/ptf_analysis/ptf_equity_analysis.py", title="Equity Analysis", icon="💸"),
        st.Page("app_pages/ptf_analysis/ptf_certificati.py", title="Certificati", icon="🏦"),
        st.Page("app_pages/ptf_analysis/ptf_clc.py", title="CLC", icon="📄"),
    ],
    "Tools": [
        st.Page("app_pages/tools/tool_lista_clienti.py", title="Lista Clienti", icon="👥"),
        st.Page("app_pages/tools/tool_pacman.py", title="Pacman", icon="🎮"),
        st.Page("app_pages/tools/tool_test.py", title="Test", icon="🧪"),
        # If you convert camelot_pdf.ipynb to .py, add it here:
        # st.Page("pages/tools/tool_camelot_pdf.py", title="Camelot PDF", icon=":page_with_curl:"),
        st.Page("app_pages/tools/tool_lat_long.py", title="Lat Long", icon="🌍"),
        st.Page("app_pages/tools/tool_intel.py", title="Intel", icon="🧠"),
    ]
})

# Run the selected page
pg.run()

