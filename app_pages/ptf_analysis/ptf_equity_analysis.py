import streamlit as st
st.set_page_config(page_title="Ptf Equity Analysis", page_icon="💹")

def show_analysis_page(df, asset_class):
    st.title(f"Analysis for {asset_class}")
    st.write(df[df['L1'] == asset_class])

show_analysis_page(st.session_state.df_full, 'Equities')
