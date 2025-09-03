import streamlit as st

def show_analysis_page(df, asset_class):
    st.title(f"Analysis for {asset_class}")
    st.write(df[df['L1'] == asset_class])


show_analysis_page(st.session_state.df_full, 'Equities')