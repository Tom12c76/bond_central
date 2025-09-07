import streamlit as st
st.set_page_config(page_title="Lista Clienti", page_icon="👥")
import camelot
import pandas as pd
import os

# Use Streamlit file uploader to select multiple PDFs
uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

all_tables = []
if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        # Save uploaded file to a temporary location
        temp_pdf_path = os.path.join("temp_data", uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner(f"Processing {uploaded_file.name}..."):
            areas_p1 = ["5,505,835,35"]
            areas = ["5,550,835,35"]
            tables_p1 = camelot.read_pdf(
                temp_pdf_path, pages="1", flavor="stream",
                table_areas=areas_p1,
            )
            tables_rest = camelot.read_pdf(
                temp_pdf_path, pages="2-end", flavor="stream",
                table_areas=areas,
            )
            tables = camelot.core.TableList(list(tables_p1) + list(tables_rest))
            for t in range(len(tables)):
                df = tables[t].df.copy()
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
                
                # Add source file and page number
                df['source_file'] = uploaded_file.name
                df['page_number'] = tables[t].page

                # Column definitions
                individual_cols = [
                    'Cognome', 'Nome', 'NDG', 'Classe', 'Tipo', 'Sportello',
                    'Data di Nascita', 'Luogo di Nascita', 'Residenza'
                ]
                company_cols = [
                    'Ragione Sociale', 'NDG', 'Classe', 'Tipo', 'Sportello',
                    'Data Costituz./Nascita', 'Sede Legale/Residenza'
                ]

                # Check format and unify columns
                if set(individual_cols).issubset(df.columns):
                    df['Tipo_Cliente'] = 'Individual'
                    all_tables.append(df)
                elif set(company_cols).issubset(df.columns):
                    df.rename(columns={
                        'Ragione Sociale': 'Cognome',
                        'Data Costituz./Nascita': 'Data di Nascita',
                        'Sede Legale/Residenza': 'Residenza'
                    }, inplace=True)
                    df['Tipo_Cliente'] = 'Company'
                    df['Nome'] = ''
                    df['Luogo di Nascita'] = ''
                    all_tables.append(df)
                else:
                    st.warning(f"Unknown format in file {uploaded_file.name}, table {t+1} on page {tables[t].page}. Skipping.")
                    continue

    # Concatenate all tables into a single DataFrame
    if all_tables:
        final_df = pd.concat(all_tables, ignore_index=True)

        # Remove rows where 'Cognome' starts with 'Collegato'
        final_df = final_df[~final_df['Cognome'].astype(str).str.startswith('Collegato')]

        # Handle multi-line records by iterating through rows
        processed_rows = []
        for i, row in final_df.iterrows():
            # Check if NDG is present and not an empty string
            if pd.notna(row['NDG']) and str(row['NDG']).strip():
                processed_rows.append(row.to_dict())
            else:
                if processed_rows:
                    last_record = processed_rows[-1]
                    for field in ['Cognome', 'Nome', 'Luogo di Nascita', 'Residenza']:
                        current_val = row.get(field)
                        if pd.notna(current_val) and str(current_val).strip():
                            last_val = last_record.get(field)
                            if pd.notna(last_val) and str(last_val).strip():
                                last_record[field] = f"{last_val} {current_val}"
                            else:
                                last_record[field] = current_val
        
        final_df = pd.DataFrame(processed_rows)

        # Drop duplicate records
        final_df.drop_duplicates(inplace=True)

        # Define final column order
        final_columns = [
            'Cognome', 'Nome', 'NDG', 'Classe', 'Tipo', 'Sportello', 
            'Data di Nascita', 'Luogo di Nascita', 'Residenza', 
            'Tipo_Cliente', 'source_file', 'page_number'
        ]
        
        # Add missing columns and reorder
        for col in final_columns:
            if col not in final_df.columns:
                final_df[col] = None
        final_df = final_df[final_columns]

        # Download button for the consolidated Excel file
        excel_file = "temp_data/consolidated_data.xlsx"
        final_df.to_excel(excel_file, index=False)
        
        # Provide a downloadable link
        with open(excel_file, "rb") as f:
            st.download_button(
                label="Download Consolidated Data",
                data=f,
                file_name="consolidated_data.xlsx",
                mime="application/vnd.ms-excel"
            )

        # Display the consolidated table
        st.subheader("Consolidated Clienti Data")
        st.dataframe(final_df)

    else:
        st.info("No valid tables found in the uploaded PDFs.")

