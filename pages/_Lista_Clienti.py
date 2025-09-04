import streamlit as st
import camelot
import pandas as pd
from pathlib import Path
from unidecode import unidecode
import PyPDF2

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
            # Get total number of pages using PyPDF2
            with open(temp_pdf_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                total_pages = len(reader.pages)
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
                df['source_file'] = uploaded_file.name

                # Map columns to unified structure
                # Individual format
                individual_cols = [
                    'Cognome', 'Nome', 'NDG', 'Classe', 'Tipo', 'Sportello',
                    'Data di Nascita', 'Luogo di Nascita', 'Residenza'
                ]
                # Company format
                company_cols = [
                    'Ragione Sociale', 'NDG', 'Classe', 'Tipo', 'Sportello',
                    'Data Costituz./Nascita', 'Sede Legale/Residenza'
                ]

                # Check which format
                if set(individual_cols).issubset(df.columns):
                    # Already in individual format
                    df['Tipo_Cliente'] = 'Persona Fisica'
                elif set(company_cols).issubset(df.columns):
                    # Map company columns to unified structure
                    df = df.rename(columns={
                        'Ragione Sociale': 'Cognome',
                        'Data Costituz./Nascita': 'Data di Nascita',
                        'Sede Legale/Residenza': 'Residenza'
                    })
                    # Fill missing columns for company
                    for col in ['Nome', 'Luogo di Nascita']:
                        if col not in df.columns:
                            df[col] = ''
                    df['Tipo_Cliente'] = 'Persona Giuridica'
                else:
                    st.warning(f"Table format not recognized in {uploaded_file.name}. Columns: {list(df.columns)}")

                # Ensure all required columns exist
                required_cols = [
                    'Cognome', 'Nome', 'NDG', 'Classe', 'Tipo', 'Sportello',
                    'Data di Nascita', 'Luogo di Nascita', 'Residenza',
                    'Tipo_Cliente', 'source_file', 'page_number'
                ]
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = ''
                # Reorder columns
                df = df[required_cols]
                all_tables.append(df)
                df['page_number'] = tables[t].page
                all_tables.append(df)
        st.info(f"Finished processing {uploaded_file.name}")
    if all_tables:
        st.success("All files processed successfully!")
        merged_df = pd.concat(all_tables, ignore_index=True)
        # Drop rows where Cognome == 'Collegato'
        merged_df = merged_df[merged_df['Cognome'] != 'Collegato']

        # Aggregate split records
        aggregated_rows = []
        current_record = None

        for idx, row in merged_df.iterrows():
            if pd.notna(row['NDG']) and str(row['NDG']).strip() != '':
                # Start a new record
                if current_record is not None:
                    aggregated_rows.append(current_record)
                current_record = row.copy()
            else:
                # Continuation line: concatenate text fields
                for col in ['Cognome', 'Nome', 'Residenza', 'Luogo di Nascita']:
                    if pd.notna(row[col]) and str(row[col]).strip() != '':
                        if pd.notna(current_record[col]):
                            current_record[col] = str(current_record[col]) + " " + str(row[col])
                        else:
                            current_record[col] = str(row[col])
        # Add the last record
        if current_record is not None:
            aggregated_rows.append(current_record)

        # Create new DataFrame
        aggregated_df = pd.DataFrame(aggregated_rows)
        st.dataframe(aggregated_df)
        # Save aggregated_df to CSV automatically
        aggregated_df.to_csv("merged_output.csv", index=False)
        st.balloons()
    else:
        st.write("No PDF tables found in the uploaded files.")