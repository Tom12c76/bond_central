import streamlit as st
st.set_page_config(page_title="Intel", page_icon="🧠")
import pandas as pd
import pyperclip
import re
import xlwings as xw

def main():
    # Get the calling workbook
    wb = xw.Book.caller()

    # Get the sheet named "output"
    sheet = wb.sheets['output']

    text_data = pyperclip.paste()
    text_data = text_data.split("Dettagli Cliente\r\n", 1)[1]
    lines = text_data.splitlines()

    extracted_fields_list = []

    for i, line in enumerate(lines):
        line = line.strip()
        print(repr(line))
        
        if not line:
            continue

        # --- Client Information ---
        if i==0:
            fields = line.split('\t')
            extracted_fields_list.append(('Cliente', fields[0].strip()))
            if fields[1] == 'Nato a:':
                extracted_fields_list.append(('Nato a', fields[2].strip()))
                extracted_fields_list.append(('Nato il', fields[3].replace("il:", "").strip()))
                extracted_fields_list.append(('Cod Fisc', fields[5].strip()))
                if i + 1 < len(lines):
                    extracted_fields_list.append(('NDG', lines[i+1].strip()))
            if fields[1] == 'Data Costituz.':
                extracted_fields_list.append(('Nato a', ''))
                extracted_fields_list.append(('Nato il', fields[2].strip()))
                extracted_fields_list.append(('Cod Fisc', 'CF '+fields[4].strip()))
                if i + 1 < len(lines):
                    extracted_fields_list.append(('NDG', lines[i+1].strip()))

        elif line.startswith("Indirizzo:"):
            fields = line.split('\t')
            extracted_fields_list.append(('Indirizzo', fields[1].strip()))
            extracted_fields_list.append(('Documento', fields[3].strip()))
            if len(fields)>5:
                extracted_fields_list.append(('Telefono', fields[5].strip()))
    # ...existing code...
