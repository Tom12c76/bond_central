import streamlit as st
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
            else:
                extracted_fields_list.append(('Telefono', ''))

        # --- Profiling Information ---
        elif line.startswith("Appropriatezza:"):
            fields = line.split('\t')
            extracted_fields_list.append(('Appropriatezza', fields[1].strip()))
            extracted_fields_list.append(('Adeguatezza', fields[3].strip()))
            extracted_fields_list.append(('Profilazione', fields[6].strip()))

        # --- Manager Information ---
        elif line.startswith("Gestore:"):
            fields = line.split('\t')
            extracted_fields_list.append(('Gestore', fields[1].strip()))
            extracted_fields_list.append(('Cod Gest', fields[3].strip()))
            extracted_fields_list.append(('Sportello', fields[5].strip()))

        # --- Contact Information (Recapiti) ---
        elif line.startswith("Telefono casa:"):
            match = re.match(r"Telefono casa:\s*(.*?)\s*E-mail casa:\s*(.*)", line)
            if match:
                extracted_fields_list.append(('Tel casa', match.group(1).strip()))
                extracted_fields_list.append(('E-mail casa', match.group(2).strip()))

        elif line.startswith("Telefono lavoro:"):
            print(repr(line))
            match = re.match(r"Telefono lavoro:\s*(.*?)\s*E-mail lavoro:\s*(.*)", line)
            if match:
                extracted_fields_list.append(('Tel lavoro', match.group(1).strip()))
                extracted_fields_list.append(('E-mail lavoro', match.group(2).strip()))

        elif line.startswith("Telefono altro:"):
            print(repr(line))
            match = re.match(r"Telefono altro:\s*(.*?)\s*E-mail altro:\s*(.*)", line)
            if match:
                extracted_fields_list.append(('Tel altro', match.group(1).strip()))
                extracted_fields_list.append(('E-mail altro', match.group(2).strip()))

        elif line.startswith("Indirizzi"):
            break


    # --- Convert the list of tuples into a pandas DataFrame ---

    # We can create a DataFrame where the first element of the tuple is the index
    # and the second element is the value in a single column.
    # Let's name the value column something generic, like 'Value'.

    if extracted_fields_list:
        # Create DataFrame using from_records
        # The index_col parameter is 0 (the first element of the tuple)
        # The columns parameter specifies the names for the columns based on the tuple elements
        df = pd.DataFrame.from_records(extracted_fields_list, columns=['Key', 'Value'], index='Key')

        # The resulting DataFrame will have 'Key' as the index and 'Value' as the column
        # If you prefer the key as a regular column and the value in another:
        # df = pd.DataFrame.from_records(extracted_fields_list, columns=['Field', 'Value'])

        print("\n--- Extracted Data DataFrame ---")
        
        
    else:
        print("No data fields extracted.")


    parsed_df = df.T.set_index('NDG')
    parsed_df['Nato il'] = pd.to_datetime(parsed_df['Nato il'], dayfirst=True)
    parsed_df['Profilazione'] = pd.to_datetime(parsed_df['Profilazione'], dayfirst=True)

    sheet.tables['parsed_data'].range.address.split('$')
    _, first_col, _, _, last_row = sheet.tables['parsed_data'].range.address.split('$')
    paste_here = first_col + str(int(last_row)+1)
    sheet.range(paste_here).options(pd.DataFrame, index=True, header=False).value = parsed_df

if __name__ == "__main__":
    xw.Book("parse.xlsm").set_mock_caller()
    main()