import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
from functools import reduce
import pathlib

st.set_page_config('FIS2', page_icon="🏰", layout='wide')
def title(url):
     st.markdown(f'<p style="color:#2f0d86;font-size:22px;border-radius:2%;"><br><br><br>{url}</p>', unsafe_allow_html=True)
def title_main(url):
     st.markdown(f'<h1 style="color:#230c6e;font-size:42px;border-radius:2%;"><br>{url}</h1>', unsafe_allow_html=True)

def success_email(html_str):
    html_str = f"""
        <p style='background-color:#baffc9;
        color: #313131;
        font-size: 15px;
        border-radius:5px;
        padding-left: 12px;
        padding-top: 10px;
        padding-bottom: 12px;
        line-height: 18px;
        border-color: #03396c;
        text-align: left;'>
        {html_str}</style>
        <br></p>"""
    st.markdown(html_str, unsafe_allow_html=True)

with open('./wave.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #fff5ee;
        }
    </style>
    """, unsafe_allow_html=True)

sidebar = st.sidebar
with sidebar:
    #st.title("FIS2 Container Status")
    title("FIS2 Container Status")
    st.write('## Menu')
    City = st.radio(
        label='Select the formula',
        options=['AP', 'RE', 'PE'],
        index=0
    )

    for i in range(16):
        st.text("")

    col1, col2 = st.columns(2)

    #with col1:
        #st.link_button("My Github", "https://github.com/JohnTan38/House-Price-Prediction")

    #with col2:
        #st.link_button(
            #"My Linkedin", "https://www.linkedin.com/in/")

if City == 'RE':

    #st.title('Formula RE 📊')
    title_main("Formula RE 📊")

    tab1, tab2 = st.tabs(['Formula RE', 'Chat with data'])

    with tab1:

        uploaded_file = st.file_uploader("Upload SKY HAP APPROVAL GENERATOR", type=["xlsx"])
        if uploaded_file is None:
            st.write("Please upload a file")
        elif uploaded_file:
            list_ws = ['DMS INVENTORY', 'REPAIR ESTIMATE', 'MOVEMENT OUT', 'AP', 'RE', 'PE']
            cols_dmsInventory = ['Container No.', 'Customer', 'Size/Type', 'Current Status']
            cols_repairEstimate = ['Container No', 'Customer', 'EOR Status', 'Surveyor Name', 'Total'] #HAPAG LLOYD (S) PTE LTD
            cols_movementOut = ['Container No.', 'Customer', 'Status']
            cols_ap = ['Container', 'Type', 'Source']
            cols_re = ['Container', 'Source']
            cols_pe = ['Container', 'Source']

            polar_dmsInventory = pl.read_excel(uploaded_file, sheet_name=list_ws[0], raise_if_empty=True, engine='openpyxl')
            dmsInventory = polar_dmsInventory.to_pandas(use_pyarrow_extension_array=True)
            polar_repairEstimate = pl.read_excel(uploaded_file, sheet_name=list_ws[1], raise_if_empty=True, 
                                                      read_options={'infer_schema_length':int(1e10)}, engine='openpyxl')                                               
            repairEstimate = polar_repairEstimate.to_pandas(use_pyarrow_extension_array=True)
            movementOut = pd.read_excel(uploaded_file, sheet_name=list_ws[2], engine='openpyxl')
            #movementOut = polar_movementOut.to_pandas(use_pyarrow_extension_array=True)
            ap = pd.read_excel(uploaded_file, sheet_name=list_ws[3], dtype=str, engine='openpyxl')
            re = pd.read_excel(uploaded_file, sheet_name=list_ws[4], dtype=str, engine='openpyxl')
            pe = pd.read_excel(uploaded_file, sheet_name=list_ws[5], dtype=str, engine='openpyxl')

            dmsInventory = dmsInventory[cols_dmsInventory]
            repairEstimate = repairEstimate[cols_repairEstimate]
            movementOut = movementOut[cols_movementOut]
            ap = ap[cols_ap]
            re = re[cols_re]
            pe = pe[cols_pe]

            mask_inventory = (dmsInventory['Customer'].isin(['HAP'])) #2
            mask_repairEstimate_0 = (repairEstimate['Customer'].isin(['HAPAG LLOYD (S) PTE LTD']))
            mask_repairEstimate_1 = (repairEstimate['Surveyor Name'].isin(['MD GLOBAL', 'Eastern Repairer', 'Abu Zaar Amiruddin Bin Jasri', ' ']))
            mask_movementOut = (movementOut['Customer'].isin(['HAP']))
            assert mask_inventory.any()
            assert mask_repairEstimate_0.any()
            assert mask_repairEstimate_1.any()
            assert mask_movementOut.any()

            dmsInventory_hap = dmsInventory[mask_inventory]
            repairEstimate_hap_0 = repairEstimate[mask_repairEstimate_0]
            repairEstimate_hap = repairEstimate_hap_0[mask_repairEstimate_1]
            movementOut_hap = movementOut[mask_movementOut]
            repairEstimate_hap['Customer'] = 'HAP' #replace 'HAPAG LLOYD (S) PTE LTD' with 'HAP'

            def remove_whitespace_from_list(lst):    
                return [element.replace(" ", "").strip() for element in lst]

            lst_ap = remove_whitespace_from_list(ap['Container'].tolist())
            lst_re = remove_whitespace_from_list(re['Container'].tolist())
            lst_pe = remove_whitespace_from_list(pe['Container'].tolist())

            mask_re_inventory = (dmsInventory_hap['Container No.'].isin(lst_re)) #select rows based on list values
            re_inventory = dmsInventory_hap[mask_re_inventory]
            re_inventory.rename(columns={'Container No.':'Container No'}, inplace=True)
            re_inventory = re_inventory.sort_values(by='Container No', ascending=True)
            
            mask_re_repair = (repairEstimate_hap['Container No'].isin(lst_re))
            re_repair = repairEstimate_hap[mask_re_repair]
            re_repair = re_repair.sort_values(by='Container No', ascending=True)
            re_repair.drop_duplicates(subset=['Container No'], keep='first', inplace=True)
            
            mask_re_movement = (movementOut_hap['Container No.'].isin(lst_re))
            re_movement = movementOut_hap[mask_re_movement]
            re_movement = re_movement.sort_values(by='Container No.', ascending=True)
            re_movement.rename(columns={'Container No.':'Container No'}, inplace=True)
            #print(re_movement) # 'dmsInventory' NOT IN DMS

            re_movement['dmsInventory'] = 'NOT IN DMS' #3

            #from functools import reduce
            lst_df = [re_inventory, re_repair, re_movement]
            merged_df = reduce(lambda left,right: pd.merge(left,right,on=['Container No'], how='outer'), lst_df)
            merged_df = merged_df.sort_values(by='Container No', ascending=True)
            
            merged_df_copy = merged_df.copy()

            def add_repair_completed_column(df):
    
                # Apply the conditions
                df['RepairCompleted'] = df.apply(lambda row: '-' if row['Surveyor Name'] == None else row['Surveyor Name'], axis=1)
                return df

            merged_df_1 = add_repair_completed_column(merged_df_copy) #4

            #5
            def replace_nan_with_dash(df, column_name):
    
                df[column_name].fillna('-', inplace=True)
                return df
            merged_df_2 = replace_nan_with_dash(merged_df_1, 'Status')

            def add_movement_out_column(df):
    
                # Apply the conditions
                df['MovementOut'] = df.apply(lambda row: 'NO GATE OUT' if row['Status'] == '-' else row['Status'], axis=1)
                return df

            merged_df_3 = add_movement_out_column(merged_df_2) #5
            
            merged_df_4 = replace_nan_with_dash(merged_df_3, 'dmsInventory') #6

            def add_dms_inventory_column(df):
    
                # Apply the conditions
                df['dmsInventory_new'] = df.apply(lambda row: row['Current Status'] if row['dmsInventory'] == '-' else row['dmsInventory'], axis=1)
                return df
            merged_df_5 = add_dms_inventory_column(merged_df_4)

            #7
            def process_size_values(df):
    
                processed_df = df.copy() # Create a copy of the original DataFrame

                # Replace specific values in the 'Size' column
                processed_df['Size/Type'].replace({
                    '45R1': '45RT',
                    '22G1': '22GP'
                }, inplace=True)

                return processed_df[['Container No', 'Size/Type', 'dmsInventory_new', 'RepairCompleted', 'MovementOut']] #7
            merged_df_6 = process_size_values(merged_df_5)
            merged_df_6.rename(columns={'Container No': 'FIS2 RE Units', 'Size/Type':'FIS2', 'dmsInventory_new': 'dmsInventory'}, inplace=True)
            merged_df_6.fillna("-", inplace=True)
            merged_df_6.drop_duplicates(keep='last', inplace=True)

        #col1, col2 = st.columns(2)

        #with col1:

            #area = st.selectbox(
                #label='Area',
                #options=['Karapakkam', 'Anna Nagar', 'Adyar'
                         #],
                #index=None
            #)

        #with col2:

            #sqft = st.number_input(
                #label='SquareFeet',
                #min_value=500,
                #max_value=3000,
                #value=None
            #)

        #col3, col4 = st.columns(2)

        #with col3:

            #bedroom = st.slider(
                #label='No of Bedrooms',
                #min_value=1,
                #max_value=10,
                #step=1,
                #value=None
            #)

        #with col4:

            #bathroom = st.slider(
                #label='No of Bathrooms',
                #min_value=1,
                #max_value=5,
                #step=1,
                #value=None
            #)

        #col5, col6 = st.columns(2)

        #with col5:

            #n_room = st.number_input(
                #label='Total Rooms',
                #min_value=2,
                #max_value=20,
                #value=None
            #)

        #with col6:

            #age = st.number_input(
                #label='Age of the property',
                #min_value=1,
                #max_value=100,
                #value=None
            #)

        #col7, col8 = st.columns(2)

        #with col7:

            #utility_avail = st.selectbox(
                #label='Utilities',
                #options=['AllPub', 'ELO', 'NoSewr ', 'NoSeWa'],
                #index=None
            #)

        #with col8:

            #street = st.selectbox(
                #label='Type of alley access to the property',
                #options=['Paved', 'Gravel', 'No Access'],
                #index=None
            #)

        #col9, col10 = st.columns(2)

        #with col9:

            #sale_cond = st.selectbox(
                #label='Condition of sale',
                #options=['AbNormal', 'Family',
                         #'Partial', 'AdjLand', 'Normal Sale'],
                #index=None
            #)

        #with col10:

            #mzzone = st.selectbox(
                #label='General zoning classification of the sale',
                #options=['A', 'RH', 'RL', 'I', 'C', 'RM'],
                #index=None
            #)

        #col11, col12 = st.columns(2)

        #with col11:

            #buildtype = st.selectbox(
                #label='Purpose of House',
                #options=['Commercial', 'House', 'Others'],
                #index=None
            #)

        #with col12:

            #park_facil = st.selectbox(
                #label='Parking Facility',
                #options=['Yes', 'No'],
                #index=None
            #)

        #col13, col14 = st.columns(2)

        #with col13:

            #reg_fee = st.number_input(
                #label='Registration fee',
                #min_value=50000,
                #max_value=2000000,
                #value=None
            #)

        #with col14:

            #commition = st.number_input(
                #label='Commition',
                #min_value=5000,
                #max_value=1000000,
                #value=None
            #)

        st.divider()

        st.write('Click to get dataframe RE')

        if st.button("Get RE"):
            with st.spinner("Processing..."):
                #from Chennai_HPP_py import pred_chennai
                #price = pred_chennai(area, sale_cond, park_facil, buildtype, utility_avail,
                                    #street, mzzone, sqft, bedroom, bathroom, n_room, reg_fee, commition, age)
                #price = 1
                merged_df_6 = merged_df_6.astype(str)
                st.dataframe(merged_df_6, use_container_width=True)
                st.success("Dataframe is ready 💸")
                

        st.divider()

        #userEmail = st.text_input('Enter your email')
        col1, col2 = st.columns(2)
        #with col1:

            #userEmail = st.selectbox(
                #label='Your Email',
                #options=['abu.zaar@sh-cogent.com.sg', 'gopi.jaganathan@sh-cogent.com.sg', 'grace.lim@sh-cogent.com.sg', 'john.tan@sh-cogent.com.sg'],
                #index=None
            #)
            #userEmail = st.text_input('Enter your email', type='default')
            #import re
            #def validate_email(email):
                #pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
                #pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+(\.[a-zA-Z]{2,})?$'
                #if re.match(pattern, email):
                    #return True
                #else:
                    #return False
            #if validate_email(userEmail):
                #userEmail = userEmail
            #else:
                #st.write('Please enter a valid email address')

        #with col2:

            #cc_lst = st.multiselect(
                #label='CC',
                #options=['abu.zaar@sh-cogent.com.sg', 'gopi.jaganathan@sh-cogent.com.sg', 'john.tan@sh-cogent.com.sg'],
                
                #key=str
            #)

        #if st.button('Email to users'):
            #userEmail = st.text_input('Enter your email')
            #import win32com.client as win32
            #import sys
            #import pandas as pd
            #import pathlib
            #import glob
            #from datetime import datetime
            #import pythoncom

            #outlook = win32.Dispatch("Outlook.Application", pythoncom.CoInitialize())#.GetNamespace("MAPI")
            #pythoncom.CoInitialize()
            #inbox = outlook.Folders(email).Folders("Inbox")

            #def sendmail(receiver, attachments, subject, body, cc_lst):
                #"""Sends an Outlook email.

                #Args:
                #receiver (str): The email address of the recipient.
                #attachments (list): A list of file paths to attachments.
                #subject (str): The subject of the email.
                #body (str): The body of the email in HTML format.
                #"""
                #outlook = win32.Dispatch("Outlook.Application")
                #mail = outlook.CreateItem(0)
                #mail.To = receiver
                #mail.Subject = subject
                #mail.HTMLBody = body
                #mail.CC = ";".join(cc_lst)

                #for attachment in attachments:
                    #attachment = path_skyHAP+attachment
                    #mail.Attachments.Add(attachment)
                #mail.Send()

            
            #receiver = userEmail
            #attachments = attachments
            #attachments = []
            #files_xl = [f for f in pathlib.Path().iterdir() if f.is_file()] #files_xl = glob.glob(path_dms+'*.xlsx')
            #for file in files_xl:
                #if 'Container_Already_Approved.xlsx' in file.name:
                    #attachments.append(file.name)       
                #if 'Container_Not_Approve.xlsx' in file.name:
                    #attachments.append(file.name)
        
            #cc = pd.read_excel(path_skyHAP+ r'CC.xlsx', sheet_name='dlpCC', engine='openpyxl')
            #cc_lst = cc['add'].tolist()
            #cc_lst = ['john.tan@sh-cogent.com.sg', 'abu.zaar@sh-cogent.com.sg']

            #subject = "Formula RE "+ datetime.now().strftime("%Y%m%d %H:%M")
            #body = """
                    #<html>
                    #<head>
                    #<title>Dear User</title>
                    #</head>
                    #<body>
                    #<p style="color: blue;font-size:25px;">Please note FIS2 RE Units.</strong><br></p>

                    #</body>
                    #</html>

                    #"""+ merged_df_6.reset_index(drop=True).to_html() +"""
        
                    #<br>This message is computer generated. """+ datetime.now().strftime("%Y%m%d %H:%M:%S")
        
            #sendmail(receiver, attachments, subject, body, cc_lst)
            #st.success("Email sent successfully")
            #success_email("Email sent successfully 🌈")
    st.markdown(f'<h1 style="color:#190303;font-size:16px;"><br><br><br><br><br><br><br>{"All Rights Reserved 2024 Cogent Holdings IT"}</h1>', unsafe_allow_html=True)

    with tab2:
        st.write(
            """Begin by uploading SKY HAP APPROVAL GENERATOR (RE) csv file. App automatically recognizes column headers and infer data types.
               Business rules based on specific user requirements are translated to code logic.
               App processes the data and generates clean, organized datatables. 
               You can view these datatables online, chat with data or create a visualization.
        """
        )

        st.divider()
        import sqlite3
        from sqlite3 import Connection
        #from openai import OpenAI
        import openai
        
        import plotly.express as px
        import plotly.graph_objs as go
        import re
        from dateutil.parser import parse
        import traceback

        def create_connection(db_name: str) -> Connection:
            conn = sqlite3.connect(db_name)
            return conn

        def run_query(conn: Connection, query: str) -> pd.DataFrame:
            df = pd.read_sql_query(query, conn)
            print(df)
            return df

        def create_table(conn: Connection, df: pd.DataFrame, table_name: str):
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        def generate_gpt_reponse(gpt_input, max_tokens):
    
            #openai_api_key = st.secrets["openai_api_key"]
            openai.api_key = openai_api_key
            #client = OpenAI(openai_api_key)
            completion = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                max_tokens=max_tokens,
                temperature=0,
                messages=[
                    {"role": "user", "content": gpt_input},
                ]
            )

            gpt_response = completion.choices[0].message['content'].strip()
            print(gpt_response)
            return gpt_response

        def extract_code(gpt_response):
            """function to extract code and sql query from gpt response"""

            if "```" in gpt_response:
                # extract text between ``` and ```
                pattern = r'```(.*?)```'
                code = re.search(pattern, gpt_response, re.DOTALL)
                extracted_code = code.group(1)

                # remove sql and python from the code (weird bug)
                for word in ['sql', 'python']:
                    if extracted_code.lower().startswith(word):
                        extracted_code = extracted_code[len(word):]
                        #extracted_code = extracted_code.replace('python', '')
                print(extracted_code)

                return extracted_code.strip() # strip leading and trailing whitespaces
            else:
                return gpt_response

        with st.sidebar:
            openai_api_key = st.text_input("YOUR_OPENAI_API_KEY", type="password")
            openai.api_key = openai_api_key

        uploaded_file = st.file_uploader("Upload CSV file", type=['csv']) #merged_df_4.reset_index(drop=True)
        if uploaded_file is None:
            st.info(f""" Please upload a csv file""")

        elif uploaded_file:
            df = pd.read_csv(uploaded_file)

            for col in df.columns:
            # check if a column name contains date substring
                if 'date' in col.lower():
                    df[col] = pd.to_datetime(df[col])
            df = df.reset_index(drop=True)
            df.columns = df.columns.str.replace(' ', '_')

            cols = df.columns
            cols = ", ".join(cols)

            with st.expander("Preview data"):
                st.table(df.reset_index(drop=True).head())

            conn = create_connection(":memory:")
            table_name = "my_table"
            create_table(conn, df, table_name)

            selected_mode = st.selectbox("What do you want to do?", ["Chat with data", "Create a visualization"])
            if selected_mode == 'Chat with data':
                user_input = st.text_area("Write a concise and clear question about your data. For example: What is the total sales in the USA in 2022?", value='What is the total sales in the USA in 2022?')
                if st.button("Get Response"):
                    try:
                        # create gpt prompt
                        gpt_input = 'Write a SQLite query based on this question: {} The table name is my_table and the table has the following columns: {}. ' \
                                    'Return only a SQLite query and nothing else'.format(user_input, cols)

                        query = generate_gpt_reponse(gpt_input, max_tokens=1000)
                        query_clean = extract_code(query)
                        result = run_query(conn, query_clean)

                        with st.expander("SQL query used"):
                            st.code(query_clean)
                        # if result df has one row and one column
                        if result.shape == (1, 1):
                            # get the value of the first row of the first column
                            val = result.iloc[0, 0]
                            # write one liner response
                            st.subheader('Your response: {}'.format(val))
                        else:
                            st.subheader("Your result 🌈")
                            st.table(result)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        
            if selected_mode == 'Create a visualization':
                user_input = st.text_area(
                    "Briefly explain what you want to plot from your data. For example: Plot total sales by country and product category", value='Plot total sales by country and product category')
                if st.button("Create a visualization"):
                    try:
                        # create gpt prompt
                        gpt_input = 'Write code in Python using Plotly to address the following request: {} ' \
                                'Use df that has the following columns: {}. Do not use animation_group argument and return only code with no import statements, use transparent background, the data has been already loaded in a df variable'.format(user_input, cols)

                        gpt_response = generate_gpt_reponse(gpt_input, max_tokens=2000)
                        extracted_code = extract_code(gpt_response)
                        extracted_code = extracted_code.replace('fig.show()', 'st.plotly_chart(fig)')
                        with st.expander("Code used"):
                            st.code(extracted_code)
                        # execute code
                        exec(extracted_code)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")



elif City == 'AP':

    st.title('Formula AP 📊')

    tab1, tab2 = st.tabs(['Formula AP', 'Chat with data'])

    with tab1:

        uploaded_file = st.file_uploader("Upload SKY HAP APPROVAL GENERATOR", type=["xlsx"])
        if uploaded_file is None:
            st.write("Please upload a file")
        elif uploaded_file:
            list_ws = ['DMS INVENTORY', 'REPAIR ESTIMATE', 'MOVEMENT OUT', 'AP', 'RE', 'PE']
            cols_dmsInventory = ['Container No.', 'Customer', 'Size/Type', 'Current Status']
            cols_repairEstimate = ['Container No', 'Customer', 'EOR Status', 'Surveyor Name', 'Total'] #HAPAG LLOYD (S) PTE LTD
            cols_movementOut = ['Container No.', 'Customer', 'Status']
            cols_ap = ['Container', 'Type', 'Source']
            cols_re = ['Container', 'Source']
            cols_pe = ['Container', 'Source']

            polar_dmsInventory = pl.read_excel(uploaded_file, sheet_name=list_ws[0], raise_if_empty=True, engine='openpyxl')
            dmsInventory = polar_dmsInventory.to_pandas(use_pyarrow_extension_array=True)
            polar_repairEstimate = pl.read_excel(uploaded_file, sheet_name=list_ws[1], raise_if_empty=True, 
                                                      read_options={'infer_schema_length':int(1e10)}, engine='openpyxl')                                               
            repairEstimate = polar_repairEstimate.to_pandas(use_pyarrow_extension_array=True)
            movementOut = pd.read_excel(uploaded_file, sheet_name=list_ws[2], engine='openpyxl')
            #movementOut = polar_movementOut.to_pandas(use_pyarrow_extension_array=True)
            ap = pd.read_excel(uploaded_file, sheet_name=list_ws[3], dtype=str, engine='openpyxl')
            re = pd.read_excel(uploaded_file, sheet_name=list_ws[4], dtype=str, engine='openpyxl')
            pe = pd.read_excel(uploaded_file, sheet_name=list_ws[5], dtype=str, engine='openpyxl')

            dmsInventory = dmsInventory[cols_dmsInventory]
            repairEstimate = repairEstimate[cols_repairEstimate]
            movementOut = movementOut[cols_movementOut]
            ap = ap[cols_ap]
            re = re[cols_re]
            pe = pe[cols_pe]

            mask_inventory = (dmsInventory['Customer'].isin(['HAP'])) #2
            mask_repairEstimate = (repairEstimate['Customer'].isin(['HAPAG LLOYD (S) PTE LTD']))
            #mask_repairEstimate_1 = (repairEstimate['Surveyor Name'].isin(['MD GLOBAL', 'Eastern Repairer', 'Abu Zaar Amiruddin Bin Jasri', ' ']))
            mask_movementOut = (movementOut['Customer'].isin(['HAP']))
            assert mask_inventory.any()
            assert mask_repairEstimate.any()            
            assert mask_movementOut.any()

            dmsInventory_hap = dmsInventory[mask_inventory]
            repairEstimate_hap = repairEstimate[mask_repairEstimate]            
            movementOut_hap = movementOut[mask_movementOut]
            repairEstimate_hap['Customer'] = 'HAP' #replace 'HAPAG LLOYD (S) PTE LTD' with 'HAP'

            def remove_whitespace_from_list(lst):    
                return [element.replace(" ", "").strip() for element in lst]

            lst_ap = remove_whitespace_from_list(ap['Container'].tolist())
            lst_re = remove_whitespace_from_list(re['Container'].tolist())
            lst_pe = remove_whitespace_from_list(pe['Container'].tolist())

            mask_ap_inventory = (dmsInventory_hap['Container No.'].isin(lst_ap)) #select rows based on list values
            ap_inventory = dmsInventory_hap[mask_ap_inventory]
            ap_inventory = ap_inventory.sort_values(by='Container No.', ascending=True)

            mask_ap_repair = (repairEstimate_hap['Container No'].isin(lst_ap))
            ap_repair = repairEstimate_hap[mask_ap_repair]
            ap_repair = ap_repair.sort_values(by='Container No', ascending=True)
            ap_repair.drop_duplicates(subset=['Container No'], keep='first', inplace=True)

            mask_ap_movement = (movementOut_hap['Container No.'].isin(lst_ap))
            ap_movement = movementOut_hap[mask_ap_movement]
            ap_movement = ap_movement.sort_values(by='Container No.', ascending=True)
            ap_movement.rename(columns={'Container No.':'Container No'}, inplace=True)

            #3
            def add_repair_completed_column(df):
                #"""
                #Adds a new column 'RepairCompleted' to the DataFrame based on conditions:
                #1. If 'EOR Status' == 'Pending Repair', set 'RepairCompleted' to an empty string.
                #2. Otherwise, set 'RepairCompleted' to the value of 'EOR Status'.
                #Args:
                    #df (pd.DataFrame): Input DataFrame with columns 'Container' and 'EOR Status'.
                #Returns:
                    #pd.DataFrame: DataFrame with additional 'RepairCompleted' column.
                #"""
                # Apply the conditions
                try:
                     df['RepairCompleted'] = df.apply(lambda row: '-' if pd.notna(row['EOR Status']) and row['EOR Status'] == 'Pending Repair' else row['EOR Status'], axis=1)
                     #df.fillna(0)
                     return df
                except Exception as e:
                     st.write(e)
                return df

            # Apply the function #3
            try:
                 repairCompleted_df = add_repair_completed_column(ap_repair)
            except Exception as e:
                 repairCompleted_df = ap_repair
            #repairCompleted_df = ap_repair     
            #4
            repairCompleted_movement = pd.merge(repairCompleted_df, ap_movement, on='Container No', how='left') #4
            def replace_nan_with_dash(df, column_name):
                
                df[column_name].fillna('-', inplace=True)
                return df
            replace_nan_with_dash(repairCompleted_movement, 'Status')

            def add_movement_out_column(df):
                """
                Adds a new column 'MovementOut' to the DataFrame based on conditions:
                1. If 'Status' == 'Pending', set 'MovementOut' to an empty string.
                2. Otherwise, set 'MovementOut' to the value of 'Status'.
                Args:
                    df (pd.DataFrame): Input DataFrame with columns 'Container' and 'Status'.
                Returns:
                    pd.DataFrame: DataFrame with additional 'MovementOut' column.
                """
                # Apply the conditions
                df['MovementOut'] = df.apply(lambda row: 'NO GATE OUT' if row['Status'] == '-' else row['Status'], axis=1)
                return df

            formulaAP_0 = add_movement_out_column(repairCompleted_movement)
            #formulaAP_0
            ap_inventory.rename(columns={'Container No.': 'Container No', 'Current Status': 'dmsInventory'}, inplace=True)
            ap_inventory.drop(columns=['Customer'], inplace=True)

            #5
            def add_dms_inventory_column(df1, df2):
                """
                Adds a new column 'dmsInventory' to df1 based on the specified conditions:
                If 'Container' in df1 is not in 'Container' from df2, set 'dmsInventory' to 'NOT IN DEPOT'.
                Otherwise, set 'dmsInventory' to the value of 'Current Status' in df1.
                Args:
                    df1 (pd.DataFrame): First DataFrame with columns 'Container' and 'Current Status'.
                    df2 (pd.DataFrame): Second DataFrame with column 'Container'.
                Returns:
                    pd.DataFrame: Resulting DataFrame with columns 'Container' and 'dmsInventory'.
                """
                df1['dmsInventory'] = df1.apply(lambda row: 'NOT IN DEPOT' if row['Container No'] not in df2['Container No'].values else row['Status'], 
                                        axis=1)
                res_df = df1[['Container No', 'dmsInventory']]
                return res_df

            not_in_depot_df = add_dms_inventory_column(ap_movement, ap_inventory) #5

            ap_inventory_not_in_depot = pd.concat([ap_inventory, not_in_depot_df])
            ap_inventory_not_in_depot.sort_values(by='Container No', ascending=True)

            formulaAP = pd.merge(formulaAP_0, ap_inventory_not_in_depot, on='Container No', how='left')
            formulaAP.drop(columns=['Customer_x', 'Customer_y', 'Status'], inplace=True)
            formulaAP.rename(columns={'Total': 'Amount', 'Size/Type': 'Size'}, inplace=True)

            #6
            #def process_repair_completed(df):
                #"""
                #Process the 'RepairCompleted' column in the given DataFrame.
                #- Replace 'Pending Customer' and 'Complete' with corresponding value from 'Surveyor Name' column.('MD GLOBAL').
                #- Keep '-' unchanged.
                #Returns a new DataFrame with only 'Container No' and 'RepairCompleted' columns.
                #"""
                #processed_df = df.copy() # Create a copy of the original DataFrame

                # Replace values in the 'RepairCompleted' column
                #try:
                     
                    #processed_df['RepairCompleted'] = processed_df.apply(lambda row: row['Surveyor Name'] if row['RepairCompleted'] in ['Pending Customer', 'Complete'] else row['RepairCompleted'],
                    #axis=1)
                #except Exception as e:
                    #st.write(e)
                #try:     
                    #return processed_df[['Container No', 'RepairCompleted']] # Return a DataFrame with only relevant columns
                #except:
                     #pass
            #formulaAP_repair = process_repair_completed(formulaAP) #6
            #try:
                 
                #formulaAP_1 = pd.merge(formulaAP, formulaAP_repair, on='Container No', how='left').drop(columns=['RepairCompleted_x'])
            #except:
                 #formulaAP_1 = formulaAP
            formulaAP_1 = formulaAP 
            #7
            def process_size_values(df):                
                processed_df = df.copy() # Create a copy of the original DataFrame

                # Replace specific values in the 'Size' column
                processed_df['Size'].replace({
                    '45R1': '45RT',
                    '45G1': '45GP',
                    '22G1': '22GP'
                }, inplace=True)

                #return processed_df[['Container No', 'Size', 'dmsInventory', 'RepairCompleted_y', 'MovementOut', 'Amount', 
                            #'EOR Status', 'Surveyor Name']] # Return a DataFrame with only relevant columns
                return processed_df[['Container No', 'Size', 'dmsInventory', 'MovementOut', 'Amount', 
                            'EOR Status']] 

            formulaAP_final = process_size_values(formulaAP) #7
            formulaAP_final.rename(columns={'Container No': 'FIS2 AP Units', 'RepairCompleted_y': 'RepairCompleted'}, inplace=True)
            formulaAP_final.fillna("-", inplace=True)
            formulaAP_final.drop_duplicates(keep='last', inplace=True)

            st.divider()

            st.write('Click to get dataframe AP')

            if st.button("Get AP"):
                with st.spinner("Processing..."):
                    #from Chennai_HPP_py import pred_chennai
                    #price = pred_chennai(area, sale_cond, park_facil, buildtype, utility_avail,
                                    #street, mzzone, sqft, bedroom, bathroom, n_room, reg_fee, commition, age)
                    #price = 1
                    formulaAP_final = formulaAP_final.astype(str)
                    st.dataframe(formulaAP_final, use_container_width=True)
                    st.success("Dataframe is ready 💸")
                    #st.write("## The Price of the House is : 💸", price)

            st.divider()

            #userEmail = st.text_input('Enter your email')
            col1, col2 = st.columns(2)

            #userEmail = st.text_input('Enter your email', type='default')
            #import re
            def validate_email(email):
                #pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
                pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+(\.[a-zA-Z]{2,})?$'
                if re.match(pattern, email):
                    return True
                else:
                    return False
            #if validate_email(userEmail):
                #userEmail = userEmail
            #else:
                #st.write('Please enter a valid email address')
            
            #with col1:

                #userEmail = st.selectbox(
                    #label='Your Email',
                    #options=['abu.zaar@sh-cogent.com.sg', 'gopi.jaganathan@sh-cogent.com.sg', 'grace.lim@sh-cogent.com.sg', 'john.tan@sh-cogent.com.sg'],
                    #index=None
                #)
            #with col2:

                #cc_lst = st.multiselect(
                    #label='CC',
                    #options=['abu.zaar@sh-cogent.com.sg', 'gopi.jaganathan@sh-cogent.com.sg', 'grace.lim@sh-cogent.com.sg', 'john.tan@sh-cogent.com.sg'],
                
                    #key=str
                #)

            #if st.button('Email to users'):
                #userEmail = st.text_input('Enter your email')
                #import win32com.client as win32
                #import sys
                #import pandas as pd
                #import pathlib
                #import glob
                #from datetime import datetime
                #import pythoncom

                #outlook = win32.Dispatch("Outlook.Application", pythoncom.CoInitialize())#.GetNamespace("MAPI")
                #pythoncom.CoInitialize()
                #inbox = outlook.Folders(email).Folders("Inbox")

                #def sendmail(receiver, attachments, subject, body, cc_lst):
                    
                    #outlook = win32.Dispatch("Outlook.Application")
                    #mail = outlook.CreateItem(0)
                    #mail.To = receiver
                    #mail.Subject = subject
                    #mail.HTMLBody = body
                    #mail.CC = ";".join(cc_lst)

                    #for attachment in attachments:
                        #attachment = path_skyHAP+attachment
                        #mail.Attachments.Add(attachment)
                    #mail.Send()

                #receiver = userEmail
                #attachments = attachments
                #attachments = []
                #files_xl = [f for f in pathlib.Path().iterdir() if f.is_file()] #files_xl = glob.glob(path_dms+'*.xlsx')
                #for file in files_xl:
                    #if 'Container_Already_Approved.xlsx' in file.name:
                        #attachments.append(file.name)       
                    #if 'Container_Not_Approve.xlsx' in file.name:
                        #attachments.append(file.name)
        
                #cc = pd.read_excel(path_skyHAP+ r'CC.xlsx', sheet_name='dlpCC', engine='openpyxl')
                #cc_lst = cc['add'].tolist()
                #cc_lst = ['john.tan@sh-cogent.com.sg', 'abu.zaar@sh-cogent.com.sg']

                #subject = "Formula AP "+ datetime.now().strftime("%Y%m%d %H:%M")
                #body = """
                        #<html>
                        #<head>
                        #<title>Dear User</title>
                        #</head>
                        #<body>
                        #<p style="color: blue;font-size:25px;">Please note FIS2 AP Units.</strong><br></p>

                        #</body>
                        #</html>

                        #"""+ formulaAP_final.reset_index(drop=True).to_html() +"""
        
                        #<br>This message is computer generated. """+ datetime.now().strftime("%Y%m%d %H:%M:%S")
        
                #sendmail(receiver, attachments, subject, body, cc_lst)
                #st.success("Email sent successfully")
                #success_email("Email sent successfully 🌈")
    st.markdown(f'<h1 style="color:#190303;font-size:16px;"><br><br><br><br><br><br><br>{"All Rights Reserved 2024 Cogent Holdings IT"}</h1>', unsafe_allow_html=True)

    with tab2:
        st.write(
            """Begin by uploading SKY HAP APPROVAL GENERATOR (AP) csv file. App automatically recognizes column headers and infer data types.
               Business rules based on specific user requirements are translated to code logic.
               App processes the data and generates clean, organized datatables. 
               You can view these datatables online, chat with data or create a visualization.
        """
        )
        st.divider()

        import sqlite3
        from sqlite3 import Connection
        #from openai import OpenAI
        import openai
        
        import plotly.express as px
        import plotly.graph_objs as go
        import re
        from dateutil.parser import parse
        import traceback

        def create_connection(db_name: str) -> Connection:
            conn = sqlite3.connect(db_name)
            return conn

        def run_query(conn: Connection, query: str) -> pd.DataFrame:
            df = pd.read_sql_query(query, conn)
            print(df)
            return df

        def create_table(conn: Connection, df: pd.DataFrame, table_name: str):
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        def generate_gpt_reponse(gpt_input, max_tokens):
    
            #openai_api_key = st.secrets["openai_api_key"]
            openai.api_key = openai_api_key
            #client = OpenAI(openai_api_key)
            completion = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                max_tokens=max_tokens,
                temperature=0,
                messages=[
                    {"role": "user", "content": gpt_input},
                ]
            )

            gpt_response = completion.choices[0].message['content'].strip()
            print(gpt_response)
            return gpt_response

        def extract_code(gpt_response):
            """function to extract code and sql query from gpt response"""

            if "```" in gpt_response:
                # extract text between ``` and ```
                pattern = r'```(.*?)```'
                code = re.search(pattern, gpt_response, re.DOTALL)
                extracted_code = code.group(1)

                # remove sql and python from the code (weird bug)
                for word in ['sql', 'python']:
                    if extracted_code.lower().startswith(word):
                        extracted_code = extracted_code[len(word):]
                        #extracted_code = extracted_code.replace('python', '')
                print(extracted_code)

                return extracted_code.strip() # strip leading and trailing whitespaces
            else:
                return gpt_response

        with st.sidebar:
            openai_api_key = st.text_input("YOUR_OPENAI_API_KEY", type="password")
            openai.api_key = openai_api_key

        uploaded_file = st.file_uploader("Upload CSV file", type=['csv']) #merged_df_4.reset_index(drop=True)
        if uploaded_file is None:
            st.info(f""" Please upload a csv file""")

        elif uploaded_file:
            df = pd.read_csv(uploaded_file)

            for col in df.columns:
            # check if a column name contains date substring
                if 'date' in col.lower():
                    df[col] = pd.to_datetime(df[col])
            df = df.reset_index(drop=True)
            df.columns = df.columns.str.replace(' ', '_')

            cols = df.columns
            cols = ", ".join(cols)

            with st.expander("Preview data"):
                st.table(df.reset_index(drop=True).head())

            conn = create_connection(":memory:")
            table_name = "my_table"
            create_table(conn, df, table_name)

            selected_mode = st.selectbox("What do you want to do?", ["Chat with data", "Create a visualization"])
            if selected_mode == 'Chat with data':
                user_input = st.text_area("Write a concise and clear question about your data. For example: What is the total sales in the USA in 2022?", value='What is the total sales in the USA in 2022?')
                if st.button("Get Response"):
                    try:
                        # create gpt prompt
                        gpt_input = 'Write a SQLite query based on this question: {} The table name is my_table and the table has the following columns: {}. ' \
                                    'Return only a SQLite query and nothing else'.format(user_input, cols)

                        query = generate_gpt_reponse(gpt_input, max_tokens=1000)
                        query_clean = extract_code(query)
                        result = run_query(conn, query_clean)

                        with st.expander("SQL query used"):
                            st.code(query_clean)
                        # if result df has one row and one column
                        if result.shape == (1, 1):
                            # get the value of the first row of the first column
                            val = result.iloc[0, 0]
                            # write one liner response
                            st.subheader('Your response: {}'.format(val))
                        else:
                            st.subheader("Your result 🌈")
                            st.table(result)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        
            if selected_mode == 'Create a visualization':
                user_input = st.text_area(
                    "Briefly explain what you want to plot from your data. For example: Plot total sales by country and product category", value='Plot total sales by country and product category')
                if st.button("Create a visualization"):
                    try:
                        # create gpt prompt
                        gpt_input = 'Write code in Python using Plotly to address the following request: {} ' \
                                'Use df that has the following columns: {}. Do not use animation_group argument and return only code with no import statements, use transparent background, the data has been already loaded in a df variable'.format(user_input, cols)

                        gpt_response = generate_gpt_reponse(gpt_input, max_tokens=2000)
                        extracted_code = extract_code(gpt_response)
                        extracted_code = extracted_code.replace('fig.show()', 'st.plotly_chart(fig)')
                        with st.expander("Code used"):
                            st.code(extracted_code)
                        # execute code
                        exec(extracted_code)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")


elif City == 'PE':

    st.title('Formula PE 📊')

    tab1, tab2 = st.tabs(['Formula PE', 'Chat with data'])

    with tab1:

        uploaded_file = st.file_uploader("Upload SKY HAP APPROVAL GENERATOR", type=["xlsx"])
        if uploaded_file is None:
            st.write("Please upload a file")
        elif uploaded_file:
            list_ws = ['DMS INVENTORY', 'REPAIR ESTIMATE', 'MOVEMENT OUT', 'AP', 'RE', 'PE']
            cols_dmsInventory = ['Container No.', 'Customer', 'Size/Type', 'Current Status']
            cols_repairEstimate = ['Container No', 'Customer', 'EOR Status', 'Surveyor Name', 'Total'] #HAPAG LLOYD (S) PTE LTD
            cols_movementOut = ['Container No.', 'Customer', 'Status']
            cols_ap = ['Container', 'Type', 'Source']
            cols_re = ['Container', 'Source']
            cols_pe = ['Container', 'Source']

            polar_dmsInventory = pl.read_excel(uploaded_file, sheet_name=list_ws[0], raise_if_empty=True, engine='openpyxl')
            dmsInventory = polar_dmsInventory.to_pandas(use_pyarrow_extension_array=True)
            polar_repairEstimate = pl.read_excel(uploaded_file, sheet_name=list_ws[1], raise_if_empty=True, 
                                                      read_options={'infer_schema_length':int(1e10)}, engine='openpyxl')                                               
            repairEstimate = polar_repairEstimate.to_pandas(use_pyarrow_extension_array=True)
            movementOut = pd.read_excel(uploaded_file, sheet_name=list_ws[2], engine='openpyxl')
            #movementOut = polar_movementOut.to_pandas(use_pyarrow_extension_array=True)
            ap = pd.read_excel(uploaded_file, sheet_name=list_ws[3], dtype=str, engine='openpyxl')
            re = pd.read_excel(uploaded_file, sheet_name=list_ws[4], dtype=str, engine='openpyxl')
            pe = pd.read_excel(uploaded_file, sheet_name=list_ws[5], dtype=str, engine='openpyxl')

            dmsInventory = dmsInventory[cols_dmsInventory]
            repairEstimate = repairEstimate[cols_repairEstimate]
            movementOut = movementOut[cols_movementOut]
            ap = ap[cols_ap]
            re = re[cols_re]
            pe = pe[cols_pe]

            mask_inventory = (dmsInventory['Customer'].isin(['HAP'])) #2
            mask_repairEstimate_0 = (repairEstimate['Customer'].isin(['HAPAG LLOYD (S) PTE LTD']))
            mask_repairEstimate_1 = (repairEstimate['Surveyor Name'].isin(['MD GLOBAL', 'Pearlie Loh Mei Wah']))
            mask_movementOut = (movementOut['Customer'].isin(['HAP']))
            assert mask_inventory.any()
            assert mask_repairEstimate_0.any()            
            assert mask_movementOut.any()

            dmsInventory_hap = dmsInventory[mask_inventory]
            repairEstimate_hap_0 = repairEstimate[mask_repairEstimate_0]
            repairEstimate_hap = repairEstimate_hap_0[mask_repairEstimate_1]
            movementOut_hap = movementOut[mask_movementOut]
            repairEstimate_hap['Customer'] = 'HAP' #replace 'HAPAG LLOYD (S) PTE LTD' with 'HAP'

            def remove_whitespace_from_list(lst): 
                """
                Removes leading and trailing whitespaces from each element in the input list.
                Args:
                    lst (list): List of strings.
                Returns:
                    list: List of strings with whitespaces removed.
                """
                return [element.replace(" ", "").strip() for element in lst]

            lst_ap = remove_whitespace_from_list(ap['Container'].tolist())
            lst_re = remove_whitespace_from_list(re['Container'].tolist())
            lst_pe = remove_whitespace_from_list(pe['Container'].tolist())

            mask_pe_inventory = (dmsInventory_hap['Container No.'].isin(lst_pe)) #select rows based on list values
            pe_inventory = dmsInventory_hap[mask_pe_inventory]
            pe_inventory.rename(columns={'Container No.':'Container No'}, inplace=True)
            pe_inventory = pe_inventory.sort_values(by='Container No', ascending=True)
            
            mask_pe_repair = (repairEstimate_hap['Container No'].isin(lst_pe))
            pe_repair = repairEstimate_hap[mask_pe_repair]
            pe_repair = pe_repair.sort_values(by='Container No', ascending=True)
            pe_repair.drop_duplicates(subset=['Container No'], keep='first', inplace=True)
            
            mask_pe_movement = (movementOut_hap['Container No.'].isin(lst_pe))
            pe_movement = movementOut_hap[mask_pe_movement]
            pe_movement = pe_movement.sort_values(by='Container No.', ascending=True)
            pe_movement.rename(columns={'Container No.':'Container No'}, inplace=True)
            #print(pe_movement) # 'dmsInventory' NOT IN DMS


            pe_movement['dmsInventory'] ='NOT IN DMS' #3
            pe_inventory.rename(columns={'Current Status':'dmsInventory'}, inplace=True)

            from functools import reduce
            lst_df = [pe_inventory, pe_repair, pe_movement]
            merged_df = reduce(lambda left,right: pd.merge(left,right,on=['Container No'], how='outer'), lst_df)
            merged_df = merged_df.sort_values(by='Container No', ascending=True)
            #merged_df

            lst_container_repair = repairEstimate['Container No'].tolist() #4
            lst_container_merged = merged_df['Container No'].tolist()
            #compare 2 lists
            def compare_lists(lst_container_merged, lst_container_repair):
                """
                Compares elements of lst_container_merged with lst_container_repair.
                If an element from lst_container_merged is in lst_container_repair, return '-'.
                Otherwise, return 'NO REPAIR'.
                """
                result = []
                for container in lst_container_merged:
                    if container in lst_container_repair:
                        result.append('-')
                    else:
                        result.append('NO REPAIR')
                return result


            repair_noRepair = compare_lists(lst_container_merged, lst_container_repair) # Output: ['-', 'NO REPAIR', '-', 'NO REPAIR']
            merged_df['repairCompleted_1'] = repair_noRepair

            #5
            def replace_nan_with_dash(df, column_name):
                
                df[column_name].fillna('-', inplace=True)
                return df
            merged_df = replace_nan_with_dash(merged_df, 'Surveyor Name')

            def update_repair_completed(df):
                """
                Update the 'repairCompleted_1' column in the given DataFrame.
                - If 'Surveyor Name' is '-', assign the value of 'Surveyor Name' to 'repairCompleted_1'.
                - Otherwise, keep 'repairCompleted_1' unchanged.
                Returns the modified DataFrame.
                """
                updated_df = df.copy() # Create a copy of the original DataFrame

                # Update 'repairCompleted_1' based on the condition
                updated_df['repairCompleted_1'] = updated_df.apply(
                    lambda row: row['Surveyor Name'] if (row['Surveyor Name'] != '-') else row['repairCompleted_1'],
                    axis=1
                )
                return updated_df

            merged_df_1 = update_repair_completed(merged_df) #5

            #6
            merged_df_1 = replace_nan_with_dash(merged_df_1, 'Status') #6

            def add_movement_out_column(df):
                """
                Adds a new column 'MovementOut' to the DataFrame based on conditions:
                1. If 'Status' == '-', set 'MovementOut' to an empty string.
                2. Otherwise, set 'MovementOut' to the value of 'Status'.
                Args:
                    df (pd.DataFrame): Input DataFrame with columns 'Container' and 'Status'.
                Returns:
                    pd.DataFrame: DataFrame with additional 'MovementOut' column.
                """
                # Apply the conditions
                df['MovementOut'] = df.apply(lambda row: 'NO GATE OUT' if row['Status'] == '-' else row['Status'], axis=1)
                return df

            merged_df_2 = add_movement_out_column(merged_df_1)

            #7
            merged_df_2 = replace_nan_with_dash(merged_df_2, 'dmsInventory_y') #7

            def update_dmsinventory_column(df):
                """
                Update the 'dmsInventory_y' column in the given DataFrame.
                - If 'dmsInventory_y' is '-', assign the value of 'dmsInventory_x' to 'dmsInventory_y'.
                - Otherwise, keep 'dmsInventory_y' unchanged.
                Returns the modified DataFrame.
                """
                updated_df = df.copy() # Create a copy of the original DataFrame

                # Update 'dmsInventory_y' based on the condition
                updated_df['dmsInventory_y'] = updated_df.apply(
                    lambda row: row['dmsInventory_x'] if (row['dmsInventory_y'] == '-') else row['dmsInventory_y'],
                    axis=1
                )
                return updated_df

            merged_df_3 = update_dmsinventory_column(merged_df_2)

            #8
            def process_size_values(df):
                """
                Process the 'Size' column in the given DataFrame.
                - Replace '45R1' with '45RT'.
                - Replace '22G1' with '22GP'.
                Returns a new DataFrame with only 'Container No' and 'Size' columns.
                """
                processed_df = df.copy() # Create a copy of the original DataFrame

                # Replace specific values in the 'Size' column
                processed_df['Size/Type'].replace({
                    '45R1': '45RT',
                    '45G1': '45GP',
                    '22G1': '22GP'
                }, inplace=True)

                return processed_df[['Container No', 'Size/Type', 'dmsInventory_y', 'repairCompleted_1', 'MovementOut']]

            merged_df_4 = process_size_values(merged_df_3) #8
            merged_df_4.rename(columns={'Container No': 'FIS2 PE Units', 'Size/Type': 'FIS2 PE Size', 'dmsInventory_y': 'dmsInventory', 'repairCompleted_1': 'RepairCompleted'}, 
                               inplace=True)
            merged_df_4.drop_duplicates(keep='last', inplace=True)
            
            
            st.divider()

            st.write('Click to get dataframe PE')

            if st.button("Get PE"):
                with st.spinner("Processing..."):                    
                    merged_df_4 = merged_df_4.astype(str)
                    st.dataframe(merged_df_4, use_container_width=True)
                    st.success("Dataframe is ready 💸")
                    
            st.divider()

            #userEmail = st.text_input('Enter your email')
            #col1, col2 = st.columns(2)
            #userEmail = st.text_input('Enter your email', type='default')
            #import re
            def validate_email(email):
                #pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
                pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+(\.[a-zA-Z]{2,})?$'
                if re.match(pattern, email):
                    return True
                else:
                    return False
            #if validate_email(userEmail):
                #userEmail = userEmail
            #else:
                #st.write('Please enter a valid email address')

            #with col1:

                #userEmail = st.selectbox(
                    #label='Your Email',
                    #options=['abu.zaar@sh-cogent.com.sg', 'gopi.jaganathan@sh-cogent.com.sg', 'grace.lim@sh-cogent.com.sg', 'john.tan@sh-cogent.com.sg'],
                    #index=None
                #)
            #with col2:

                #cc_lst = st.multiselect(
                    #label='CC',
                    #options=['abu.zaar@sh-cogent.com.sg', 'gopi.jaganathan@sh-cogent.com.sg', 'grace.lim@sh-cogent.com.sg', 'john.tan@sh-cogent.com.sg'],
                
                    #key=str
                #)

            #if st.button("Send Email to users"):
                #import win32com.client as win32
                #import sys
                #import pandas as pd
                #import pathlib
                #import glob
                #from datetime import datetime
                #import pythoncom

                #outlook = win32.Dispatch("Outlook.Application", pythoncom.CoInitialize())#.GetNamespace("MAPI")
                #inbox = outlook.Folders(email).Folders("Inbox")
                #pythoncom.CoInitialize

                #def sendmail(receiver, attachments, subject, body, cc_lst):
                    
                    #outlook = win32.Dispatch("Outlook.Application")
                    #mail = outlook.CreateItem(0)
                    #mail.To = receiver
                    #mail.Subject = subject
                    #mail.HTMLBody = body
                    #mail.CC = ";".join(cc_lst)

                    #for attachment in attachments:
                        #attachment = path_skyHAP+attachment
                        #mail.Attachments.Add(attachment)
                    #mail.Send()

                #receiver = userEmail
                #attachments = attachments
                #attachments = []
                #files_xl = [f for f in pathlib.Path().iterdir() if f.is_file()] #files_xl = glob.glob(path_dms+'*.xlsx')
                #for file in files_xl:
                    #if 'Container_Already_Approved.xlsx' in file.name:
                        #attachments.append(file.name)       
                    #if 'Container_Not_Approve.xlsx' in file.name:
                        #attachments.append(file.name)
        
                #cc = pd.read_excel(path_skyHAP+ r'CC.xlsx', sheet_name='dlpCC', engine='openpyxl')
                #cc_lst = cc['add'].tolist()
                #cc_lst = ['john.tan@sh-cogent.com.sg']#, 'abu.zaar@sh-cogent.com.sg']

                #subject = "Formula PE "+ datetime.now().strftime("%Y%m%d %H:%M")
                #body = """
                        #<html>
                        #<head>
                        #<title>Dear User</title>
                        #</head>
                        #<body>
                        #<p style="color: blue;font-size:25px;">Please note FIS2 PE Units.</strong><br></p>

                        #</body>
                        #</html>

                        #"""+ merged_df_4.reset_index(drop=True).to_html() +"""
        
                        #<br>This message is computer generated. """+ datetime.now().strftime("%Y%m%d %H:%M:%S")
        
                #sendmail(receiver, attachments, subject, body, cc_lst)
                #st.success("Email sent successfully")
                #success_email("Email sent successfully 🌈")
    st.markdown(f'<h1 style="color:#190303;font-size:16px;"><br><br><br><br><br><br><br>{"All Rights Reserved 2024 Cogent Holdings IT"}</h1>', unsafe_allow_html=True)
                

    with tab2:

        st.write(
            """Begin by uploading SKY HAP APPROVAL GENERATOR (PE) csv file. App automatically recognizes column headers and infer data types.
               Business rules based on specific user requirements are translated to code logic.
               App processes the data and generates clean, organized datatables. 
               You can view these datatables online, chat with data or create a visualization.

        """
        )

        st.divider()
        import sqlite3
        from sqlite3 import Connection
        #from openai import OpenAI
        import openai
        
        import plotly.express as px
        import plotly.graph_objs as go
        import re
        from dateutil.parser import parse
        import traceback

        def create_connection(db_name: str) -> Connection:
            conn = sqlite3.connect(db_name)
            return conn

        def run_query(conn: Connection, query: str) -> pd.DataFrame:
            df = pd.read_sql_query(query, conn)
            print(df)
            return df

        def create_table(conn: Connection, df: pd.DataFrame, table_name: str):
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        def generate_gpt_reponse(gpt_input, max_tokens):
    
            #openai_api_key = st.secrets["openai_api_key"]
            openai.api_key = openai_api_key
            #client = OpenAI(openai_api_key)
            completion = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                max_tokens=max_tokens,
                temperature=0,
                messages=[
                    {"role": "user", "content": gpt_input},
                ]
            )

            gpt_response = completion.choices[0].message['content'].strip()
            print(gpt_response)
            return gpt_response

        def extract_code(gpt_response):
            """function to extract code and sql query from gpt response"""

            if "```" in gpt_response:
                # extract text between ``` and ```
                pattern = r'```(.*?)```'
                code = re.search(pattern, gpt_response, re.DOTALL)
                extracted_code = code.group(1)

                # remove sql and python from the code (weird bug)
                for word in ['sql', 'python']:
                    if extracted_code.lower().startswith(word):
                        extracted_code = extracted_code[len(word):]
                        #extracted_code = extracted_code.replace('python', '')
                print(extracted_code)

                return extracted_code.strip() # strip leading and trailing whitespaces
            else:
                return gpt_response

        with st.sidebar:
            openai_api_key = st.text_input("YOUR_OPENAI_API_KEY", type="password")
            openai.api_key = openai_api_key

        uploaded_file = st.file_uploader("Upload CSV file", type=['csv']) #merged_df_4.reset_index(drop=True)
        if uploaded_file is None:
            st.info(f""" Please upload a csv file""")

        elif uploaded_file:
            df = pd.read_csv(uploaded_file)

            for col in df.columns:
            # check if a column name contains date substring
                if 'date' in col.lower():
                    df[col] = pd.to_datetime(df[col])
            df = df.reset_index(drop=True)
            df.columns = df.columns.str.replace(' ', '_')

            cols = df.columns
            cols = ", ".join(cols)

            with st.expander("Preview data"):
                st.table(df.head())

            conn = create_connection(":memory:")
            table_name = "my_table"
            create_table(conn, df, table_name)

            selected_mode = st.selectbox("What do you want to do?", ["Chat with data", "Create a visualization"])
            if selected_mode == 'Chat with data':
                user_input = st.text_area("Write a concise and clear question about your data. For example: What is the total sales in the USA in 2022?", value='What is the total sales in the USA in 2022?')
                if st.button("Get Response"):
                    try:
                        # create gpt prompt
                        gpt_input = 'Write a SQLite query based on this question: {} The table name is my_table and the table has the following columns: {}. ' \
                                    'Return only a SQLite query and nothing else'.format(user_input, cols)

                        query = generate_gpt_reponse(gpt_input, max_tokens=1000)
                        query_clean = extract_code(query)
                        result = run_query(conn, query_clean)

                        with st.expander("SQL query used"):
                            st.code(query_clean)
                        # if result df has one row and one column
                        if result.shape == (1, 1):
                            # get the value of the first row of the first column
                            val = result.iloc[0, 0]
                            # write one liner response
                            st.subheader('Your response: {}'.format(val))
                        else:
                            st.subheader("Your result 🌈")
                            st.table(result)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        
            if selected_mode == 'Create a visualization':
                user_input = st.text_area(
                    "Briefly explain what you want to plot from your data. For example: Plot total sales by country and product category", value='Plot total sales by country and product category')
                if st.button("Create a visualization"):
                    try:
                        # create gpt prompt
                        gpt_input = 'Write code in Python using Plotly to address the following request: {} ' \
                                'Use df that has the following columns: {}. Do not use animation_group argument and return only code with no import statements, use transparent background, the data has been already loaded in a df variable'.format(user_input, cols)

                        gpt_response = generate_gpt_reponse(gpt_input, max_tokens=2000)
                        extracted_code = extract_code(gpt_response)
                        extracted_code = extracted_code.replace('fig.show()', 'st.plotly_chart(fig)')
                        with st.expander("Code used"):
                            st.code(extracted_code)
                        # execute code
                        exec(extracted_code)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")


