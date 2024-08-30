# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
from PIL import Image


# Since data is already worked on in the Notebook, we can use the latest csv file
df_clean = pd.read_csv('bank-final.csv')

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Data Cleaning","Data Visualization", "Machine Learning"],
        icons=["house", "clipboard-data", "book", "robot"],
        menu_icon="cast",
        default_index=0,
    )

if selected == 'Home':
    st.title(':bank: Bank Marketing Overview')
    st.markdown("***")
    image = Image.open('Images/bank.jpg')
    st.image(image, use_column_width=True)
    st.write('The data is related with direct marketing campaigns *(phone calls)* of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit *(variable y)*.:')
    st.write("""
            ## <u> Input variables: </u>
            ### bank client data:
            - age (numeric)
            - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
            blue-collar","self-employed","retired","technician","services") 
            - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
            - education (categorical: "unknown","secondary","primary","tertiary")
            - default: has credit in default? (binary: "yes","no")
            - balance: average yearly balance, in euros (numeric) 
            - housing: has housing loan? (binary: "yes","no")
            - loan: has personal loan? (binary: "yes","no")
            ### related with the last contact of the current campaign:
            - contact: contact communication type (categorical: "unknown","telephone","cellular") 
            - day: last contact day of the month (numeric)
            - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
            - duration: last contact duration, in seconds (numeric)
            ### other attributes:
            - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
            - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
            - previous: number of contacts performed before this campaign and for this client (numeric)
            - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

            ## <u> Output variable (desired target): </u>
            - y - has the client subscribed a term deposit? (binary: "yes","no")

            """)
    st.markdown("***")
    st.write('The data we will use for this demo is the following:')
    st.write(df_clean.head(5))