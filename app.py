# Import necessary libraries
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import plotly.express as px
from PIL import Image
import pickle

def format_date(date):
    # Format the date as "dd MMM yyyy"
    return date.strftime("%d %b %Y")

# Since data is already worked on in the Notebook, we can use the latest csv file
df_old = pd.read_csv('bank-full.csv', sep=';')
df_clean = pd.read_csv('bank-final.csv')

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Data Cleaning", "Data Visualization", "Machine Learning"],
        icons=["house", "clipboard-data", "book","robot"],
        menu_icon="cast",
        default_index=0,
    )

##########################################################################################################################################

if selected == 'Home':
    st.title(':bank: Bank Marketing Overview')
    st.markdown("***")
    image = Image.open('Images/bank.jpg')
    st.image(image, use_column_width=True)
    st.write('The data is related with direct marketing campaigns *(phone calls)* of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit *(variable y)*.:')
    st.write("""
            ## Input variables: 
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

            ## Output variable (desired target):
            - y - has the client subscribed a term deposit? (binary: "yes","no")

            """)
    st.markdown("***")
    st.write('The data we will use for this demo is the following:')
    st.write(df_clean.head(5))

##########################################################################################################################################

if selected == 'Data Visualization':
    st.title(':bar_chart: Data Visualization')
    st.markdown("***")
    st.markdown("<h2 style='text-align: center; color: black;'><u>I- Univariate Analysis</u></h2>"
                , unsafe_allow_html=True)
    uni_select = st.selectbox(
            'Choose a column to plot:',
            df_clean.select_dtypes(include='number').columns
        )
    st.write(px.histogram(df_clean[uni_select], x=uni_select))

    st.markdown("***")
    st.markdown("<h2 style='text-align: center; color: black;'><u>II- Bivariate Analysis</u></h2>"
                , unsafe_allow_html=True)
    st.write('<h4>1. Plotting the relationship between Selected Column and term_deposit</h4>', unsafe_allow_html=True)
    bi_select = st.selectbox(
            'Choose a column to plot:',
            df_clean.select_dtypes(include='O').columns
        )
    #st.write(px.histogram(df_clean[bi_select], x=uni_select))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="custom-left-align">', unsafe_allow_html=True)
        st.plotly_chart(px.box(df_clean, x='term_deposit', y=bi_select, color='term_deposit',).update_layout(autosize=False, width=500, height=500), use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.plotly_chart(px.histogram(df_clean, x=bi_select, barmode='overlay', color='term_deposit').update_layout(autosize=False, width=500, height=500), use_container_width=True)
   
##########################################################################################################################################

if selected == 'Data Cleaning':
    st.title(':sparkles: Data Cleaning')
    st.markdown("***")   
    # First header with smaller size
    st.markdown("<h2 style='text-align: center; color: black;'><u>1- Data Management</u></h2>"
                , unsafe_allow_html=True)
    st.write('Unfortunately we don\'t have much cleaning to do since we the data has a lot of outliers and all the columns are needed since they describe the customer behaviour.')
    st.write(df_old.describe())
    st.write('- We can rename **y** column to **term_deposit** for readability.')
    st.write('- We can see that there are negative values in the **balance** column, we can deal with that by changing the sign')
    st.write('- The **previous** column is suspicious in the max value, so we can check for it if we can remove this for being an outlier or deal with it with an imputer')
    st.write(f'- The number of **balance** values below 0 is {df_old[df_old['balance'] < 0].count()[0]}, we can deal with that by changing the sign.')
    st.markdown("<h2 style='text-align: center; color: black;'><u>2- Remove Outliers</u></h2>"
                , unsafe_allow_html=True)
    st.write('- We notice in our data there are some outliers that we need to handle, since as shown below there is a value that seems to be far away, we can remove it')
    col1, col2 = st.columns(2)
    with col1:
        st.write(df_old['previous'].plot(kind='box', figsize=(3,4), patch_artist=True))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    with col2:
        df = df_old[df_old['previous'] < 250]
        st.write(df['previous'].plot(kind='box', figsize=(3,4), patch_artist=True))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    st.markdown("<h2 style='text-align: center; color: black;'><u>3- Final Data</u></h2>"
                , unsafe_allow_html=True)
    st.write('- Our data finally will look like that below:')
    st.write(df_clean.head(5))

##########################################################################################################################################


##########################################################################################################################################

if selected == 'Machine Learning':
    st.info("""Please select data from the below drop down lists and click Run button to run the model.""")
    
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    balance = st.number_input('Balance', min_value=0, max_value=None, value=20000)
    campaign = st.number_input('Campaign', min_value=0, max_value=None, value=5)
    pdays = st.number_input('Pdays', min_value=-1, max_value=None, value=5)
    previous = st.number_input('Previous', min_value=0, max_value=None, value=5)
    duration = st.number_input('Duration', min_value=0, max_value=None, value=5)

    selected_date = st.date_input('Date')
    formatted_date = format_date(selected_date)
    month = selected_date.strftime("%b").lower()
    day = selected_date.strftime("%d")

    col1, col2, col3 = st.columns(3)
    with col1:
        job = df_clean['job'].unique()
        # Create a select box (dropdown list)
        job_option = st.selectbox(
            'Choose a Job:',
            job
        )
    with col2:
        marital = df_clean['marital'].unique()
        # Create a select box (dropdown list)
        marital_option = st.selectbox(
            'Choose a Marital Status:',
            marital
        )
    with col3:
        education = df_clean['education'].unique()
        # Create a select box (dropdown list)
        education_option = st.selectbox(
            'Choose an Education:',
            education
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        default = df_clean['default'].unique()
        # Create a select box (dropdown list)
        default_option = st.selectbox(
            'Choose a Default:',
            default
        )
    with col2:
        housing = df_clean['housing'].unique()
        # Create a select box (dropdown list)
        housing_option = st.selectbox(
            'Choose a Housing:',
            housing
        )
    with col3:
        loan = df_clean['loan'].unique()
        # Create a select box (dropdown list)
        loan_option = st.selectbox(
            'Choose a Loan:',
            loan
        )

    col1, col2 = st.columns(2)
    
    with col1:
        contact = df_clean['contact'].unique()
        # Create a select box (dropdown list)
        contact_option = st.selectbox(
            'Choose a Contact:',
            contact
        )
    with col2: 
        poutcome = df_clean['poutcome'].unique()
        # Create a select box (dropdown list)
        poutcome_option = st.selectbox(
            'Choose a Poutcome:',
            poutcome
        )
    
    if st.button('Run'):
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        data = pd.DataFrame([[age, 
                            job_option,
                            marital_option, 
                            education_option, 
                            default_option, 
                            balance, 
                            housing_option, 
                            loan_option, 
                            contact_option, 
                            day, 
                            month, 
                            duration, 
                            campaign, 
                            pdays, 
                            previous, 
                            poutcome_option]], 
                            columns=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'],
                            index=[0])
        result = model.predict(data)
        if result == 0:
            st.error('Sorry, you will not subscribe to the term deposit')
        else:
            st.success('Congratulation, you will subscribe to the term deposit')

