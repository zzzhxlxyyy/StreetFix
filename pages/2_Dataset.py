import streamlit as st 
import plotly.express as px
import pandas as pd
# import matplotlib as plt
# from streamlit_card import card

st.set_page_config(
    page_title ="StreetFix"
)

st.title("The Dataset")

st.markdown("""
    Dataset for this project is obtained from the **Crowdsensing-based Road Damage Detection Challenge (CRDDC’2022)**. 
    There are 4 damage categories in this dataset, which are:
""")

cols = st.columns([0.05, 0.425, 0.05, 0.425, 0.05])

with cols[1]:
    # card(text="",title="", image="items\D00.png")
    st.markdown("*Longitudinal Crack (D00)*")
    st.image('assets/D00.png', 
             use_column_width=True)
    
    st.markdown("*Alligator Crack (D20)*")
    st.image('assets/D20.png',
            use_column_width=True)

with cols[3]:
    st.markdown("*Transverse Crack (D10)*")
    st.image('assets/D10.png', 
            use_column_width=True)
    
    st.markdown("*Pothole (D40)*")
    st.image('assets/D40.png', 
            use_column_width=True)    

st.divider()
st.write("""
    ### Exploratory Data Analysis (EDA)
    #### Class Distribution 
""")

df = pd.read_csv('assets/class_distribution.csv')

# Plot class distribution using Plotly Express
fig = px.bar(df, x='Class Labels', y=['Train Counts', 'Validation Counts', 'Test Counts'],
                 labels={'value': 'Counts', 'variable': 'Dataset'},
                 title='Class Distribution',
                 color_discrete_sequence=['blue', 'green', 'red'])

st.plotly_chart(fig,theme=None)

st.write("""
    Based on the graph, the highest number of class is Transverse Crack (D10) while the lowest number of class in the dataset is Pothole (D40).
    
    #### Bounding Box Area Disribution
""")
col1, col2, col3 = st.columns(3)

def plot_bbox_area_distribution(csv_filename, set_name):
    # Read data from CSV file
    df = pd.read_csv(csv_filename)

    # Plot bounding box area distribution using Plotly Express
    fig = px.bar(df, x='Area', y='Frequency', title=f'Bounding Box Area Distribution of {set_name}',
             labels={'Area': 'Area', 'Frequency': 'Frequency'},
             color_discrete_sequence=['blue'])

    st.plotly_chart(fig)

train_area = 'assets/bbox_area_distribution_trainset.csv'
test_area = 'assets/bbox_area_distribution_testset.csv'
val_area = 'assets/bbox_area_distribution_valset.csv'

plot_bbox_area_distribution(train_area, "Train Set")
plot_bbox_area_distribution(test_area, "Test Set")
plot_bbox_area_distribution(val_area, "Validation Set")

st.write("""
   
Upon analysis of the graphical representations, it becomes evident that the majority of bounding box areas are comparatively small. 
Consequently, the emphasis of the project will be directed towards the detection of smaller objects.
    
""")


