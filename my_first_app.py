# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:39:14 2023

@author: Francisco.Diaz
"""
import seaborn as sns
import streamlit as st
# import plotly.express as px
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


   
if __name__ == "__main__":
    
    with st.sidebar:
        st.image('data:image/png;base64,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', width = 300)
        
        st.markdown("""
        ## Francisco Diaz
        ## 20th November 2023
        ## Soccermatics Project 3
        Analyze the attacking run patters in the Champions League final 2023
        
        """)
       
        st.sidebar.title("Upload Runs Data") 
        
        st.write('Please upload runs file')
        # uploaded_runs_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'parquet'], key="upload_run_file")

    
        # # Select Team
        # teams = sorted(df['Team'].unique().tolist())
        # selected_team = st.sidebar.selectbox("Select Team", teams)    
     
    st.title('Analyze the attacking run patters in the Champions League final 2023')   
     
    city_vs_inter_image = 'https://azfpictures.blob.core.windows.net/test/city_vs_inter.png'
    st.image(city_vs_inter_image,  width = 500)
    
    # if uploaded_runs_file is not None: 
        
    tab1, tab2 = st.tabs(["Runs by players in final third :twisted_rightwards_arrows:","Average run direction :repeat:"])  
 
    with tab1:
        button_player_runs = st.button("Show Players Runs in Final Third", key="show_players_runs_final_third")
        # Plot runs by players in final third
  
     
     
    with tab2:
        button_avg_run_direction = st.button("Show Run Directions", key="show_run_direction")
        
        check_forward_runs = st.checkbox('Forward Runs')
        check_final_third  = st.checkbox('Final Third')    
        check_target       = st.checkbox('Target')    


                
                
                