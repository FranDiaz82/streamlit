import pyodbc
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
from mplsoccer import Pitch, Sbopen, VerticalPitch, Radar, FontManager, grid, PyPizza
from urllib.request import urlopen
from PIL import Image
import seaborn as sns
import streamlit as st
# import plotly.express as px
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from highlight_text import fig_text
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
import itertools
import requests
from PIL import Image
from io import BytesIO
from PIL import Image
from highlight_text import ax_text
from urllib.request import urlopen
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from PIL import Image
from highlight_text import ax_text
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import io
import tempfile

def func_load_data(uploaded_file):
    
    # Check the file extension and read accordingly
    if uploaded_file.name.endswith('.csv'):
        # Read CSV file
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        # Read Excel file
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.parquet'):
        # Read Parquet file
        df = pd.read_parquet(uploaded_file)
    else:
        st.error("Unsupported file format")    
    
    return df


def func_plot_runs_by_player_in_final_third(df_runs):

    xg_runs = df_runs
    
    xg_runs = df_runs[df_runs['Forward runs'] == True]
    
    # Plot all runs made by each player in the Final Third entry
    xg_runs = xg_runs[xg_runs['start_x'] >= 70]    
    
    # Filter data for each team
    xg_runs_inter = xg_runs[xg_runs['team_name'] == "Inter"]
    xg_runs_city  = xg_runs[xg_runs['team_name'] == "Manchester City"]
    
    # Create a nested dictionary for player colors
    team_player_colors = {}
    
    teams = ["Inter", "Manchester City"]
    for team in teams:
        team_data = xg_runs[xg_runs['team_name'] == team]
        team_players = team_data['player'].unique()
        cmap = plt.cm.get_cmap('tab20', len(team_players))  # You may choose a different colormap
        team_player_colors[team] = {player: cmap(i) for i, player in enumerate(team_players)}
    
    # Initialize figure with two subplots
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(24, 10))

    # Function to draw pitch, runs and add legend
    def func_plot_team_runs(ax, team_data, team_name):
        pitch = Pitch(pitch_type="opta", half=False, corner_arcs=True, goal_type='box', pitch_color="w", linewidth=1, spot_scale=0, line_color="k", line_zorder=1)
        pitch.draw(ax=ax)
    
        # Calculate the final third
        final_third_start = 105 * (2/3)
        ax.axvspan(final_third_start, 105, ymin=0, ymax=1, alpha=0.2, color='grey')
    
        # Plot the arrows for runs
        for index, row in team_data.iterrows():
            color = team_player_colors[team_name][row['player']]
            pitch.arrows(row['start_x'], row['start_y'], row['end_x'], row['end_y'],
                          width=1.8, headwidth=5, headlength=5, headaxislength=5,
                          color=color, alpha=0.8, zorder=3, ax=ax)
    
        # Create legend handles for players
        legend_handles = [plt.Line2D([0], [0], color=color, marker='o', linestyle='', label=f"{player}")
                          for player, color in team_player_colors[team_name].items()]
    
        
        if team_name == 'Inter':
            dist = -0.15
            club_image = 'https://azfpictures.blob.core.windows.net/test/inter.png'
        else:
            dist = -0.2
            club_image = 'https://azfpictures.blob.core.windows.net/test/city.png'          
            
        # Add updated legend to the plot
        ax.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(0, dist),
                  fontsize='small', framealpha=0.0, ncol=3)
    
        ax.set_title(f'Forward Runs by player in the final third for {team_name}')
    
    # Plot runs for Inter
    func_plot_team_runs(axs[0], xg_runs_inter, "Inter")
    
    # Plot runs for Manchester City
    func_plot_team_runs(axs[1], xg_runs_city, "Manchester City")    

    figure_one = plt.figure()
    
    # Assign the figure to the variable
    figure_one = fig    
    
    st.pyplot(figure_one)          


def func_plot_average_run_direction(df_runs):
    
    st.write('Attacking direction:')
    arrow_image = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXsAAACFCAMAAACND6jkAAAAclBMVEX///8AAAD5+fkQEBDq6uro6Og3Nzft7e3m5ubg4ODd3d1MTExJSUlERERZWVni4uI+Pj5lZWVqampTU1NgYGBRUVGLi4swMDBAQEBwcHBycnJra2s6OjojIyMICAgeHh6lpaUYGBh8fHyGhoaSkpK3t7e+cuwmAAADDElEQVR4nO3dDW4aMRiE4Zgs3R9gtwFCG2iT0qb3v2JFUwkofBZqGY9E3ucEq5E1GK/XvrvDvxh9q9yP8G49pYfa/Qzv1DqlNCd8g9Em7TwSfnlv0Sdqp7x1SoRvMXpKe9ROUfN0iB/cco5G/c6SeX4hJ9GnNGHkl/FwEn1KG8IvYLQ5E31K/dj9YLfvTOG8WTDy1dZB9Mx29NpPYfgLakes+xCG3zPyxZowe2pHrolH/oTaEevizl8y8sUytcOqptrwJa4d1nbEcrVD+GLNM+HbtB+pHZvua/yDS/hizUsYft+4H+7WdZ/j2mGqKdZu49ohfLEmnmpSO2pdZrbDyBcbwuypHblxZlWT2hHL1A471tRY1TSq4lXNKbUj1sXh9ywviFVx7bCqqdbEywvUjlpu6wgjX6y6j8Nn5IsNM2rHJrNdkNpRq+I3Wf3gfrhbl6sddqyJDfHIZ56vVsWd/0jtiLVx7cwY+WJNPM+ndtTq+B/utHU/3K3L1M6UkS82Zq+mTx2/RpxRO2JtZscaI1+sjmuHVU21KjPP50+W2BB3/tFH0CNc3d04zP5oVXP6fI9re4m35x/WTvxPDBr72iH74hYt2dvMGrK3+XPSFNk7TDqyt/ldO2TvsasdsjeZdWRv853sXRj3NvS9DfMcG+b3NvyvtWE9x4Z1TJv9+j3vrQS2l723cr/bvEWXvq/F9V26TwFXx/4cm5qvbV3Yj2lTx9GzD1mL/fc2fHdiU2dOkaJwpJothWPC97U2fFduw3kKNpwjYsP5OTacG2XDeWk2nBNow/mYNpwLa8N5yDacA27D+fc23Ptgw30nNtzzY8P9VjZcJ2nDfYY23ONpQ+HYcG+zDfeV27SZm2uJXqphz6VLF08uiV5riFcuKRwtVi5tMu9ml0Qv1WRWLtl9JpUpnDmjXiq3GYTopTIb//hkWWxF4fhE4S+IXu98+D2FU8KPM9FvGPVlnI58XpUU8/pX9CyfFXRcO8xwijqsHXafFbavHVYui1sRvc+KwvF55WfWZ0Xh+Pz8/+h/ARHqSHEvtC5KAAAAAElFTkSuQmCC'
    st.image(arrow_image,  width = 100)           
        
    # Use 'tab10' colormap
    unique_players = df_runs['player'].unique()
    num_players = len(unique_players)
    tab10 = cm.get_cmap('tab20', num_players)  # Get 10 colors from tab10
    colors = itertools.cycle(tab10.colors)  # Cycle through colors if there are more than 10 players
    player_colors = dict(zip(unique_players, colors))
    
    pitch = Pitch(pitch_type="opta", half=False, corner_arcs=True, goal_type='box', pitch_color="w", linewidth=2, spot_scale=0, line_color="k", line_zorder=1)
    fig, ax = pitch.draw(figsize=(24, 10))
    
    
    # Calculate the final third
    final_third_start = 105 * (2/3)
    ax.axvspan(final_third_start, 105, ymin=0, ymax=1, alpha=0.2, color='grey')    
    
    legend_patches = []
    
    df_runs['Distance'] = df_runs['Distance'].astype(int)
    
    for index, row in df_runs.iterrows():
        color = player_colors[row['player']]
        
        factor=1
        if row['Forward runs'] == True and check_forward_runs == False:
            factor=2
        
        pitch.arrows(row['start_x'], row['start_y'], row['end_x'], row['end_y'],
                     width=1.8*factor, headwidth=5*factor/2, headlength=5, headaxislength=5,
                     color=color, alpha=0.8, zorder=3, ax=ax)
        
        # Calculate midpoint of the arrow for text annotation
        mid_x = (row['start_x'] + row['end_x']) / 2
        mid_y = (row['start_y'] + row['end_y']) / 2
        
        # Display distance at the midpoint of the arrow
        ax.text(mid_x, mid_y, str(row['Distance']), color='black', fontsize=12, ha='center', va='center')            
            
        # Create legend patches
        if row['player'] not in [patch.get_label() for patch in legend_patches]:
            patch = mpatches.Patch(color=color,  label=row['player'])
            legend_patches.append(patch)
    
    # Add legend to the plot
    plt.legend(handles=legend_patches, loc='lower left', bbox_to_anchor=(0, -0.6), 
               fontsize=15, framealpha=0.0, ncol=3)
    
    ax.set_title('Run directions per player')
    
    # Creating a new figure for Streamlit to display
    figure_two = plt.figure()
    figure_two = fig    
    
    st.pyplot(figure_two)
    
    
def func_get_player_stats(df_runs):
    
    stats_runs = df_runs
    
    stats_runs['Back Runs']  = (stats_runs['end_x'] <= stats_runs['start_x']).astype(int)
       
    
    df_runs_selected = stats_runs[["team_name","player","period","Distance",'Runs into box','max_speed','Back Runs','Forward runs','Target']]
    
    df_runs_grouped = df_runs_selected.groupby(['team_name','player']).agg({
                                                    'Distance': 'sum',
                                                    'period':   'count',
                                                    'Runs into box': lambda x: x.sum() ,
                                                    'max_speed':'max',
                                                    'Back Runs': 'sum',
                                                    'Forward runs': lambda x: x.sum() ,
                                                    'Target': lambda x: x.sum() ,
                                                }).reset_index()      
    
    df_runs_grouped = df_runs_grouped.rename(columns={'team_name': 'Team'})
    df_runs_grouped = df_runs_grouped.rename(columns={'player': 'Player'})
    df_runs_grouped = df_runs_grouped.rename(columns={'Distance': 'Dist. Run'})
    df_runs_grouped = df_runs_grouped.rename(columns={'period': 'No Runs'})
    df_runs_grouped = df_runs_grouped.rename(columns={'max_speed': 'Max Speed'})
    df_runs_grouped = df_runs_grouped.rename(columns={'Runs into box': 'Runs inbox'})
        
    
    # st.table(df_runs_grouped)
    st.dataframe(df_runs_grouped)


def func_connect_to_sql():
    # # Connection parameters
    server = 'LUATSAP'
    database = 'Akamai_Videos'
    username = 'sapuat.user'
    password = '$@Pu@+Kdi31Df2'	

    # Create a connection
    cnxn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
    
    cursor = cnxn.cursor()
    cursor.execute("TRUNCATE TABLE [Akamai_Videos].[dbo].[images]")   


def func_generate_frames(df_tracking,player_to_visualize):
    
    
    # func_connect_to_sql()
    ########################################################################################################
    # Connection parameters
    server = 'LUATSAP'
    database = 'Akamai_Videos'
    username = 'sapuat.user'
    password = '$@Pu@+Kdi31Df2'	

    # Create a connection
    cnxn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
    
    cursor = cnxn.cursor()
    cursor.execute("TRUNCATE TABLE [Akamai_Videos].[dbo].[images]")   
    ########################################################################################################
    
    pitch_length = 105
    pitch_width = 68

    def transform_x_coordinates(x):
        return x / pitch_length * 100


    def transform_y_coordinates(x):
        return 100 - (x / pitch_width * 100)   


    df_tracking = df_tracking[df_tracking['vx'] != 0]
    
    # Set team colors
    team_colors = {
        'Manchester City': '#6cabdd',
        'Inter': '#010E80',
    }
    
    # Ensure output directory exists
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over each unique frame in the dataset
    for frame, period in df_tracking[['frame', 'period']].drop_duplicates().itertuples(index=False):
        selected_frame = df_tracking[(df_tracking['frame'] == frame) & (df_tracking['period'] == period)]
    
        # Transform coordinates to wyscout
        selected_frame['start_x'] = transform_x_coordinates(selected_frame['x'])
        selected_frame['start_y'] = transform_y_coordinates(selected_frame['y'])
    
        fig, ax = plt.subplots(figsize=(10, 7))
        pitch = Pitch(pitch_type="wyscout", goal_type='box', pitch_color="w", linewidth=1, spot_scale=0, line_color="k", line_zorder=1)
        pitch.draw(ax=ax)
        
        for team, players in selected_frame.groupby('team_name'):
            # Determine the color for the team or the ball
            if team == 'ball':
                team_color = 'k'
            else:
                team_color = team_colors.get(team, 'k')
            
            # Iterate over players
            for idx, row in players.iterrows():
                # Check if the player is Denzel Dumfries
                if row['player'] == player_to_visualize:
                    color = 'red'
                else:
                    color = team_color
                
                # Plotting the player/ball
                pitch.scatter(row['start_x'], row['start_y'], color=color, ec='k', ax=ax, lw=1, 
                              zorder=3 if team == 'ball' else 2, s=50 if team == 'ball' else 200)
                
                pitch.arrows(row['start_x'], row['start_y'], row['start_x'] + row['vx'], row['start_y'] - row['vy'],
                              width=1.8, headwidth=5, headlength=5, headaxislength=5,
                              color=color, alpha=0.8, zorder=3, ax=ax)
                
                # Annotating jersey number for players
                if team != 'ball':
                    pitch.annotate(row['jersey_number'], xy=(row['start_x'], row['start_y']), 
                                    c='w', va='center', ha='center', size=8, weight='bold', alpha=1, ax=ax)    
            
    
        # image  = fig #fig.open(f"{output_dir}/freeze_frame_{period}_{frame}.jpg", format='jpg', dpi=200, bbox_inches='tight')
        # Add title
        ax.set_title(f"{player_to_visualize} - Frame {frame} - Period {period}")
        
        
        
        # Convert the figure to a BytesIO object
        buf = BytesIO()
        fig.savefig(buf, format='JPEG')
        buf.seek(0)
        img_binary = buf.getvalue()
        
        # Close the buffer
        buf.close()
            
        
        name = f"freeze_frame_{period}_{frame}.jpg"       
        
        
        # # Save image
        # fig.savefig(f"{output_dir}/freeze_frame_{period}_{frame}.jpg", format='jpg', dpi=200, bbox_inches='tight')
        # plt.close(fig)  # Close the figure to free memory
        
        

        cursor.execute("INSERT INTO [Akamai_Videos].[dbo].[images] (image, description, name) VALUES (?, ?, ?)",(img_binary, player_to_visualize, name))        
        cnxn.commit()
        
    cursor.close()
    
    
    #########################################################################################################
    button_reproduce_video = st.button("Generate Frames", key="reproduce_video_jey")
    # Plot runs by players in final third
    if button_reproduce_video:
        # Create a connection
        cnxn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
        cursor = cnxn.cursor()
        
        # Retrieve all images
        cursor.execute("SELECT image FROM images order by id")
        rows = cursor.fetchall()
        
        images = []
        for row in rows:
            # Convert binary data to an image
            img_stream = io.BytesIO(row[0])
            image = Image.open(img_stream)
            images.append(image)
        
        # Close the cursor and connection
        cursor.close()
        cnxn.close()
        
        # Create a temporary file to save video
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        video_path = temp_file.name
        
        height = 700
        width = 1000
        frame_rate = 1
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust the codec as needed
        video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height), True)  # Adjust frame rate and size as needed
        
        for img in images:
            # Convert PIL Image to a format compatible with OpenCV
            open_cv_image = np.array(img)
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
            # Write the frame to the video
            video.write(open_cv_image)
        
        # Release the VideoWriter object
        video.release()
        
        # Read the video from the temporary file and stream it
        with open(video_path, 'rb') as file:
            video_bytes = file.read()
            video_stream = io.BytesIO(file.read())
        
        # Display the video in Streamlit
        # st.video(video_stream)
        
        # Remove the temporary file (optional)
        temp_file.close()
        
        # Create a download button for the video
        st.download_button(
            label="Download Video",
            data=video_bytes,
            file_name="output_video.mp4",
            mime="video/mp4"
        )
                    
    
    
    
    #########################################################################################################



def func_show_forward_runs():
    
    st.write('Please select tracks file:')
    uploaded_track_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'parquet'], key="upload_track_file")
    
    # Check if a file has been uploaded
    if uploaded_track_file is None:
        st.write("Please upload a file to proceed.")    
    else:
        
        df_runs = func_load_data(uploaded_runs_file)        
        
        df_tracks = func_load_data(uploaded_track_file)
               
        # st.dataframe(df_tracks)
        
        # Select Player
        players = sorted(df_runs['player'].unique().tolist())
        # players_with_select_all = ["Select All"] + players
        selected_player = st.selectbox("Select Player", players, key="visualize_players")


        if "Select All" in selected_player:
            selected_player = players
        
        df_runs = df_runs[df_runs['player'] == selected_player]
        # Filter by Forward Runs only
        df_runs = df_runs[df_runs['Forward runs'] == True]
        
        df_runs_plot_pitch  = df_runs

        df_runs = df_runs.rename(columns={'team_name': 'Team'})
        # df_runs = df_runs.rename(columns={'player': 'Player'})
        df_runs = df_runs.rename(columns={'jersey': 'Jersey'})
        df_runs = df_runs.rename(columns={'period': 'Period'})
        df_runs = df_runs.rename(columns={'frame_start': 'Frame Start'})
        df_runs = df_runs.rename(columns={'frame_end': 'Frame End'})
        
        tracks_columns = ["Team", "player",'Jersey','Period',"Distance" ,"Frame Start",'Frame End']
        
        df_runs = df_runs[tracks_columns]
        
        df_runs = df_runs.sort_values(by=['Team', 'player','Period','Frame Start'])
              
        st.write('Please select the players to analyze from the Filter Panel')
        
        def dataframe_with_selections(df_runs):
            df_with_selections = df_runs.copy()
            df_with_selections.insert(0, "Select", False)
        
            # Get dataframe row-selections from user with st.data_editor
            edited_df = st.data_editor(
                df_with_selections,
                hide_index=True,
                column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                disabled=df_runs.columns,
            )
        
            # Filter the dataframe using the temporary column, then drop the column
            selected_rows = edited_df[edited_df.Select]
            return selected_rows.drop('Select', axis=1)
        
        
        selection = dataframe_with_selections(df_runs)
        
        player_to_visualize = selection['player'].min()
        period_to_visualize = selection['Period'].min()
        min_dataframe = selection['Frame Start'].min()
        max_dataframe = selection['Frame End'].max()
        
        # st.write(selection)
        
        # st.write(player_to_visualize)
        # st.write(min_dataframe)
        # st.write(max_dataframe)
        
        #Filter tracks dataframe
        
        # df_tracks = df_tracks[df_tracks['player'] == player_to_visualize]
        df_tracks = df_tracks[df_tracks['period'] == period_to_visualize]
        df_tracks = df_tracks[df_tracks['frame'] >= min_dataframe]
        df_tracks = df_tracks[df_tracks['frame'] <= max_dataframe]
        
        # st.dataframe(df_tracks)
        
        func_plot_average_run_direction(df_runs_plot_pitch)
        
        func_generate_frames(df_tracks,player_to_visualize)
   
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
        uploaded_runs_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'parquet'], key="upload_run_file")
    
        # Check if a file has been uploaded
        if uploaded_runs_file is not None:
            df_runs = func_load_data(uploaded_runs_file)
            # st.dataframe(df_runs)  
            st.write('Please select the players to analyze from the Filter Panel')
     
        else:
            st.write("Please upload a file to proceed.")    
    
        # # Select Team
        # teams = sorted(df['Team'].unique().tolist())
        # selected_team = st.sidebar.selectbox("Select Team", teams)    
     
    st.title('Analyze the attacking run patters in the Champions League final 2023')   
     
    city_vs_inter_image = 'https://azfpictures.blob.core.windows.net/test/city_vs_inter.png'
    st.image(city_vs_inter_image,  width = 500)
    
    if uploaded_runs_file is not None: 
        
        tab1, tab2, tab3 = st.tabs(["Runs by players in final third :twisted_rightwards_arrows:","Average run direction :repeat:","Display Forward Runs:film_projector:"])  
 
        with tab1:
            button_player_runs = st.button("Show Players Runs in Final Third", key="show_players_runs_final_third")
            # Plot runs by players in final third
            if button_player_runs:
                func_plot_runs_by_player_in_final_third(df_runs)         
         
         
        with tab2:
            button_avg_run_direction = st.button("Show Run Directions", key="show_run_direction")
            
            check_forward_runs = st.checkbox('Forward Runs')
            check_final_third  = st.checkbox('Final Third')    
            check_target       = st.checkbox('Target')    
            
            if check_forward_runs:
                df_runs = df_runs[df_runs['Forward runs'] == True]
                
            if check_final_third:
                df_runs = df_runs[df_runs['start_x'] >= 70]       
                
            if check_target:
                df_runs = df_runs[df_runs['Target'] == True]                   
            
            # Select Player
            players = sorted(df_runs['player'].unique().tolist())
            players_with_select_all = ["Select All"] + players
            selected_player = st.multiselect("Select Player(s)", players_with_select_all) 

            if button_avg_run_direction:
                
                if "Select All" in selected_player:
                    selected_player = players
                
                df_runs = df_runs[df_runs['player'].isin(selected_player)]
                
                func_get_player_stats(df_runs)
                func_plot_average_run_direction(df_runs)
                
        with tab3:
            # button_show_runs = st.button("Show Runs", key="show_runs")
            func_show_forward_runs()
            # st.write('hola')
                
                
                
                