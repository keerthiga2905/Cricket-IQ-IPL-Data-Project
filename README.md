# Cricket-IQ-IPL-Data-Project
VisualizinVisualizing IPL data and building real-time match predictions with Streamlitg IPL data and building real-time match predictions with Streamlit
The Indian Premier League (IPL) is a popular Twenty20 cricket tournament held annually in India. This project aims to perform exploratory data analysis (EDA) and create data visualizations on an IPL dataset, along with predicting the winner of an IPL match using historical data and machine learning models.

It takes match details as input and predicts the winning probabilities using a trained Random Forest model.

🔹 Predict IPL match winners using a machine learning model
🔹 Simple UI with dropdowns & numeric inputs
🔹 Results shown in a bar chart (win probabilities)

📂 Project Structure
CricketIQ_Infosys_Internship_Feb2025/

│── 📜 README.md                      # Project documentation  
│── 📦 requirements.txt                # List of dependencies  
│  
├── 📁 datasets/                       # Raw datasets  
│   ├── 📄 matches.csv                 # IPL match data  
│   ├── 📄 deliveries.csv              # Ball-by-ball data  
│  
├── 🤖 models/                         # Trained models & encoders  
│   ├── 🏆 final_rf_model.pkl          # Trained Random Forest model  
│   ├── 🏷️ le_team.pkl                 # Label encoder for teams  
│   ├── 🏟️ le_venue.pkl                # Label encoder for venues  
│  
├── 📒 notebooks/                      # Jupyter Notebooks for EDA & training  
│   ├── 📊 eda_data_processing.ipynb   # Data preprocessing & EDA  
│   ├── 🏋️ model_training.ipynb        # Model training  
│  
├── 🖥️ scripts/                        # Python scripts for model & UI  
│   ├── 🎨 streamlit_app.py            # Streamlit UI script  
│   ├── 🏗️ train_model.py              # Script for model training 


2️⃣ Create & Activate Virtual Environment
Mac/Linux:
python -m venv venv
source venv/bin/activate
Windows:
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
📜 requirements.txt

numpy==2.1.3
pandas==2.1.4
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.2
streamlit==1.33.0
joblib==1.3.2
pickle-mixin==1.0.2

📊 Data Processing & EDA
Datasets Used:
matches.csv: Contains match-level details.
deliveries.csv: Contains ball-by-ball details.
Processing Steps:
Missing Values: Cleaned and handled missing data.
Feature Encoding: Encoded categorical features such as teams and venues using Label Encoding.
Additional Features: Calculated features like current run rate (CRR) and required run rate (RRR).
Visualization: Utilized matplotlib.pyplot and seaborn for data visualization.
Check: eda_data_processing.ipynb



🛠 Model Training
Model: Random Forest Classifier
Features Used: Batting Team, Bowling Team, Venue, Total Runs, Wickets, Overs, Target (for 2nd innings), Run Rate Impact.
Data Encoding: Label Encoding used for teams and venues (saved in le_team.pkl and le_venue.pkl).
Model Visualization: Visualized feature importance using matplotlib.pyplot.
The trained model is saved as final_rf_model.pkl.

Check: model_training.ipynb

🎯 Streamlit UI
Run the Streamlit app to predict the match outcome in real-time.

Command to Start the App:
streamlit run scripts/streamlit_app.py
UI Features:
Select Batting Team, Bowling Team, and Venue.
Input Runs, Wickets, Overs (and Target for 2nd innings).
Automatically computes CRR & RRR.
Displays winning probabilities using a bar chart (Matplotlib + Streamlit).
Example Prediction:
Input:
Batting Team: CSK
Bowling Team: MI
Venue: Wankhede Stadium
Total Runs: 160
Wickets Lost: 4
Overs Completed: 15
Target (if 2nd Innings): 180
Prediction Output:
CSK: 65%
MI: 35%
Bar Chart:
import matplotlib.pyplot as plt

teams = ["CSK", "MI"]
probabilities = [65, 35]

plt.bar(teams, probabilities, color=["yellow", "blue"])
plt.xlabel("Teams")
plt.ylabel("Win Probability (%)")
plt.title("IPL Match Prediction Result")
plt.show()
This is implemented in streamlit_app.py, where the winning probabilities are displayed using Matplotlib in Streamlit.

📝 Future Enhancements
Use Deep Learning (LSTMs) for better predictions.
Add Live API Integration for real-time match updates.
Improve UI with dynamic graphs.
