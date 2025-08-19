import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir,"final_project.ods")

data = pd.read_excel(data_path, engine="odf", dtype=str)