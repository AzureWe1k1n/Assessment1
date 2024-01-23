import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class Moons:
    def __init__(self):
        connectable = "sqlite:///jupiter.db"
        query = "SELECT * FROM moons"
        self.data = pd.read_sql(query, connectable)
    
    def summary_statistics(self):
        return self.data.describe()
    
    def correlation_analysis(self):
        return self.data.corr()
    
    def plot_moon_distances(self):
        self.data.plot(kind='bar', x='moon', y='distance_km')
        plt.xlabel('Distance (km)')
        plt.ylabel('Moon')
        plt.title('Distances of Moons')
        plt.show()
        
    def get_moon_data(self, moon_name):
        return self.data[self.data['moon'] == moon_name]
    
    def prepare_data_for_regression(self):
    # Assuming 'period_days' and 'distance_km' are columns in the dataset
        self.data['T_squared'] = np.square(self.data['period_days'] * 24 * 3600)  # Convert to seconds
        self.data['a_cubed'] = np.power(self.data['distance_km'] * 1000, 3)  # Convert to meters
        
    def setup_and_train_model(self):
        X = self.data[['a_cubed']]
        y = self.data['T_squared']
        self.model = LinearRegression()
        self.model.fit(X, y)
        
    def predict_jupiter_mass(self, G=6.67e-11):
    # The slope of the regression line corresponds to 4π²/GM
        GM = 4 * np.pi**2 / self.model.coef_[0]
        return GM / G
    