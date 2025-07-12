import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class EWasteDataSimulator:
    def __init__(self, db_path: str = "ewaste_flow.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def generate_daily_data(self, start_date: str, end_date: str):
        """Generate realistic e-waste drop-off data"""
        print("Generating e-waste drop-off data...")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        data = []
        current_date = start_dt
        drop_id = 1
        
        while current_date <= end_dt:
            # Day patterns
            is_weekend = current_date.weekday() >= 5
            season = self.get_season(current_date)
            
            # Generate data for each location and waste type
            for location_id in range(1, 4):  # 3 locations
                for waste_type_id in range(1, 6):  # 5 waste types
                    
                    # Base quantity patterns
                    base_quantity = self.get_base_quantity(waste_type_id)
                    
                    # Apply multipliers
                    quantity = base_quantity
                    if is_weekend:
                        quantity *= 0.7  # Less on weekends
                    if season == "Fall":
                        quantity *= 1.3  # More in fall
                    
                    # Add randomness
                    quantity = int(quantity * random.uniform(0.5, 1.5))
                    quantity = max(1, quantity)
                    
                    if quantity > 0:
                        data.append({
                            'drop_id': drop_id,
                            'location_id': location_id,
                            'waste_type_id': waste_type_id,
                            'drop_quantity': quantity,
                            'timestamp': current_date,
                            'event_flag': random.choice([0, 0, 0, 1]),
                            'weather_conditions': random.choice(['Clear', 'Rainy', 'Cloudy']),
                            'day_of_week': current_date.weekday(),
                            'is_weekend': is_weekend,
                            'is_holiday': False,
                            'season': season
                        })
                        drop_id += 1
            
            current_date += timedelta(days=1)
        
        return data
    
    def get_season(self, date):
        month = date.month
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Winter'
    
    def get_base_quantity(self, waste_type_id):
        base_quantities = {1: 25, 2: 15, 3: 45, 4: 12, 5: 20}
        return base_quantities.get(waste_type_id, 20)
    
    def save_to_database(self, data):
        cursor = self.conn.cursor()
        
        for record in data:
            cursor.execute('''
                INSERT INTO drop_offs 
                (drop_id, location_id, waste_type_id, drop_quantity, timestamp, 
                 event_flag, weather_conditions, day_of_week, is_weekend, is_holiday, season)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (record['drop_id'], record['location_id'], record['waste_type_id'],
                  record['drop_quantity'], record['timestamp'], record['event_flag'],
                  record['weather_conditions'], record['day_of_week'], 
                  record['is_weekend'], record['is_holiday'], record['season']))
        
        self.conn.commit()
        print(f"Saved {len(data)} records to database!")
    
    def close(self):
        self.conn.close()

# Generate data
if __name__ == "__main__":
    simulator = EWasteDataSimulator()
    data = simulator.generate_daily_data('2023-01-01', '2023-06-30')
    simulator.save_to_database(data)
    simulator.close()
    print("Data simulation complete!")