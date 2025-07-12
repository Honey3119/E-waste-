import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple

class EWasteDatabase:
    def __init__(self, db_path: str = "ewaste_flow.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
    def create_tables(self):
        """Create normalized database schema"""
        cursor = self.conn.cursor()
        
        # Locations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS locations (
                location_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                address TEXT,
                latitude REAL,
                longitude REAL,
                capacity INTEGER,
                operating_hours TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Waste types table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS waste_types (
                waste_type_id INTEGER PRIMARY KEY,
                type_name TEXT UNIQUE NOT NULL,
                category TEXT,
                processing_cost REAL,
                average_weight REAL
            )
        ''')
        
        # Drop-offs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drop_offs (
                drop_id INTEGER PRIMARY KEY,
                location_id INTEGER,
                waste_type_id INTEGER,
                drop_quantity INTEGER,
                timestamp TIMESTAMP,
                event_flag BOOLEAN DEFAULT 0,
                weather_conditions TEXT,
                day_of_week INTEGER,
                is_weekend BOOLEAN,
                is_holiday BOOLEAN,
                season TEXT,
                FOREIGN KEY (location_id) REFERENCES locations (location_id),
                FOREIGN KEY (waste_type_id) REFERENCES waste_types (waste_type_id)
            )
        ''')
        
        self.conn.commit()
        print("Database tables created successfully!")
    
    def populate_reference_data(self):
        """Populate reference tables with initial data"""
        cursor = self.conn.cursor()
        
        # Sample locations
        locations = [
            (1, "Downtown Recycling Center", "123 Main St", 40.7128, -74.0060, 1000, "8AM-6PM"),
            (2, "Northside E-Waste Hub", "456 Oak Ave", 40.7589, -73.9851, 800, "9AM-5PM"),
            (3, "Westside Collection Point", "789 Pine Rd", 40.7282, -74.0776, 600, "8AM-4PM")
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO locations 
            (location_id, name, address, latitude, longitude, capacity, operating_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', locations)
        
        # Waste types
        waste_types = [
            (1, "Smartphones", "Electronics", 2.5, 0.15),
            (2, "Laptops", "Electronics", 5.0, 2.5),
            (3, "Batteries", "Hazardous", 1.0, 0.05),
            (4, "Printers", "Electronics", 4.0, 8.0),
            (5, "Small Appliances", "Appliances", 3.5, 5.0)
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO waste_types 
            (waste_type_id, type_name, category, processing_cost, average_weight)
            VALUES (?, ?, ?, ?, ?)
        ''', waste_types)
        
        self.conn.commit()
        print("Reference data populated successfully!")

# Test the database
if __name__ == "__main__":
    db = EWasteDatabase()
    db.populate_reference_data()
    print("Database setup complete!")