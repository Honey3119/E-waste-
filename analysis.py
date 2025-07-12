import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

class EWasteAnalyzer:
    def __init__(self, db_path: str = "ewaste_flow.db"):
        self.conn = sqlite3.connect(db_path)
    
    def analyze_data(self):
        print("📊 E-WASTE FLOW+ ANALYSIS REPORT")
        print("=" * 50)
        
        # Load data
        df = pd.read_sql_query('''
            SELECT d.*, l.name as location_name, w.type_name as waste_type_name
            FROM drop_offs d
            JOIN locations l ON d.location_id = l.location_id
            JOIN waste_types w ON d.waste_type_id = w.waste_type_id
        ''', self.conn)
        
        # Basic statistics
        print(f"📈 Dataset Overview:")
        print(f"   Total records: {len(df):,}")
        print(f"   Date range: {df['timestamp'].min()[:10]} to {df['timestamp'].max()[:10]}")
        print(f"   Total e-waste collected: {df['drop_quantity'].sum():,} units")
        print(f"   Average daily collection: {df['drop_quantity'].mean():.1f} units")
        
        # Seasonal analysis
        print(f"\n🌱 Seasonal Patterns:")
        seasonal = df.groupby('season')['drop_quantity'].agg(['sum', 'mean']).round(1)
        for season, stats in seasonal.iterrows():
            print(f"   {season}: {stats['sum']:,} total, {stats['mean']:.1f} avg per drop")
        
        # Day of week analysis
        print(f"\n📅 Day of Week Patterns:")
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = df.groupby('day_of_week')['drop_quantity'].mean().round(1)
        for day_num, avg in daily.items():
            print(f"   {days[day_num]}: {avg:.1f} units average")
        
        # Location analysis
        print(f"\n📍 Location Performance:")
        location_stats = df.groupby('location_name')['drop_quantity'].agg(['sum', 'count']).round(1)
        for location, stats in location_stats.iterrows():
            print(f"   {location}: {stats['sum']:,} total units, {stats['count']:,} drop-offs")
        
        # Waste type analysis
        print(f"\n🗑️ Top Waste Types:")
        waste_stats = df.groupby('waste_type_name')['drop_quantity'].sum().sort_values(ascending=False)
        for waste_type, total in waste_stats.items():
            print(f"   {waste_type}: {total:,} units")
        
        # Weekend vs weekday
        weekend_avg = df[df['is_weekend'] == True]['drop_quantity'].mean()
        weekday_avg = df[df['is_weekend'] == False]['drop_quantity'].mean()
        print(f"\n⏰ Weekend vs Weekday:")
        print(f"   Weekday average: {weekday_avg:.1f} units")
        print(f"   Weekend average: {weekend_avg:.1f} units")
        print(f"   Weekend impact: {((weekend_avg/weekday_avg-1)*100):+.1f}%")
        
        return df
    
    def create_visualizations(self, df):
        print(f"\n📊 Creating visualizations...")
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Seasonal distribution
        seasonal_data = df.groupby('season')['drop_quantity'].sum()
        axes[0,0].pie(seasonal_data.values, labels=seasonal_data.index, autopct='%1.1f%%')
        axes[0,0].set_title('E-Waste by Season')
        
        # 2. Daily patterns
        daily_data = df.groupby('day_of_week')['drop_quantity'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0,1].bar(days, daily_data.values)
        axes[0,1].set_title('Average Drop-offs by Day')
        axes[0,1].set_ylabel('Average Quantity')
        
        # 3. Location comparison
        location_data = df.groupby('location_name')['drop_quantity'].sum()
        axes[1,0].bar(range(len(location_data)), location_data.values)
        axes[1,0].set_title('Total Drop-offs by Location')
        axes[1,0].set_xticks(range(len(location_data)))
        axes[1,0].set_xticklabels([name[:10] + '...' for name in location_data.index], rotation=45)
        
        # 4. Waste type distribution
        waste_data = df.groupby('waste_type_name')['drop_quantity'].sum().sort_values(ascending=True)
        axes[1,1].barh(range(len(waste_data)), waste_data.values)
        axes[1,1].set_title('Total by Waste Type')
        axes[1,1].set_yticks(range(len(waste_data)))
        axes[1,1].set_yticklabels(waste_data.index)
        
        plt.tight_layout()
        plt.savefig('ewaste_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Analysis chart saved as 'ewaste_analysis.png'")

# Run analysis
if __name__ == "__main__":
    analyzer = EWasteAnalyzer()
    df = analyzer.analyze_data()
    analyzer.create_visualizations(df)
    print("\n🎉 Analysis complete!")