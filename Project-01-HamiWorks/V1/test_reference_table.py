#!/usr/bin/env python3
"""
Quick test script to verify the reference table functionality works
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create mock data for testing if the real data isn't available
def create_mock_data():
    """Create mock data structure for testing"""
    
    # Mock people_index data
    people_data = {
        'id': ['105001', '105002', '105003', '105004', '105005'],
        'name': ['Ali Boloor', 'Hamid Sadeghi', 'Marjan Ganjizade', 'Fariba Hemayati', 'Fatemah Latif'],
        'reference_id': ['file_1', 'file_2', 'file_3', 'file_4', 'file_5']
    }
    
    return pd.DataFrame(people_data)

def test_reference_table():
    """Test the reference table functionality with mock data"""
    
    try:
        from DataAnalyzer import DataAnalyzer
        analyzer = DataAnalyzer()
        
        # If real data exists, use it
        if analyzer.data_loader.people_index is not None:
            print("✓ Using real data from DataAnalyzer")
        else:
            print("! No real data found, would need to mock it for testing")
            return
            
    except Exception as e:
        print(f"! Could not import DataAnalyzer or load data: {e}")
        print("Creating standalone test with mock data...")
        
        # Create standalone test
        people_index = create_mock_data()
        
        # Test the reference summary creation logic
        ref_data = people_index.copy()
        ref_data['File#'] = ref_data['reference_id'].str.extract(r'file_(\d+)').astype(int)
        summary = ref_data[['File#', 'id', 'name']].sort_values('File#')
        summary.columns = ['File#', 'ID', 'Name']
        summary = summary.reset_index(drop=True)
        
        print("✓ Mock reference summary created:")
        print(summary)
        
        # Test basic table creation logic
        table_data = []
        for _, row in summary.iterrows():
            name_display = row['Name'][:15] + "..." if len(row['Name']) > 15 else row['Name']
            table_data.append([f"{row['File#']}", "→", f"{row['ID']}", name_display])
        
        print("\n✓ Table data structure:")
        for row in table_data:
            print(f"  {row}")
        
        # Create a simple test plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simple bar plot
        x = summary['File#']
        y = [10, 15, 8, 12, 20]  # Mock values
        bars = ax.bar(x, y, color='lightblue', edgecolor='navy', alpha=0.7)
        
        ax.set_title('Test Plot with Reference Information', fontsize=12)
        ax.set_xlabel('File Number', fontsize=10)
        ax.set_ylabel('Mock Values', fontsize=10)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Create a mock table manually
        table = ax.table(cellText=table_data,
                        colLabels=['File#', '', 'ID', 'Name'],
                        cellLoc='left',
                        loc='center',
                        bbox=[0.73, 0.7, 0.25, 0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(table_data) + 1):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        # Add title above the table
        ax.text(0.85, 0.95, 'Hami Reference', 
               transform=ax.transAxes,
               fontsize=9, 
               weight='bold',
               ha='center', 
               va='top')
        
        plt.tight_layout()
        plt.show()
        
        print("\n✓ Test completed successfully!")
        return True
    
    # Test with real DataAnalyzer if available
    try:
        # Test reference summary
        ref_summary = analyzer.create_reference_summary(save_csv=False)
        print(f"✓ Reference summary created with {len(ref_summary)} entries")
        print("First 5 entries:")
        print(ref_summary.head())
        
        # Test adding reference table to a simple plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simple test plot
        x = range(5)
        y = [10, 15, 8, 12, 20]
        ax.bar(x, y, color='lightblue', edgecolor='navy', alpha=0.7)
        ax.set_title('Test Plot with Reference Table', fontsize=12)
        ax.set_xlabel('Test Categories', fontsize=10)
        ax.set_ylabel('Test Values', fontsize=10)
        
        # Add reference table
        analyzer.add_reference_table(ax, position='upper right', max_rows=8)
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Reference table functionality tested successfully!")
        return True
        
    except Exception as e:
        print(f"! Error testing with real data: {e}")
        return False

if __name__ == "__main__":
    print("Testing Reference Table Functionality")
    print("=" * 40)
    test_reference_table()
