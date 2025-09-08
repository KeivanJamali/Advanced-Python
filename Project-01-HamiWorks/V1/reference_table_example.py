#!/usr/bin/env python3
"""
Example script demonstrating how to use the reference table functionality
in DataAnalyzer class.
"""

import matplotlib.pyplot as plt
import numpy as np
from DataAnalyzer import DataAnalyzer

# Initialize the analyzer
analyzer = DataAnalyzer()

# Example 1: Create a standalone reference summary
print("=== Reference Summary ===")
ref_summary = analyzer.create_reference_summary()
print(ref_summary.head(10))
print(f"\nTotal number of Hamis: {len(ref_summary)}")

# Example 2: Use the built-in plotting functions with reference tables
print("\n=== Built-in Plots with Reference Tables ===")

# Plot total requests per employee with reference table
requests_per_employee = analyzer.total_requests_per_employee(plot=True, show_reference=True)

# Plot total messages per request with reference table
messages_per_request = analyzer.total_messages_per_request(plot=True, top_n=20, show_reference=True)

# Example 3: Custom plot with reference table
print("\n=== Custom Plot with Reference Table ===")

# Create some sample data for demonstration
fig, ax = plt.subplots(figsize=(12, 8))

# Sample data (you can replace this with your actual data)
file_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
values = [45, 38, 52, 29, 61, 33, 47, 55, 41, 36]

bars = ax.bar(file_numbers, values, color='lightcoral', edgecolor='darkred', alpha=0.7)
ax.set_title('Custom Analysis with Reference Table', fontsize=14)
ax.set_xlabel('File Number', fontsize=12)
ax.set_ylabel('Some Metric', fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add the reference table to the plot
analyzer.add_reference_table(ax, position='upper right', max_rows=8, 
                           fontsize=8, title="Hami Reference")

plt.tight_layout()
plt.show()

# Example 4: Different positions for the reference table
print("\n=== Reference Table in Different Positions ===")

positions = ['upper right', 'upper left', 'lower right', 'lower left']

for i, position in enumerate(positions):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simple plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    ax.plot(x, y, 'b-', alpha=0.7)
    ax.set_title(f'Reference Table Position: {position}', fontsize=12)
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.grid(True, alpha=0.3)
    
    # Add reference table
    analyzer.add_reference_table(ax, position=position, max_rows=6, 
                               fontsize=7, title="Reference")
    
    plt.tight_layout()
    plt.show()

print("\n=== Usage Instructions ===")
print("""
To use the reference table in your plots:

1. For built-in plotting functions:
   - Set show_reference=True when calling plot functions
   - Example: analyzer.total_requests_per_employee(plot=True, show_reference=True)

2. For custom plots:
   - Create your plot using matplotlib
   - Call analyzer.add_reference_table(ax) to add the table
   - Customize position, size, and appearance as needed

3. Available positions:
   - 'upper right', 'upper left', 'lower right', 'lower left'
   - 'center right', 'center left'

4. Customization options:
   - max_rows: limit number of entries shown
   - fontsize: adjust text size
   - alpha: control transparency
   - title: customize table title
""")
