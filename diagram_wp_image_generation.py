import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

def create_gantt_chart():
    # Define the project timeline data
    # Reversed order so WP1 appears at the top of the chart
    data = [
        {'Task': 'WP 5. Project Report', 'Start': '2026-03-16', 'End': '2026-06-30'},
        {'Task': 'WP 4. Testing', 'Start': '2026-06-08', 'End': '2026-06-18'},
        {'Task': 'WP 3. Deep Learning model\n& software pipeline', 'Start': '2026-04-21', 'End': '2026-06-07'},
        {'Task': 'WP 2. Data Augmentation & Heuristics', 'Start': '2026-03-16', 'End': '2026-04-20'},
        {'Task': 'WP 1. Introduction and research', 'Start': '2026-01-20', 'End': '2026-03-15'}
    ]

    df = pd.DataFrame(data)
    df['Start'] = pd.to_datetime(df['Start'])
    df['End'] = pd.to_datetime(df['End'])
    df['Duration'] = (df['End'] - df['Start']).dt.days

    # Setup the plot
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#ffffff')

    # Create the horizontal bars
    ax.barh(df['Task'], df['Duration'], left=df['Start'], color='#4A7EBB', edgecolor='#2B3B60', height=0.6, zorder=3)

    # Format the grid and axes
    ax.grid(axis='x', linestyle='--', color='gray', alpha=0.5, zorder=0)
    
    # Format the X-axis (Dates)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.xticks(rotation=45, fontsize=12)
    
    # Format the Y-axis (Labels - made much bigger as requested)
    plt.yticks(fontsize=14, fontweight='bold', color='#2B3B60')

    # Title and layout
    plt.title('4.3 Time Plan (Gantt Diagram)', fontsize=20, fontweight='bold', pad=20, color='#2B3B60')
    plt.tight_layout()
    
    # Save the Gantt chart
    plt.savefig('Gantt_Chart.png', dpi=300)
    print("Successfully saved 'Gantt_Chart.png'")


def create_wbs():
    # Setup the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off') # Hide axes

    dark_blue = '#2B3B60'
    light_blue = '#4A7EBB'

    # 1. Main Title Box (Top Center)
    ax.text(50, 92, 
            "Channel Knowledge Map Prediction with Deep Learning\nfor 6G UAV enabled Networks", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='white',
            bbox=dict(boxstyle="square,pad=1.5", facecolor=dark_blue, edgecolor='none', zorder=3))

    # 2. WP Data and positions (x, y, text)
    wps = [
        (50, 76, "WP 1. INTRODUCTION AND RESEARCH"),
        (30, 56, "WP 2. DATA AUGMENTATION\nAND HEURISTICS"),
        (30, 36, "WP 3. DEEP LEARNING MODEL\nAND SOFTWARE PIPELINE"),
        (30, 16, "WP 4. TESTING"),
        (70, 56, "WP 5. PROJECT REPORT")
    ]

    # 3. Draw Connecting Lines
    ax.plot([50, 50], [92, 76], color='#8FAADC', lw=3, zorder=1) # Main Title to WP1
    ax.plot([50, 50], [76, 68], color='#8FAADC', lw=3, zorder=1) # WP1 down to horizontal branch
    ax.plot([30, 70], [68, 68], color='#8FAADC', lw=3, zorder=1) # Horizontal branch
    
    ax.plot([30, 30], [68, 56], color='#8FAADC', lw=3, zorder=1) # Branch down to WP2
    ax.plot([70, 70], [68, 56], color='#8FAADC', lw=3, zorder=1) # Branch down to WP5 (Parallel)
    
    ax.plot([30, 30], [56, 36], color='#8FAADC', lw=3, zorder=1) # WP2 down to WP3
    ax.plot([30, 30], [36, 16], color='#8FAADC', lw=3, zorder=1) # WP3 down to WP4
    
    # 4. Draw the WP Boxes
    for x, y, text in wps:
        ax.text(x, y, text,
                ha='center', va='center', fontsize=12, fontweight='bold', color='white',
                bbox=dict(boxstyle="square,pad=1.5", facecolor=light_blue, edgecolor='none', zorder=3))

    # Title and Layout
    plt.title('4.1 Work Breakdown Structure', fontsize=20, fontweight='bold', pad=20, color=dark_blue)
    plt.tight_layout()
    
    # Save the WBS chart
    plt.savefig('Work_Breakdown_Structure.png', dpi=300)
    print("Successfully saved 'Work_Breakdown_Structure.png'")

if __name__ == "__main__":
    create_gantt_chart()
    create_wbs()