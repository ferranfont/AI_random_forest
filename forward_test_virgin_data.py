import pandas as pd
import sys
import os

# Configuration: Change this to visualize different dates
CSV_FILE = "time_and_sales_nq_20251104"

# Add current dir to path to import analyze script and processing function
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from visualize_ai_signals import visualize_signals

# Import the processing function from clean_data script
from utils.clean_data_csv_to_ticks_per_second import process_historical_file


def forward_test_with_auto_process(csv_file_name, model_path=None):
    """
    Forward test with automatic processing:
    1. Check if TPS file exists in data_ticks_per_second/
    2. If not, process it from data/ using clean_data_csv_to_ticks_per_second
    3. Visualize using the processed file
    """
    
    # Paths - base_dir should be the script's directory (AI_random_forest/)
    base_dir = os.path.dirname(__file__)
    raw_csv = os.path.join(base_dir, 'data', f'{csv_file_name}.csv')
    tps_csv = os.path.join(base_dir, 'data_ticks_per_second', f'tps_{csv_file_name}.csv')
    
    print(f"Forward testing file: {csv_file_name}")
    
    # Check if processed TPS file exists
    if not os.path.exists(tps_csv):
        print(f"⚠️  Processed TPS file not found: {tps_csv}")
        
        # Check if raw file exists
        if not os.path.exists(raw_csv):
            print(f"❌ ERROR: Raw CSV file not found: {raw_csv}")
            print("Please ensure the file exists in the data/ folder.")
            return
        
        # Process the file using the consistent algorithm
        print(f"🔄 Processing raw file using clean_data_csv_to_ticks_per_second algorithm...")
        print(f"   Input: {raw_csv}")
        print(f"   Output: {tps_csv}")
        
        try:
            process_historical_file(raw_csv, tps_csv)
            print("✅ Processing complete!")
        except Exception as e:
            print(f"❌ ERROR during processing: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print(f"✅ Found processed TPS file: {tps_csv}")
    
    # Now visualize using the processed file (with consistent factor_tps calculation)
    print("\n🎨 Running AI Model Visualization...")
    visualize_signals(csv_path=tps_csv, model_path=model_path)

if __name__ == "__main__":
    # Use CSV_FILE variable defined at top
    # Pre-trained model path
    MODEL_PATH = r"d:\PYTHON\ALGOS\AI_random_forest\initiation_model.pkl"
    
    forward_test_with_auto_process(CSV_FILE, MODEL_PATH)
