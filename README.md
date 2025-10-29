# KalmanFilter Tracking

This project performs **object tracking** in videos or image sequences using Kalman Filters.  
It implements **three different models**:

- **CV (Constant Velocity)**: Assumes constant speed.  
- **CA (Constant Acceleration)**: Assumes constant acceleration.  
- **CT (Constant Turn)**: Assumes constant turn rate.  

These three models are combined using **IMM (Interacting Multiple Model)** to achieve more accurate and robust tracking.

## Installation

1. Use **Python 3.10 or higher**.  
2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

3. Install the required packages:
pip install -r requirements.txt

To run this project, download the **GOT-10k Validation** dataset:

- [Download GOT-10k Validation (Kaggle)](https://www.kaggle.com/datasets/abhimanyukarshni/got10k?resource=download-directory&select=val)  
- Place the downloaded data inside a folder named `val/` in the project directory.  
- Example path: `val/GOT-10k_Val_000001/`