Model Name: AushadhiNet

Technical Version: AushadhiNet-GATv2-128-CVD

Architecture: Graph Attention Network v2 (Multi-head)

Input: Molecular SMILES + Drug Interaction Topology

Mission: Safeguarding cardiac patients by predicting adverse drug-drug interactions (DDIs) before they happen.


## Model Working:
{
  "interaction": true,
  "type": "pharmacokinetic",
  "side_effects": ["Bradycardia", "Hypotension"],
  "confidence": 0.83
}





# Dhadkan Model Testing Guide

## Overview
This guide explains how to test your trained Dhadkan ECG heart disease detection model with custom inputs.

## Files Created
1. **`test_dhadkan_model.py`** - Main testing interface
2. **`prepare_ecg_data.py`** - Helper to prepare your ECG data
3. **`TESTING_GUIDE.md`** - This guide

## Quick Start

### Step 1: Run the Testing Program
```bash
python test_dhadkan_model.py
```

### Step 2: Choose Testing Option
The program offers 4 options:

#### Option 1: Test with Sample Data
- Uses actual data from your training dataset
- Shows how the model performs on known data
- Compares prediction vs actual label

#### Option 2: Test with Synthetic Data
- Uses computer-generated ECG-like signals
- Good for quick testing without real data
- Creates both "normal" and "abnormal" patterns

#### Option 3: Test with Your Own ECG File
- Upload your own ECG data from CSV files
- Most useful for real-world testing

#### Option 4: Exit
- Closes the program

## Preparing Your Own ECG Data

### Method 1: Use the Data Preparation Helper
```bash
python prepare_ecg_data.py
```

This helper can:
- Create sample ECG data for testing
- Convert text/dat files to CSV format

### Method 2: Manual CSV Preparation
Create a CSV file with your ECG data:

```csv
ecg_values
0.123
0.456
0.789
...
```

## Data Requirements

### Input Format
- **File Type**: CSV format
- **Data Type**: Numerical values (float/int)
- **Length**: Any length (model will handle padding/truncating)
- **Expected Length**: 123,993 data points (based on training data)

### Data Preprocessing (Automatic)
The model automatically:
- **Truncates** data longer than 123,993 points
- **Pads with zeros** data shorter than 123,993 points
- **Reshapes** data for CNN input format

## Understanding Results

### Output Explanation
```
Prediction Probability: 0.0370    # Raw probability (0-1)
Predicted Class: 0                # Binary class (0=Normal, 1=Disease)
Result: NEGATIVE - Normal Heart Activity
Risk Level: LOW RISK             # Risk assessment
Confidence: 96.3%                # Model confidence
```

### Risk Levels
- **LOW RISK**: Probability < 0.2 or > 0.8
- **MODERATE RISK**: Probability between 0.2-0.8
- **HIGH RISK**: Probability > 0.8 for positive cases

### Classes
- **Class 0**: Normal heart activity (NEGATIVE)
- **Class 1**: Heart disease detected (POSITIVE)

## Example Usage Scenarios

### Scenario 1: Testing with Real ECG Data
1. Have your ECG data in CSV format
2. Run `python test_dhadkan_model.py`
3. Choose option 3
4. Enter your CSV file path
5. Review results

### Scenario 2: Quick Model Validation
1. Run `python test_dhadkan_model.py`
2. Choose option 1 (sample data)
3. See how model performs on known data

### Scenario 3: Synthetic Data Testing
1. Run `python test_dhadkan_model.py`
2. Choose option 2
3. Test with computer-generated patterns

## Troubleshooting

### Common Issues

#### Model Not Found
```
[ERROR] Model file not found: Dhadkan-Ens-v1_final.keras
```
**Solution**: Train the model first using `python gemini_file_fixed.py`

#### File Not Found
```
[ERROR] File not found: your_file.csv
```
**Solution**: Check file path and ensure file exists

#### Data Format Issues
```
[ERROR] Could not process file
```
**Solution**: 
- Ensure CSV format
- Check for proper numerical data
- Use `prepare_ecg_data.py` to convert other formats

### Performance Notes
- **Large Files**: Files with >100k data points may take longer to process
- **Memory**: Very large ECG files might require more RAM
- **Speed**: First prediction is slower due to model loading

## Advanced Usage

### Custom Threshold
You can modify the classification threshold in `test_dhadkan_model.py`:
```python
self.threshold = 0.5  # Change this value (0.0 to 1.0)
```

### Batch Testing
To test multiple files, modify the script to loop through a directory of CSV files.

### Integration
The `DhadkanTester` class can be imported and used in your own applications:
```python
from test_dhadkan_model import DhadkanTester

tester = DhadkanTester()
tester.load_model()
prob, pred_class = tester.predict(your_ecg_data)
```

## Model Performance Notes

### Current Model Characteristics
- **Training Accuracy**: ~96%
- **Class Imbalance**: Heavily skewed toward normal cases
- **Bias**: May under-predict positive cases due to imbalance

### Recommendations
- Consider multiple predictions for critical decisions
- Validate with medical professionals
- Use as screening tool, not diagnostic tool

## Next Steps

1. **Test with your data**: Start with option 3 using your ECG files
2. **Validate results**: Compare with known diagnoses if available
3. **Adjust threshold**: Experiment with different classification thresholds
4. **Collect feedback**: Note any misclassifications for model improvement

## Support

If you encounter issues:
1. Check this guide first
2. Verify file formats and paths
3. Ensure model was trained successfully
4. Check Python environment and dependencies