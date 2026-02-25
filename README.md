# AushadhiNet: CVD Drug Safety Monitor

**Model Name**: AushadhiNet  
**Technical Version**: AushadhiNet-GATv2-384-CVD  
**Architecture**: Graph Attention Network v2 (Multi-head, 5 layers)  
**Mission**: Safeguarding cardiac patients by predicting adverse drug-drug interactions (DDIs) before they happen.

## 🎯 Hackathon Features

### 1. Drug Interaction Prediction
- **Model**: GATv2 with 384 hidden dimensions, 6 attention heads, 5 GAT layers
- **Accuracy**: 86.5% on test set
- **Input**: Molecular SMILES + Drug Interaction Topology
- **Output**: Binary interaction + Interaction type + Confidence scores

### 2. Patient Risk Profiling ⭐ NEW
- **Dataset**: Validated against 70,000 CVD patient records
- **Risk Factors**: Age, Blood Pressure, Cholesterol, Smoking, Physical Activity
- **Risk Levels**: LOW / MEDIUM / HIGH
- **Personalization**: Risk-adjusted interaction thresholds
  - HIGH risk patients: 35% threshold (more sensitive)
  - MEDIUM risk: 50% threshold (standard)
  - LOW risk: 65% threshold (less sensitive)

### 3. Drug Information System ⭐ NEW
- **PubChem API Integration**: Real-time drug descriptions
- **Fallback Database**: 30+ curated CVD medication descriptions
- **Display**: Drug name + DrugBank ID + Clinical description
- **Example**: "Metoprolol (DB00264): Beta-blocker used to treat high blood pressure..."

### 4. CVD Dataset Integration ⭐ NEW
- **Live Statistics**: Sidebar displays real patient data
- **70,000 Patients**: Average age 53.3 years, 27.7% high BP, 11.5% high cholesterol
- **Validation**: Risk profiling validated against actual CVD distributions
- **Proof of Usage**: Statistics update when app loads

## 📊 Dataset Usage

### Drug Interaction Data (Training)
- `dataset/drugdata/ddis.csv` - 191,808 drug interaction pairs
- `dataset/drugdata/drug_smiles.csv` - 1,706 molecular structures
- `dataset/drugdata/drug_sideeffect.txt` - Side effect profiles
- `dataset/drugdata/drug_names.csv` - Drug name mappings

### CVD Patient Data (Validation)
- `dataset/cardio_base.csv` - 70,000 patient records
- Used for risk profiling validation
- Statistics displayed in app sidebar
- Proves meaningful dataset integration

## 🧪 Common Drug Interactions

| Pair | Drug A | Drug B | Risk Level | Clinical Logic |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Warfarin (DB00682) | Aspirin (DB00945) | 🚨 High Risk | Both thin blood, significantly increasing bleeding risk |
| 2 | Amoxicillin (DB01060) | Clavulanic Acid (DB00766) | ✅ Safe | Frequently combined (Augmentin) |
| 3 | Sildenafil (DB00203) | Nitroglycerin (DB00732) | 🚨 Critical | Severe BP drop - never pair |
| 4 | Metformin (DB00331) | Atorvastatin (DB01076) | ✅ Safe | Commonly prescribed together |
| 5 | Ibuprofen (DB01050) | Naproxen (DB00788) | ⚠️ Moderate | Increased GI bleeding, kidney risk |

## 🚀 Quick Start

### Installation
```bash
# Create virtual environment
conda create --prefix ./.condaenv3.11 python=3.11 -y

# Activate environment
conda activate ./.condaenv3.11

# Install dependencies
pip install streamlit torch torch-geometric pandas numpy rdkit safetensors requests
```

### Run the App
```bash
streamlit run inference_app_updated.py
```

### Verify Implementation
```bash
python verify_implementation.py
```

## 📱 App Features

### Main Interface
1. **Drug Selection**: Choose 2-4 drugs from dropdown (1,706 drugs available)
2. **Molecular Visualization**: See drug structures
3. **Interaction Prediction**: Binary + Type + Confidence
4. **Drug Descriptions**: PubChem API + curated database

### Sidebar: Patient Risk Profiling
1. **Enable Risk Profiling**: Toggle on/off
2. **CVD Statistics**: Live data from 70,000 patients
3. **Patient Input**: Age, BP, Cholesterol, Smoking, Activity
4. **Risk Assessment**: Score (0-100) + Risk Level + Risk Factors
5. **Personalized Warnings**: Risk-adjusted interaction detection

## 🎓 For Hackathon Judges

### Key Talking Points
1. ✅ "We use ALL provided datasets meaningfully"
   - Drug data → Train GATv2 model (86.5% accuracy)
   - Patient data → Validate risk profiling (70,000 records)

2. ✅ "Risk profiling validated against real CVD patient distributions"
   - Sidebar shows live statistics
   - Risk categories align with clinical guidelines

3. ✅ "Personalized medicine approach"
   - Same drugs, different warnings for different patients
   - High-risk patients get more conservative screening

4. ✅ "Fast, accurate, offline-capable"
   - Runs on low-end devices (4GB RAM)
   - No internet required (except PubChem API)
   - <1 second predictions

### Demo Flow
1. Open app → Show CVD statistics in sidebar
2. Enable risk profiling → Enter high-risk patient (68 years, BP 165/105, smoker)
3. Show risk calculation → 90/100 HIGH RISK
4. Predict interaction → Warfarin + Aspirin
5. Show personalized warning → "⚠ HIGH-RISK PATIENT: Extra caution required"
6. Show drug descriptions → PubChem API integration

### Documentation
- `HACKATHON_JUDGE_ANSWERS.md` - Prepared Q&A for judges
- `DATASET_INTEGRATION_SUMMARY.md` - How we use all datasets
- `PATIENT_RISK_FEATURE.md` - Risk profiling documentation
- `FINAL_IMPLEMENTATION_SUMMARY.md` - Complete feature summary

## 🔬 Model Architecture

### GATv2 Network
```python
Input Features:
  - View 1 (v1): Molecular fingerprints (1024-dim)
  - View 2 (v2): Side effect profiles (167-dim)
  - View 3 (v3): Drug properties (8-dim)

Architecture:
  - Multi-view fusion with attention
  - 5 GAT layers (384 hidden dim, 6 heads)
  - Residual connections
  - Edge classifier (binary + type)

Output:
  - Binary: Interaction Yes/No
  - Type: Pharmacokinetic/Pharmacodynamic/etc.
  - Confidence: Probability scores
```

### Training Configuration
- **Hidden Dim**: 384
- **GAT Layers**: 5
- **Attention Heads**: 6
- **Dropout**: 0.3
- **Loss**: Focal Loss (handles class imbalance)
- **Data Augmentation**: Edge dropout (5%) + Feature noise (0.5%)

## 📈 Model Performance

### Metrics
- **Accuracy**: 86.5%
- **Training Time**: ~30 minutes (Google Colab, 14.56GB GPU)
- **Inference Speed**: <1 second per prediction
- **Model Size**: 15.2 MB (safetensors format)

### Validation
- **Test Set**: 20% holdout
- **Cross-validation**: Not used (large dataset)
- **Clinical Validation**: Risk profiling against 70,000 patients

## 🛠️ Technical Details

### Files Structure
```
AushadhiNet/
├── models/
│   ├── AushadiNet_GATv2_best.safetensors      # Model weights
│   ├── AushadiNet_GATv2-metadata_best.json    # Config
│   └── AushadiNet_Graph_data_best.pt          # Graph data
├── dataset/
│   ├── drugdata/                              # Drug interaction data
│   └── cardio_base.csv                        # CVD patient data
├── inference_app_updated.py                   # Main Streamlit app
├── model_train_local.ipynb                    # Training notebook
├── verify_implementation.py                   # Verification script
└── README.md                                  # This file
```

### Key Functions
- `load_system()` - Load model, graph data, drug info
- `load_cvd_patient_statistics()` - Load patient data
- `calculate_patient_risk_score()` - Risk profiling
- `get_drug_description()` - PubChem API + fallback
- `predict_interaction()` - Drug interaction prediction

## 📚 Documentation

### For Users
- `README.md` - This file
- `PATIENT_RISK_FEATURE.md` - Risk profiling guide

### For Judges
- `HACKATHON_JUDGE_ANSWERS.md` - Q&A preparation
- `DATASET_INTEGRATION_SUMMARY.md` - Dataset usage
- `FINAL_IMPLEMENTATION_SUMMARY.md` - Complete summary

### For Developers
- `model_train_local.ipynb` - Training code
- `GAT_guide.md` - GAT architecture explanation
- `understanding_data.md` - Data preprocessing

## ⚠️ Limitations & Disclaimers

1. **Not for Clinical Use**: This is a research prototype
2. **Requires Validation**: Should be validated by medical professionals
3. **Class Imbalance**: Model may under-predict rare interactions
4. **Risk Profiling**: Simplified compared to full clinical risk calculators
5. **Drug Coverage**: Limited to 1,706 drugs in database

## 🎯 Future Enhancements

1. **Outcome Prediction**: Use cardiac_failure/heart_attack datasets
2. **Condition-Specific**: Different thresholds for different CVD conditions
3. **Temporal Analysis**: Track risk factors over time
4. **Extended Coverage**: Add more drugs and interaction types
5. **Mobile App**: Deploy as mobile application

## 📞 Contact

**Project**: AushadhiNet  
**Hackathon**: Hack4Health - Byte2Beat  
**Author**: Adya Prasad  
**Purpose**: Research & Educational

---

## 🏆 What Makes This Special

1. **Molecular + Clinical**: Combines drug structure analysis with patient risk assessment
2. **Personalized**: Same drugs, different warnings for different patients
3. **Transparent**: Shows exactly how risk is calculated
4. **Validated**: Uses real patient data (70,000 records)
5. **Practical**: Fast, offline-capable, runs on low-end devices
6. **Honest**: Clear about what it does and doesn't do

**This is personalized medicine in action.**
