# AushadiNet-GATv2: Graph Attention Network for Predicting Drug-Drug Interactions in  Cardiovascular Polypharmacy

**Model Name**: AushadhiNet  
**Technical Version**: AushadhiNet-GATv2-DDI 
**Architecture**: Graph Attention Network v2 (Multi-head, multiple layers)  
**Mission**: Safeguarding cardiac patients by predicting adverse drug-drug interactions (DDIs) before they happen.

As per the World Heart Report 2025, Cardiovascular disease (CVD) remains the leading cause of death globally, with a projected 20.5 million deaths in 2025, rising to 35.6 million by 2050. And the cardiovascular treatment experiences some of the highest rates of adverse drug reactions (ADRs) primarily due to **polypharmacy** the fact that heart patients typically require multiple simultaneous medications and according to recent _pharmacological studies_ indicating that over **_53%_** of these ADRs are potentially avoidable. ML models trained on massive amounts of therapeutically relevant data may help physicians make more informed clinical decisions before prescribing drugs.

Existing DDI prediction systems rely on manually curated databases covering limited drugs and require manual mapping of new drugs, and memorise interactions instead of learning molecular patterns. I present **AushadhiNet-GATv2** , a graph neural network architecture that learns molecular-level interaction patterns from chemical structures

## What it does
AushadhiNet-GATv2 is a graph neural network model that predicts drug-pair interactions with probability scores and classifies interaction types (potential adverse drug reactions) across 86 mechanisms. The system alerts physicians and patients before prescribing or taking medications, preventing harmful drug combinations.

I integrate it with streamlit application with manual drugs entry  automated OCR-powered image scanning and patient profiling for production-level use and serving end users.

### Key features:
- **High Accuracy**: 90% accuracy and 97.94% recall on 19,000 DDI pair
- **Prescription Scanner**: OCR-powered image scanning with fuzzy drug name matching
- **Risk Profiling**: Patient-specific cardiovascular risk stratification validated on 70,000 CVD patient records
- **Real-time Inference**: <100ms prediction latency, deployable on consumer hardware
- **Multi-drug Analysis**: Checks all pairwise combinations for upto 4 concurrent medications
- **Extensible Architecture**: Supports fine-tuning on additional datasets (new drugs, rare interactions, institution-specific data) for continuous accuracy improvement

##  Dataset Usage
### Drug Interaction Data (Training)
- `dataset/drugdata/ddis.csv` - 191,808 drug interaction pairs
- `dataset/drugdata/drug_smiles.csv` - 1,706 molecular structures
- `dataset/drugdata/drug_names.csv` - Side effect profiles
- `dataset/drugdata/ddi_type_mapping.json`

### CVD Patient Data (Validation)
- `dataset/cardio_base.csv` - 70,000 patient records
- Used for risk profiling validation
- Statistics displayed in app sidebar
- Proves meaningful dataset integration

##  Quick Start

### Installation
```bash
# Create virtual environment
conda create --prefix ./.condaenv3.11 python=3.11 -y

# Activate environment
conda activate ./.condaenv3.11

# Install dependencies
pip install requirements.txt
```

### Run
```
<run> mdel training jupyter notebook
```

### Run the streamlit app
```bash
streamlit run streamlit_app.py
```

## Run with Coder Workspace

This project uses [Coder](https://coder.com) for reproducible Cloud Development Environments.

### Prerequisites
- [Coder installed](https://coder.com/docs/install)
- Docker Desktop running

### Launch Workspace
```bash
coder server                                              # Start Coder server
coder login http://localhost:3000                        # Authenticate
coder templates push ddi-predictor --directory coder-template/
coder create ddi-workspace --template ddi-predictor     # Spin up workspace

# to stop
coder stop ddi-workspace
```

Open http://localhost:3000 → click **"DDI Predictor App"** → Streamlit launches instantly.

