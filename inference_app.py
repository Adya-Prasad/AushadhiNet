import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json
import itertools
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from safetensors.torch import load_file
from rdkit import Chem
from rdkit.Chem import Draw

# --- 1. CONFIGURATION & PATHS ---
MODEL_WEIGHTS = 'models/AushadiNet_GATv2_best.safetensors'
MODEL_CONFIG = 'models/AushadiNet_GATv2-metadata_best.json'
GRAPH_DATA = 'models/AushadiNet_Graph_data_best.pt'
SMILES_PATH = 'dataset/drugdata/drug_smiles.csv'
NAMES_PATH = 'dataset/drugdata/drug_names.csv'


st.set_page_config(
    page_title="AushadhiNet: CVD Drug Safety Monitor", 
    page_icon="💓",
    layout="centered"
)

# --- 2. MODEL ARCHITECTURE (MUST MATCH YOUR TRAINING MODEL!) ---
class GATv2NN(nn.Module):
    """
    This MUST match the architecture used during training!
    Based on your metadata: hidden_dim=384, n_heads=6, n_gat_layers=3
    """
    def __init__(self, dims, hidden_dim, n_heads, n_types, dropout=0.3, n_gat_layers=3):
        super().__init__()
        
        # Validate
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        
        self.hidden_dim = hidden_dim
        self.n_gat_layers = n_gat_layers
        
        # View Projectors (FIXED: Use BatchNorm + ReLU, not LayerNorm + GELU)
        self.proj_v1 = nn.Sequential(
            nn.Linear(dims['v1'], hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        self.proj_v2 = nn.Sequential(
            nn.Linear(dims['v2'], hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        self.proj_v3 = nn.Sequential(
            nn.Linear(dims['v3'], hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Feature Attention (FIXED: Use // 4 not // 2)
        self.feat_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Dynamic GAT Layers (FIXED: Match training architecture)
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for i in range(n_gat_layers):
            if i == 0:
                # First layer: multi-head with concatenation
                gat = GATv2Conv(
                    hidden_dim, 
                    hidden_dim // n_heads,
                    heads=n_heads, 
                    concat=True,
                    dropout=dropout
                )
            else:
                # Subsequent layers: single head
                gat = GATv2Conv(
                    hidden_dim, 
                    hidden_dim, 
                    heads=1, 
                    concat=False, 
                    dropout=dropout
                )
            
            self.gat_layers.append(gat)
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
        
        # Edge Classifier (FIXED: Match training - simpler architecture)
        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        
        self.head_bin = nn.Linear(hidden_dim // 2, 2)
        self.head_type = nn.Linear(hidden_dim // 2, n_types)
    
    def get_node_embeddings(self, x_v1, x_v2, x_v3, edge_index):
        """Generate node embeddings from multi-view features."""
        # Project views
        h1 = self.proj_v1(x_v1)
        h2 = self.proj_v2(x_v2)
        h3 = self.proj_v3(x_v3)
        
        # Multi-view fusion
        stack = torch.stack([h1, h2, h3], dim=1)
        scores = F.softmax(self.feat_attention(stack), dim=1)
        h_fused = torch.sum(stack * scores, dim=1)
        
        # Apply GAT layers with residual connections
        h = h_fused
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norm_layers)):
            h_residual = h
            h = gat(h, edge_index)
            h = norm(h + h_residual)  # Residual connection
            if i == 0:
                h = F.elu(h)  # Activation after first layer
        
        return h
    
    def forward_edges_from_emb(self, node_emb, edge_label_index):
        """Predict edge interactions from node embeddings."""
        src, dst = edge_label_index[0], edge_label_index[1]
        edge_feat = torch.cat([node_emb[src], node_emb[dst]], dim=-1)
        shared = self.edge_encoder(edge_feat)
        return self.head_bin(shared), self.head_type(shared)
    
    def forward(self, x_v1, x_v2, x_v3, edge_index, edge_label_index):
        """Full forward pass (for inference)."""
        node_emb = self.get_node_embeddings(x_v1, x_v2, x_v3, edge_index)
        return self.forward_edges_from_emb(node_emb, edge_label_index)


# --- 3. ROBUST RESOURCE LOADING ---
# --- 3. ROBUST RESOURCE LOADING ---
@st.cache_resource
def load_system():
    """Load model, metadata, graph data, and drug information."""
    
    # Load Metadata
    if not os.path.exists(MODEL_CONFIG):
        raise FileNotFoundError(f"Config JSON not found at {MODEL_CONFIG}")
    
    with open(MODEL_CONFIG, 'r') as f:
        metadata_list = json.load(f)
    
    metadata = metadata_list[0] if isinstance(metadata_list, list) else metadata_list
    config = metadata.get('training_onfiguration', metadata.get('training_configuration', {}))
    
    if not config:
        raise ValueError("Could not find training configuration in metadata JSON")
    
    print(f"✓ Loaded config: hidden_dim={config['hidden_dim']}, n_heads={config['n_heads']}, n_gat_layers={config['n_gat_layers']}")
    
    # Load Graph Data
    if not os.path.exists(GRAPH_DATA):
        raise FileNotFoundError(f"Graph data not found at {GRAPH_DATA}")
    
    # Fix for PyTorch 2.6: Allow sklearn LabelEncoder
    try:
        import torch.serialization
        from sklearn.preprocessing import LabelEncoder
        
        # Method 1: Add to safe globals
        torch.serialization.add_safe_globals([LabelEncoder])
        graph_data = torch.load(GRAPH_DATA, map_location='cpu', weights_only=False)
    except Exception as e:
        # Method 2: Use context manager (fallback)
        print(f"⚠️  Trying alternative loading method...")
        with torch.serialization.safe_globals([LabelEncoder]):
            graph_data = torch.load(GRAPH_DATA, map_location='cpu', weights_only=False)
    drug_map = graph_data['drug_map']
    
    print(f"✓ Loaded graph data with {len(drug_map)} drugs")
    
    # Load SMILES
    if not os.path.exists(SMILES_PATH):
        raise FileNotFoundError(f"SMILES file not found at {SMILES_PATH}")
    
    smiles_df = pd.read_csv(SMILES_PATH)
    
    # Load drug names
    name_dict = {}
    if os.path.exists(NAMES_PATH):
        names_df = pd.read_csv(NAMES_PATH)
        name_dict = dict(zip(names_df['drug_id'], names_df['drug_name']))
        print(f"✓ Loaded {len(name_dict)} drug names")
    else:
        name_dict = {drug_id: drug_id for drug_id in drug_map.keys()}
        print("⚠️  Drug names file not found, using IDs as names")
    
    # Initialize Model
    model = GATv2NN(
        dims={'v1': 1024, 'v2': 167, 'v3': 8},
        hidden_dim=config['hidden_dim'],
        n_heads=config['n_heads'],
        n_types=graph_data['n_types'],
        dropout=config['dropout'],
        n_gat_layers=config['n_gat_layers']
    )
    
    # Load weights
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS}")
    
    state_dict = load_file(MODEL_WEIGHTS)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    return model, drug_map, smiles_df, name_dict, config, graph_data


# --- 4. PREDICTION FUNCTION ---
def predict_interaction(model, graph_data, drug_a_id, drug_b_id, threshold=0.5, device='cpu'):
    """
    Predict drug-drug interaction using the trained model.
    
    This is the REAL prediction function (not placeholder).
    Based on your training notebook's predict_cardio_interaction_safe function.
    """
    model.eval()
    
    drug_map = graph_data['drug_map']
    
    # Check if drugs exist
    if drug_a_id not in drug_map or drug_b_id not in drug_map:
        return {
            'error': True,
            'message': f"Drug ID not found in database. Available drugs: {len(drug_map)}"
        }
    
    # Get node indices
    u = drug_map[drug_a_id]
    v = drug_map[drug_b_id]
    
    query_node_indices = torch.tensor([u, v], dtype=torch.long)
    
    # Extract 1-hop subgraph (same as training)
    from torch_geometric.utils import k_hop_subgraph
    
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=query_node_indices,
        num_hops=1,
        edge_index=graph_data['edge_index'],
        relabel_nodes=True
    )
    
    subgraph_u = mapping[0]
    subgraph_v = mapping[1]
    query_edge = torch.tensor([[subgraph_u], [subgraph_v]], device=device)
    
    # Get features
    x_v1 = graph_data['x_v1'][subset].to(device)
    x_v2 = graph_data['x_v2'][subset].to(device)
    x_v3 = graph_data['x_v3'][subset].to(device)
    edge_index = edge_index.to(device)
    
    with torch.no_grad():
        # Get predictions
        logits_bin, logits_type = model(x_v1, x_v2, x_v3, edge_index, query_edge)
        
        # Binary prediction
        probs_bin = torch.softmax(logits_bin, dim=1)[0]
        prob_interaction = probs_bin[1].item()
        binary_pred = 1 if prob_interaction > threshold else 0        
        
        # Type prediction
        probs_type = torch.softmax(logits_type, dim=1)[0]
        type_idx = torch.argmax(logits_type, dim=1).item()
        type_prob = probs_type[type_idx].item()
        
        # Get type name
        try:
            type_name = graph_data['encoder'].inverse_transform([type_idx])[0]
        except:
            type_name = str(type_idx)
    
    return {
        'error': False,
        'binary_prediction': binary_pred,
        'interaction_probability': prob_interaction,
        'predicted_type': type_name,
        'type_probability': type_prob,
        'drug_a': drug_a_id,
        'drug_b': drug_b_id
    }


# --- INITIALIZATION ---
try:
    model, drug_map, smiles_df, name_dict, config, graph_data = load_system()
    smiles_dict = dict(zip(smiles_df['drug_id'], smiles_df['smiles']))
    
    # Load metadata for display
    with open(MODEL_CONFIG, 'r') as f:
        metadata_list = json.load(f)
    metadata = metadata_list[0] if isinstance(metadata_list, list) else metadata_list
    
except Exception as e:
    st.error(f"✘ Initialization Failed: {e}")
    st.stop()


# Helper for the searchable dropdown
def get_label(drug_id):
    name = name_dict.get(drug_id, "Unknown")
    return f"{name} ({drug_id})"



# --- 5. UI STYLING & LAYOUT ---
st.markdown("""
<style>

/* ---------------- APP LAYOUT ---------------- */

.stApp {
    background-color: #262624;
}

.block-container {
    max-width: 1300px;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* ---------------- HEADERS ---------------- */

.main-header {
    font-size: 2rem;
    color: #b06045;
    text-align: center;
    font-weight: 800;
    margin-bottom: 0;
}

.paragraph {
    font-size: 1.15rem;
    color: #dbcfcc;
    text-align: center; 
    padding-bottom: 0.5rem;
}

.section-heading {
    color: #cb785c; 
    font-size: 1.3rem;
    font-weight: 600;
    text-align: center;
    padding-top: 2rem;   
    border-top: 1px solid #dbcfcc;
}

/* Outer box */
div[data-baseweb="select"] > div {
    background-color: #ecebe3 !important;
    border-radius: 10px !important;
    border: none !important;
}

/* Typed text */
div[data-baseweb="select"] input {
    color: #392d2b !important;
    caret-color: #392d2b !important;
}

/* Selected value */
div[data-baseweb="select"] div {
    color: #392d2b !important;
}

/* Placeholder */
div[data-baseweb="select"] input::placeholder {
    color: #392d2b !important;
    opacity: 0.8 !important;
}
div[data-testid="stButton"] {
    display: flex;
    align-items: flex-end;
}

.drug-title {
    text-align: center;
    font-size: 1.1rem;
    font-weight: 700;
    color: #666;
}
div[data-testid="stButton"] > button {
    background-color: #cb785c !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600;
    padding: 0.6rem 2.5rem;
    transition: all 0.2s ease-in-out;
}

div[data-testid="stButton"] > button:hover {
    background-color: #b06045 !important;

}
div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 2px solid #D2691E !important;
            background-color: #FFFFFF !important;
            border-radius: 12px;
        }
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #ecebe3;
    color: #4e5e6c;
    text-align: center;
    padding: 12px;
    font-size: 0.9rem;
    z-index: 1000;
    border-radius: 20px 20px 0px 0px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">AushadhiNet: Monitoring CVD Drugs Prescription</div>', unsafe_allow_html=True)
st.markdown('<div class="paragraph">AushadhiNet-GATv2: A lightweight and very accuarate model for validating CVD and commonly prescribed medications to determine if they may interact negatively, and can run on low-level devices and offline.<br> Hack4Health: Byte2Beat Hackathon, Research Project by Adya Prasad</div>', unsafe_allow_html=True)

st.markdown('<div class="section-heading">Enter Drugs Combinations! (upto 4)</div>', unsafe_allow_html=True)

if "drug_count" not in st.session_state:
    st.session_state.drug_count = 2

MAX_DRUGS = 4

# Create dynamic columns (drug inputs + add button)
total_columns = st.session_state.drug_count + (
    1 if st.session_state.drug_count < MAX_DRUGS else 0
)

cols = st.columns(total_columns)

selected_drugs = []

for i in range(st.session_state.drug_count):
    with cols[i]:
        drug = st.selectbox(
            f"Drug {i+1}",
            list(drug_map.keys()),
            format_func=get_label,
            key=f"drug_{i}"
        )
        selected_drugs.append(drug)

# add button in last column
if st.session_state.drug_count < MAX_DRUGS:
    with cols[-1]:
        st.write(" ")
        if st.button("十"):
            st.session_state.drug_count += 1
            st.rerun()


if st.button("PREDICT"):

    # Remove duplicates
    unique_drugs = list(dict.fromkeys(selected_drugs))

    if len(unique_drugs) < 2:
        st.warning("⚠︎ Please select at least two distinct drugs.")
    elif len(set(selected_drugs)) != len(selected_drugs):
        st.warning("⚠︎ Please select different drugs in each field.")
        st.stop()
    else:
        pairs = list(itertools.combinations(unique_drugs, 2))

        for drug_a, drug_b in pairs:

            with st.spinner(f"Analyzing {name_dict.get(drug_a)} + {name_dict.get(drug_b)} ..."):
                result = predict_interaction(
                    model,
                    graph_data,
                    drug_a,
                    drug_b,
                    threshold=0.5,
                    device='cpu'
                )

            if result.get('error'):
                st.error(f"✘ {result['message']}")
                continue

            name_a = name_dict.get(drug_a, drug_a)
            name_b = name_dict.get(drug_b, drug_b)

            score = result['interaction_probability']
            safe_prob = 1 - score
            binary_pred = result['binary_prediction']
            pred_type = result['predicted_type']
            type_prob = result['type_probability']

            # ---- Generate Molecular Images ----
            mol_a = Chem.MolFromSmiles(smiles_dict.get(drug_a, ""))
            mol_b = Chem.MolFromSmiles(smiles_dict.get(drug_b, ""))

            img_a = Draw.MolToImage(mol_a, size=(200, 200)) if mol_a else None
            img_b = Draw.MolToImage(mol_b, size=(200, 200)) if mol_b else None
            # ---------- WRAPPER START ----------
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 1.3, 1])
                with col1:
                    st.markdown(f'<div class="paragraph">{name_a}</div>', unsafe_allow_html=True)
                    if img_a:
                        st.image(img_a, width='stretch')
                with col2:
                    text_color = "#ce0000" if binary_pred == 1 else "#0e992a"
                    status_text = "⚠︎ ADVERSE INTERACTION DETECTED " if binary_pred == 1 else "(✔) SAFE PAIR, NO INTERACTION DETECTED"

                    st.markdown(f"""
                    <div style="padding:0.5rem; align-item:center; display: inline;">
                        <ul style="font-size:1.05rem; line-height:1.6; padding-left:1.2rem;">
                        <li style="font-weight:600; font-size:1.14rem; margin-bottom:8px; color:{text_color};">{status_text}</li>
                        <li>Drug Pair: <b >{name_a} + {name_b}</b></li>
                        {f"<li>Interaction Probability: <b> {score:.2%}</b></li>" if binary_pred == 1 else f"<li>Safe Probability: <b> {safe_prob:.2%}</b></li>"}
                        {f"<li>Interaction Type: <b> {pred_type}</b></li>" if binary_pred == 1 else ""}
                        {f"<li>Interaction Type Probability: <b> {type_prob:.2%}</b></li>" if binary_pred == 1 else ""}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="paragraph">{name_b}</div>', unsafe_allow_html=True)
                    if img_b:
                        st.image(img_b, width='stretch')

# Footer
st.markdown(
    f"""
    <div class='footer'>
        ● Model: AushadiNet_GATv2 ({config['hidden_dim']}D, {config['n_gat_layers']} layers) 
        ● Hackathon Version: For Research & Educational Purposes Only
    </div>
    """,
    unsafe_allow_html=True
)