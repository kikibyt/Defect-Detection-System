from graphviz import Digraph

# Set rankdir='LR' for left-to-right layout
dot = Digraph(comment='Infographic Defect Detection Pipeline', format='png', graph_attr={'rankdir': 'LR'})

# -------------------------
# Data (Blue, with emoji)
# -------------------------
dot.node('A', '🗄️ Data Loading\n(MVTecDataset)\ntrain/test split', style='filled', fillcolor='lightblue')
dot.node('B', '🖼️ Data Preprocessing\nResize, Normalize, ToTensor', style='filled', fillcolor='lightblue')

# -------------------------
# Model / Code (Purple, gear emoji)
# -------------------------
dot.node('C', '⚙️ Model Definition\nAutoencoder\n(encoder + decoder)', style='filled', fillcolor='plum')
dot.node('D', '⚙️ Training Loop\n- Loss: MSE + (1 - SSIM)\n- Optimizer: Adam\n- Compute AUROC', style='filled', fillcolor='plum')
dot.node('H', '☁️ FastAPI Deployment\n- Load ONNX\n- /detect endpoint\n- Drift Detection\n- Returns defect prediction', style='filled', fillcolor='plum')

# -------------------------
# Artifacts / Outputs (Pink, box emoji)
# -------------------------
dot.node('E', '📦 Save Best Model\n(autoencoder_best.pth)', style='filled', fillcolor='lightpink')
dot.node('F', '📦 Compute Threshold\n(threshold.npy)', style='filled', fillcolor='lightpink')
dot.node('G', '📦 Export ONNX Model\n(autoencoder.onnx)', style='filled', fillcolor='lightpink')

# -------------------------
# Connections
# -------------------------
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH'])

# -------------------------
# Legend
# -------------------------
dot.node('I', 'Legend:\n🗄️ Blue: Data\n⚙️ Purple: Code/Model\n📦 Pink: Artifacts\n☁️ Purple: Deployment', shape='note')
dot.edge('A', 'I', style='dashed')

# -------------------------
# Render diagram
# -------------------------
dot.render('infographic_defect_pipeline_row', view=True)
