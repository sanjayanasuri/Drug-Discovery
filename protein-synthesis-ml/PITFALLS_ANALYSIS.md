# Pitfalls Analysis & Roadmap

This document analyzes the current implementation against the identified pitfalls and provides a roadmap for addressing them.

## Current Status vs. Pitfalls

### ❌ Pitfall 1: "One model per virus" does not scale
**Status: NOT SOLVED** (Partially addressed)

**What we have:**
- ✅ Config-driven multi-target system (`config/targets.yaml`)
- ✅ Easy addition of new targets via ChEMBL download
- ✅ Dynamic target loading in registry

**What's missing:**
- ❌ Protein-aware ML (Sequence → Binding model)
- ❌ ESM2/ProtBERT embeddings for protein sequences
- ❌ AlphaFold structure integration
- ❌ Graph neural networks with protein pockets
- ❌ Universal model: `(protein_sequence, ligand_smiles) → probability`

**Current limitation:** Still requires training a separate model for each target.

**Roadmap:**
1. **Phase 1: Protein Embeddings** (High Priority)
   - Integrate ESM2 or ProtBERT for protein sequence embeddings
   - Create `protein_embedder.py` module
   - Modify featurization to include protein embeddings

2. **Phase 2: Structure-Aware Models** (Medium Priority)
   - Integrate AlphaFold structures or PDB structures
   - Use graph neural networks (GNNs) for protein-ligand interaction
   - Tools: PyTorch Geometric, DGL

3. **Phase 3: Universal Binding Model** (Long-term)
   - Train single model: `f(protein_embedding, ligand_fingerprint) → binding_probability`
   - Fine-tune on target-specific data when available
   - Transfer learning from related targets

**Estimated effort:** 2-4 weeks for Phase 1, 1-2 months for full implementation

---

### ❌ Pitfall 2: Docking is slow and unreliable
**Status: NOT SOLVED** (Basic batch docking added)

**What we have:**
- ✅ Batch docking script (`batch_docking.py`)
- ✅ AutoDock Vina integration
- ✅ UI for batch docking

**What's missing:**
- ❌ ML-based docking rescoring (GNINA/DiffDock)
- ❌ Faster, more accurate docking
- ❌ GPU-accelerated docking

**Current limitation:** Still using Vina, which is slow and has limited accuracy.

**Roadmap:**
1. **Phase 1: GNINA Integration** (High Priority)
   - GNINA is faster and more accurate than Vina
   - Add `gnina_docking.py` module
   - Replace Vina calls with GNINA where available

2. **Phase 2: DiffDock Integration** (Medium Priority)
   - DiffDock from MIT/FAIR is state-of-the-art
   - Requires PyTorch and GPU
   - Add `diffdock_docking.py` module

3. **Phase 3: Hybrid Approach** (Long-term)
   - Use Vina for initial screening
   - Rescore top hits with GNINA/DiffDock
   - Ensemble docking scores

**Estimated effort:** 1 week for GNINA, 2-3 weeks for DiffDock

---

### ⚠️ Pitfall 3: ADMET models are "general-purpose" not virus-specific
**Status: PARTIALLY SOLVED**

**What we have:**
- ✅ Calibration (CalibratedClassifierCV)
- ✅ Model selection (RandomForest vs XGBoost)
- ✅ Cross-validation for model quality

**What's missing:**
- ❌ Uncertainty estimates (confidence intervals)
- ❌ Ensemble models for robustness
- ❌ Out-of-distribution detection

**Current limitation:** ADMET is target-agnostic (which is correct), but we need better uncertainty quantification.

**Roadmap:**
1. **Phase 1: Uncertainty Estimation** (High Priority)
   - Add prediction intervals for regression models
   - Calibrated probabilities with confidence bounds
   - Use ensemble variance as uncertainty proxy

2. **Phase 2: Ensemble Models** (Medium Priority)
   - Train multiple models per task
   - Average predictions with uncertainty
   - Tools: scikit-learn's VotingClassifier/Regressor

3. **Phase 3: OOD Detection** (Long-term)
   - Detect molecules far from training distribution
   - Flag low-confidence predictions
   - Use Tanimoto similarity to nearest training compound

**Estimated effort:** 1 week for uncertainty, 1-2 weeks for ensembles

---

### ⚠️ Pitfall 4: Molecule generation may drift into chemical nonsense
**Status: PARTIALLY SOLVED**

**What we have:**
- ✅ Medicinal chemistry filters (PAINS, structural alerts)
- ✅ Similarity constraints (Tanimoto similarity window)
- ✅ Lipinski's Rule of Five
- ✅ Atom/ring count limits

**What's missing:**
- ❌ Synthetic accessibility scoring (SA Score)
- ❌ Fragment-based constraints
- ❌ Retrosynthesis feasibility (ASKCOS, AIZynthFinder)
- ❌ Property-based reward functions

**Current limitation:** Filters prevent bad molecules but don't ensure synthesizability.

**Roadmap:**
1. **Phase 1: Synthetic Accessibility** (High Priority)
   - Integrate SA Score (RDKit-based)
   - Add `synthetic_accessibility.py` module
   - Filter by SA Score threshold

2. **Phase 2: Fragment Constraints** (Medium Priority)
   - Use BRICS fragments more systematically
   - Constrain mutations to known fragment libraries
   - Tools: RDKit BRICS, ChEMBL fragment libraries

3. **Phase 3: Retrosynthesis** (Long-term)
   - Integrate ASKCOS or AIZynthFinder
   - Check if generated molecules are synthesizable
   - Prioritize molecules with known synthetic routes

**Estimated effort:** 1 week for SA Score, 2-3 weeks for retrosynthesis

---

### ❌ Pitfall 5: Scaling to millions of molecules
**Status: NOT SOLVED** (CLI mode added, but still CPU-based)

**What we have:**
- ✅ CLI pipeline mode (`cli_pipeline.py`)
- ✅ Batch processing support
- ✅ Parquet output for large datasets

**What's missing:**
- ❌ GPU batching and vectorized inference
- ❌ Faiss for similarity search
- ❌ Distributed processing
- ❌ Memory-efficient streaming

**Current limitation:** CPU-based, sequential processing, limited scalability.

**Roadmap:**
1. **Phase 1: Vectorized Inference** (High Priority)
   - Batch all model predictions (already partially done)
   - Use numpy vectorization
   - Profile and optimize bottlenecks

2. **Phase 2: GPU Acceleration** (Medium Priority)
   - Move models to GPU (PyTorch/TensorFlow)
   - Batch inference on GPU
   - Tools: PyTorch, TensorFlow, JAX

3. **Phase 3: Faiss Integration** (Medium Priority)
   - Replace clustering with Faiss for similarity search
   - GPU-accelerated nearest neighbor search
   - Handle millions of molecules efficiently

4. **Phase 4: Distributed Processing** (Long-term)
   - Dask or Ray for distributed computing
   - Cloud deployment (AWS Batch, GCP)
   - Parallel docking across nodes

**Estimated effort:** 1 week for vectorization, 2-3 weeks for GPU, 1-2 weeks for Faiss

---

## Summary Table

| Pitfall | Status | Priority | Estimated Effort |
|---------|--------|----------|------------------|
| 1. One model per virus | ❌ Not Solved | High | 2-4 weeks (Phase 1) |
| 2. Docking slow/unreliable | ❌ Not Solved | High | 1-3 weeks |
| 3. ADMET uncertainty | ⚠️ Partially Solved | Medium | 1-2 weeks |
| 4. Chemical nonsense | ⚠️ Partially Solved | Medium | 1-3 weeks |
| 5. Scaling to millions | ❌ Not Solved | High | 2-4 weeks |

## Recommended Implementation Order

### Sprint 1 (2-3 weeks): Critical Scalability
1. **GPU batching and vectorized inference** (Pitfall 5)
2. **GNINA docking integration** (Pitfall 2)
3. **Uncertainty estimation for ADMET** (Pitfall 3)

### Sprint 2 (2-3 weeks): Model Quality
1. **Protein embeddings (ESM2)** (Pitfall 1 - Phase 1)
2. **Synthetic accessibility scoring** (Pitfall 4 - Phase 1)
3. **Faiss for similarity search** (Pitfall 5 - Phase 3)

### Sprint 3 (3-4 weeks): Advanced Features
1. **DiffDock integration** (Pitfall 2 - Phase 2)
2. **Structure-aware models** (Pitfall 1 - Phase 2)
3. **Retrosynthesis integration** (Pitfall 4 - Phase 3)

## Quick Wins (Can implement immediately)

1. **SA Score integration** - 1-2 days
   - Use RDKit's SA Score implementation
   - Add to `medchem_filters.py`

2. **Uncertainty from ensemble variance** - 2-3 days
   - Train multiple models, use std as uncertainty
   - Add to `train_admet_enhanced.py`

3. **Vectorized batch inference** - 1-2 days
   - Already partially done, just need to ensure all models use batching
   - Profile and optimize

4. **GNINA wrapper** - 2-3 days
   - Similar to Vina wrapper, just different command-line tool
   - Add to `docking.py`

## Conclusion

**Current state:** The improvements made address the "low-hanging fruit" and make the system more production-ready, but do NOT solve the fundamental scalability and model architecture issues.

**Next steps:** Focus on Pitfalls 1, 2, and 5 for maximum impact:
- Protein-aware ML (biggest architectural change)
- Better docking (immediate user value)
- GPU/scaling (enables real-world usage)

The system is now **better organized and more maintainable**, which makes implementing these advanced features easier.

