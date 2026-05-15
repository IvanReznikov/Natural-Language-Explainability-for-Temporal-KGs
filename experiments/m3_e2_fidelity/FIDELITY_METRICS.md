# M3-E2: Explanation Fidelity Metrics (8 weeks)

### Objective
Develop metrics measuring explanation fidelity across temporal relationship types.

### Notes on Implementation
- The code implementation provides **automatic proxy metrics** suitable for large-scale scoring.
- Metrics that require **user studies** or **expert validation** are tracked but returned as `null` by the scorer.

### Notation
- Let $G$ be the gold record with gold facts/graph context.
- Let $y$ be the generated explanation text.
- Let $\mathcal{D}_G$ be the set of gold temporal anchors (dates/interval endpoints) extracted from $G$.
- Let $\mathcal{A}_G$ be the set of gold “fact atoms” (entity strings, relation strings, and anchor years) extracted from $G$.
- Let $\mathcal{Y}(y)$ be the set of years mentioned in $y$.

---

#### M3-E2a: Point-in-Time Fidelity Metrics (2 weeks)
**Metrics Defined:**
1. **Timestamp Accuracy**
	- Automatic proxy: for best matching predicted date $\hat{d}$ vs gold date $d$:
	  - $1.0$ if $|\hat{d}-d| \le 1$ day
	  - $0.8$ if $|\hat{d}-d| \le 7$ days
	  - $0.0$ otherwise
2. **Context Relevance**
	- Automatic proxy: $\frac{|\{a \in \mathcal{A}_G: a \subset y\}|}{|\mathcal{A}_G|}$ (string containment over extracted “fact atoms”).
3. **Unnecessary Detail Ratio**
	- Automatic proxy: $\text{UDR} = \frac{|\mathcal{Y}(y) \setminus \mathcal{Y}(G)|}{|\mathcal{Y}(y)|}$, where $\mathcal{Y}(G)$ are years present in gold anchors.
	- Inverse score: $\text{UDS} = 1 - \text{UDR}$.
4. **Ambiguity Resolution**
	- Human metric (user study), not computed automatically.

**Test Cases:** 100 point-in-time explanations (20 from each domain)

**Deliverables:**
- Fidelity metrics specification (mathematical definitions)
- Scoring algorithm (implementation)
- Validation results (100 explanations scored)

---

#### M3-E2b: Interval Fidelity Metrics (2 weeks)
**Metrics Defined:**
1. **Boundary Accuracy**
	- Automatic proxy: average of start/end tier accuracy (same tiers as point-in-time) using predicted interval endpoints.
2. **Duration Correctness**
	- Automatic proxy: compare gold duration (in days) to the closest predicted duration (from endpoint difference or explicit duration mentions), scored with the same tier thresholds.
3. **Overlap Representation**
	- Automatic proxy: presence of overlap markers in $y$ (e.g., “while”, “simultaneously”).
	- Full correctness may require annotation if overlaps are subtle.
4. **Interval Comparison Clarity**
	- Automatic proxy: presence of explicit comparison markers (e.g., “compared”, “longer”, “shorter”, “whereas”).

**Test Cases:** 100 interval explanations

**Deliverables:**
- Fidelity metrics (interval-specific)
- Boundary validation algorithm
- Validation results

---

#### M3-E2c: Sequence Fidelity Metrics (2 weeks)
**Metrics Defined:**
1. **Ordering Accuracy**
	- Automatic proxy: pairwise ordering accuracy based on first mention indices of gold-ordered events in $y$.
2. **Step Completeness**
	- Automatic proxy: recall of gold-ordered event mentions in $y$.
3. **Causal Coherence**
	- Automatic proxy: presence of causal markers.
4. **Narrative Consistency**
	- Human metric (or future work using contradiction detection), not computed automatically.

**Test Cases:** 100 sequence explanations

**Deliverables:**
- Fidelity metrics (sequence-specific)
- Ordering validation algorithm
- Validation results

---

#### M3-E2d: Causality Fidelity Metrics (2 weeks)
**Metrics Defined:**
1. **Causal Link Accuracy**
	- Expert metric, not computed automatically.
2. **Temporal Constraint Correctness**
	- Automatic proxy: same best-match timestamp tier score against gold anchors (where present).
3. **Alternative Cause Awareness**
	- Automatic proxy: presence of alternative-cause markers (“alternative”, “also”, “other”).
4. **Confidence Calibration**
	- Human/expert metric; code emits a proxy “confidence signal” based on hedging vs certainty markers.

**Test Cases:** 100 causal explanations

**Deliverables:**
- Fidelity metrics (causality-specific)
- Causal validation protocol
- Validation results