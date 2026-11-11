"""
=============================================================================
DNA Sequence Classification & Genome Pattern Predictor
=============================================================================
A Streamlit web application for computational biology that classifies DNA
sequences into groups using machine learning (Decision Tree / Random Forest).

Assignment: "Predict the Genome Pattern: AI/ML-based classification of DNA sequences"
Course: Introduction to Computational Biology
SDG Alignment: SDG 3 (Good Health & Well-Being), SDG 9 (Industry, Innovation & Infrastructure)

Required packages:
    pip install streamlit pandas numpy scikit-learn plotly
=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
import re

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="DNA Sequence Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for polish ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .group-a { background-color: #d4edda; color: #155724; border: 2px solid #28a745; }
    .group-b { background-color: #cce5ff; color: #004085; border: 2px solid #007bff; }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 1: APP TITLE & OVERVIEW
# =============================================================================
st.markdown("<h1 class='main-header'>🧬 DNA Sequence Classification &amp; Genome Pattern Predictor</h1>",
            unsafe_allow_html=True)

st.markdown("""
> **Course:** Introduction to Computational Biology  
> **Assignment:** Predict the Genome Pattern — AI/ML-based classification of DNA sequences  
> **SDG Alignment:** SDG 3 (Good Health & Well-Being) · SDG 9 (Industry, Innovation & Infrastructure)
""")

with st.expander("📖 About This Application", expanded=False):
    st.markdown("""
    This interactive tool demonstrates how **machine learning** can be applied to
    **genomic data** for sequence classification. It aligns with:

    - **SDG 3 — Good Health & Well-Being:** Understanding DNA patterns is
      fundamental to diagnostics, drug discovery, and personalised medicine.
    - **SDG 9 — Industry, Innovation & Infrastructure:** AI/ML-driven
      bioinformatics tools represent cutting-edge innovation in life-science
      infrastructure.

    **Workflow:**
    1. View or modify a labelled dataset of DNA sequences (Group A / Group B).
    2. The app automatically extracts six numerical features from each sequence.
    3. A classifier (Decision Tree or Random Forest) is trained on these features.
    4. Enter any unknown DNA sequence to predict its group with confidence scores.
    """)

st.divider()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_sequence(seq: str) -> str:
    """
    Clean and validate a DNA sequence string.
    Removes whitespace, converts to uppercase, and checks for invalid characters.
    Returns the cleaned sequence or raises ValueError.
    """
    cleaned = re.sub(r"\s+", "", seq).upper()
    invalid = set(cleaned) - {"A", "T", "G", "C"}
    if invalid:
        raise ValueError(
            f"Invalid character(s) in sequence: {', '.join(sorted(invalid))}. "
            "Only A, T, G, C are allowed."
        )
    if len(cleaned) == 0:
        raise ValueError("Sequence cannot be empty.")
    return cleaned


def extract_features(sequence: str) -> dict:
    """
    Extract the six required features from a DNA sequence.

    Features:
        1. Count of 'A'
        2. Count of 'T'
        3. Count of 'G'
        4. Count of 'C'
        5. Sequence Length
        6. GC Content (%) — percentage of G and C bases in the sequence
    """
    seq = sequence.upper()
    count_a = seq.count("A")
    count_t = seq.count("T")
    count_g = seq.count("G")
    count_c = seq.count("C")
    length = len(seq)
    gc_content = ((count_g + count_c) / length * 100) if length > 0 else 0.0

    return {
        "Count_A": count_a,
        "Count_T": count_t,
        "Count_G": count_g,
        "Count_C": count_c,
        "Seq_Length": length,
        "GC_Content_%": round(gc_content, 2),
    }


def build_feature_dataframe(sequences: dict, labels: dict) -> pd.DataFrame:
    """
    Given a dict of {id: sequence} and {id: group_label}, return a DataFrame
    with extracted features and group labels for every sequence.
    """
    rows = []
    for sid, seq in sequences.items():
        feats = extract_features(seq)
        feats["Sequence_ID"] = sid
        feats["Sequence"] = seq
        feats["Group"] = labels.get(sid, "Unknown")
        rows.append(feats)

    df = pd.DataFrame(rows)
    # Reorder columns for readability
    col_order = ["Sequence_ID", "Sequence", "Count_A", "Count_T",
                 "Count_G", "Count_C", "Seq_Length", "GC_Content_%", "Group"]
    return df[[c for c in col_order if c in df.columns]]


# =============================================================================
# SECTION 2: DATASET & INPUT MANAGEMENT
# =============================================================================
st.header("📂 1. Dataset & Input Management")

# ── Default sample dataset ───────────────────────────────────────────────────
DEFAULT_SEQUENCES = {
    "S1": "ATGCGATC",
    "S2": "ATGCGTAC",
    "S3": "AATTGCAT",
    "S4": "ATGCAATG",
    "S5": "AATGCATG",
    "S6": "GCGCGTAT",
    "S7": "GCGCGCGC",
    "S8": "GCCGGCGC",
    "S9": "GCGCTACG",
    "S10": "CCGCGCGG",
}

DEFAULT_LABELS = {
    "S1": "Group A",
    "S2": "Group A",
    "S3": "Group A",
    "S4": "Group A",
    "S5": "Group A",
    "S6": "Group B",
    "S7": "Group B",
    "S8": "Group B",
    "S9": "Group B",
    "S10": "Group B",
}

# Initialise session state on first run
if "sequences" not in st.session_state:
    st.session_state.sequences = dict(DEFAULT_SEQUENCES)
    st.session_state.labels = dict(DEFAULT_LABELS)

col_data1, col_data2 = st.columns([3, 2])

with col_data1:
    st.subheader("Current Dataset")
    raw_df = pd.DataFrame([
        {"Sequence_ID": sid, "Sequence": seq, "Group": st.session_state.labels[sid]}
        for sid, seq in st.session_state.sequences.items()
    ])
    st.dataframe(raw_df, use_container_width=True, hide_index=True)

with col_data2:
    st.subheader("Add / Modify a Sequence")
    with st.form("add_sequence_form", clear_on_submit=True):
        new_id = st.text_input("Sequence ID (e.g., S11)", placeholder="S11")
        new_seq = st.text_input("DNA Sequence", placeholder="ATGCGATC")
        new_group = st.selectbox("Group Label", ["Group A", "Group B"])
        submitted = st.form_submit_button("➕ Add / Update Sequence")

        if submitted:
            if not new_id.strip():
                st.error("Please provide a Sequence ID.")
            else:
                try:
                    cleaned = validate_sequence(new_seq)
                    st.session_state.sequences[new_id.strip()] = cleaned
                    st.session_state.labels[new_id.strip()] = new_group
                    st.success(f"Sequence **{new_id.strip()}** added/updated successfully!")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    # Reset button
    if st.button("🔄 Reset to Default Dataset"):
        st.session_state.sequences = dict(DEFAULT_SEQUENCES)
        st.session_state.labels = dict(DEFAULT_LABELS)
        st.rerun()

st.divider()

# =============================================================================
# SECTION 3: AUTOMATED FEATURE EXTRACTION
# =============================================================================
st.header("🔬 2. Automated Feature Extraction")

st.markdown("""
Six numerical features are **automatically computed** for every sequence:
| # | Feature | Description |
|---|---------|-------------|
| 1 | Count of A | Number of Adenine bases |
| 2 | Count of T | Number of Thymine bases |
| 3 | Count of G | Number of Guanine bases |
| 4 | Count of C | Number of Cytosine bases |
| 5 | Sequence Length | Total number of bases |
| 6 | GC Content (%) | (G + C) / Length × 100 |
""")

# Build and display the feature table
feature_df = build_feature_dataframe(st.session_state.sequences, st.session_state.labels)

st.subheader("Computed Feature Table")
st.dataframe(
    feature_df.style.format({"GC_Content_%": "{:.2f}"}),
    use_container_width=True,
    hide_index=True,
)

# ── Quick visual: GC Content distribution by group ───────────────────────────
fig_gc = px.box(
    feature_df,
    x="Group",
    y="GC_Content_%",
    color="Group",
    points="all",
    title="GC Content Distribution by Group",
    color_discrete_map={"Group A": "#28a745", "Group B": "#007bff"},
)
fig_gc.update_layout(showlegend=False)
st.plotly_chart(fig_gc, use_container_width=True)

st.divider()

# =============================================================================
# SECTION 4: MACHINE LEARNING MODEL TRAINING
# =============================================================================
st.header("🤖 3. Machine Learning Model Training")

FEATURE_COLS = ["Count_A", "Count_T", "Count_G", "Count_C", "Seq_Length", "GC_Content_%"]

# ── Classifier selection ─────────────────────────────────────────────────────
col_ml1, col_ml2 = st.columns([1, 2])

with col_ml1:
    classifier_name = st.selectbox(
        "Select Classifier",
        ["Decision Tree", "Random Forest"],
        help="Choose a scikit-learn classifier to train on the feature data.",
    )
    random_state = st.number_input("Random State (seed)", value=42, min_value=0, step=1)

    if classifier_name == "Random Forest":
        n_estimators = st.slider("Number of Trees", min_value=10, max_value=200, value=100, step=10)

# Prepare training data
X_train = feature_df[FEATURE_COLS].values
y_train = feature_df["Group"].values

# Check that we have at least two classes
unique_classes = np.unique(y_train)
if len(unique_classes) < 2:
    st.error("⚠️ The dataset must contain sequences from **both** Group A and Group B to train a classifier.")
    st.stop()

# Instantiate the classifier
if classifier_name == "Decision Tree":
    clf = DecisionTreeClassifier(random_state=int(random_state))
else:
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=int(random_state))

# Train
clf.fit(X_train, y_train)

# Training accuracy (resubstitution)
y_pred_train = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)

# Cross-validation (leave-one-out style for small datasets, or 3-fold)
n_folds = min(5, len(X_train))
if n_folds >= 2:
    cv_scores = cross_val_score(clf, X_train, y_train, cv=n_folds, scoring="accuracy")
else:
    cv_scores = np.array([train_accuracy])

with col_ml2:
    st.subheader("Training Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Training Accuracy", f"{train_accuracy * 100:.1f}%")
    m2.metric(f"CV Accuracy ({n_folds}-fold)", f"{cv_scores.mean() * 100:.1f}%")
    m3.metric("CV Std Dev", f"±{cv_scores.std() * 100:.1f}%")

    # Classification report
    with st.expander("📊 Detailed Classification Report"):
        report = classification_report(y_train, y_pred_train, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

# ── Feature Importance ───────────────────────────────────────────────────────
st.subheader("Feature Importance")

importances = clf.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Importance": importances
}).sort_values("Importance", ascending=True)

fig_imp = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title=f"Feature Importance ({classifier_name})",
    color="Importance",
    color_continuous_scale="Viridis",
)
fig_imp.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Importance Score")
st.plotly_chart(fig_imp, use_container_width=True)

st.divider()

# =============================================================================
# SECTION 5: UNKNOWN SEQUENCE PREDICTION
# =============================================================================
st.header("🔮 4. Predict Group for an Unknown Sequence")

col_pred1, col_pred2 = st.columns([1, 1])

with col_pred1:
    unknown_seq_input = st.text_input(
        "Enter an unknown DNA sequence:",
        value="ATGCGCGATC",
        help="Type or paste a DNA sequence using only A, T, G, C characters.",
    )

    predict_clicked = st.button("🧪 Predict Group", type="primary", use_container_width=True)

with col_pred2:
    if predict_clicked:
        try:
            cleaned_unknown = validate_sequence(unknown_seq_input)
            unknown_features = extract_features(cleaned_unknown)

            st.markdown("**Extracted Features for Unknown Sequence:**")
            uf_df = pd.DataFrame([unknown_features])
            st.dataframe(uf_df, use_container_width=True, hide_index=True)

            # Predict
            X_unknown = np.array([[
                unknown_features["Count_A"],
                unknown_features["Count_T"],
                unknown_features["Count_G"],
                unknown_features["Count_C"],
                unknown_features["Seq_Length"],
                unknown_features["GC_Content_%"],
            ]])

            prediction = clf.predict(X_unknown)[0]
            probabilities = clf.predict_proba(X_unknown)[0]
            class_labels = clf.classes_

            # Display prediction with styled box
            css_class = "group-a" if prediction == "Group A" else "group-b"
            emoji = "🟢" if prediction == "Group A" else "🔵"
            st.markdown(
                f"<div class='prediction-box {css_class}'>"
                f"{emoji} Predicted: <strong>{prediction}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Confidence / probability scores
            st.markdown("**Confidence Scores:**")
            prob_df = pd.DataFrame({
                "Group": class_labels,
                "Probability": probabilities,
            })
            st.dataframe(
                prob_df.style.format({"Probability": "{:.4f}"}).bar(
                    subset=["Probability"], color="#4CAF50"
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Probability gauge chart
            fig_prob = go.Figure(go.Bar(
                x=class_labels,
                y=probabilities,
                marker_color=["#28a745" if c == "Group A" else "#007bff" for c in class_labels],
                text=[f"{p:.1%}" for p in probabilities],
                textposition="auto",
            ))
            fig_prob.update_layout(
                title="Prediction Probability",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                xaxis_title="",
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        except ValueError as e:
            st.error(f"❌ {e}")

st.divider()

# =============================================================================
# SECTION 6: CRITICAL DISCUSSION & REPORT ELEMENTS
# =============================================================================
st.header("📝 5. Critical Discussion & Report Elements")

# Tab-based layout for discussion sections
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Feature Importance",
    "🌍 Real-World Application",
    "⚠️ Limitations of AI/ML",
    "🤔 Edge Cases & Discussion",
])

with tab1:
    st.subheader("Which Features Were Most Useful and Why?")

    # Dynamic interpretation based on actual model results
    sorted_imp = importance_df.sort_values("Importance", ascending=False)
    top_feature = sorted_imp.iloc[0]["Feature"]
    top_score = sorted_imp.iloc[0]["Importance"]

    st.markdown(f"""
    Based on the trained **{classifier_name}** model, the most important feature is
    **{top_feature}** (importance score: {top_score:.4f}).

    **Interpretation:**

    - **GC Content (%)** is typically the most discriminative feature for separating
      DNA sequence groups. GC content reflects the thermal stability of a DNA
      duplex—higher GC content means more hydrogen bonds and higher melting
      temperature. Many biological classification tasks (e.g., distinguishing
      bacterial species) rely heavily on GC content.

    - **Individual base counts (A, T, G, C)** provide the raw composition profile.
      Due to Chargaff's rules, A ≈ T and G ≈ C in double-stranded DNA, so these
      counts are partially redundant but still informative for single-stranded or
      short sequence analysis.

    - **Sequence Length** is less informative when all sequences have similar
      lengths, but becomes important when groups differ in sequence length.

    The bar chart above shows the relative contribution of each feature to the
    model's classification decisions. Features with higher importance scores had
    more influence on the split decisions within the tree(s).
    """)

    st.dataframe(
        sorted_imp.reset_index(drop=True).style.format({"Importance": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Real-World Genomics Application")
    st.markdown("""
    ### 🧫 Cancer Genomics — Tumour vs. Normal Tissue Classification

    One of the most impactful real-world applications of ML-based DNA sequence
    classification is **distinguishing tumour DNA from normal tissue DNA** using
    features extracted from genomic sequences.

    **How it works:**
    1. DNA is extracted from both tumour biopsies and matched normal tissue.
    2. Sequences are profiled for features like GC content, mutation frequency,
       microsatellite instability, and methylation patterns.
    3. ML classifiers (Random Forests, SVMs, Deep Learning) are trained on labelled
       datasets to identify cancer-associated genomic signatures.
    4. New patient samples can be classified to assist in early diagnosis.

    **Impact:**
    - **Liquid biopsies** use circulating tumour DNA (ctDNA) in blood to detect
      cancer non-invasively.
    - Companies like *Grail (Galleri test)* and *Foundation Medicine* use ML on
      genomic data for early cancer detection.
    - This directly supports **SDG 3** by improving health outcomes through early
      detection and personalised treatment.

    **Connection to this assignment:**  
    The same pipeline we use here—feature extraction → classification—underpins
    these clinical-grade genomic tools, just at a much larger scale with more
    sophisticated features.
    """)

with tab3:
    st.subheader("Major Limitation of Using AI/ML in Genomic Analysis")
    st.markdown("""
    ### ⚠️ Training Data Bias & Limited Generalisability

    The most significant limitation is **training data bias and overfitting to
    narrow datasets**:

    - **Small & non-representative training sets:** Our demo uses only 10 short
      sequences. Real genomes are billions of bases long with complex structures.
      A model trained on a tiny, curated dataset may fail catastrophically on
      real-world genomic data.

    - **Population bias:** Most genomic reference databases are skewed toward
      European-ancestry populations. ML models trained on these data may produce
      inaccurate predictions for underrepresented populations, exacerbating health
      disparities (contrary to SDG 3).

    - **Feature simplicity:** Counting individual bases and GC content ignores
      crucial biological context—codon usage, regulatory motifs, epigenetic
      modifications, and three-dimensional genome architecture.

    - **Black-box models:** Complex models (deep neural networks) used in
      practice are difficult to interpret, making it hard for biologists and
      clinicians to trust and validate predictions.

    - **Distribution shift:** Genomic data from different species, tissues, or
      sequencing technologies may have very different statistical properties,
      causing models to lose accuracy outside their training distribution.

    **Takeaway:** AI/ML is a powerful tool in genomics, but it must be used with
    domain expertise, rigorous validation, and awareness of data limitations.
    """)

with tab4:
    st.subheader("What Happens if a Sequence Is Drastically Different?")
    st.markdown("""
    ### 🤔 Out-of-Distribution (OOD) Sequences

    **Scenario:** What if a user inputs a sequence that is very different from
    anything in the training set—e.g., a 1000-base sequence when all training
    sequences are 8 bases, or a sequence with 100% A content?

    **Problems that arise:**

    1. **Forced classification:** The model *must* assign either Group A or
       Group B—it has no "I don't know" option. This means it will confidently
       misclassify OOD sequences without any warning.

    2. **Misleading confidence scores:** Decision trees and random forests
       produce probability estimates based on leaf node distributions. An OOD
       sample may land in a leaf node with 100% purity (high confidence) even
       though the prediction is meaningless.

    3. **Extrapolation failure:** Tree-based models cannot extrapolate beyond
       the range of features seen during training. A sequence with `Seq_Length = 500`
       would be handled by the same leaf as the nearest training length, regardless
       of how different it actually is.

    **Mitigation strategies:**
    - Add **anomaly/novelty detection** before classification (e.g., isolation
      forests, distance thresholds).
    - Implement **confidence thresholds** — flag predictions below a certain
      probability as "uncertain".
    - Use **domain-informed validation** — check if extracted features fall
      within biologically plausible ranges.
    - Regularly **retrain** models as new labelled data becomes available.
    """)

    # Interactive demo: let user try an extreme sequence
    st.markdown("---")
    st.markdown("**🧪 Try it yourself!** Enter an extreme/unusual sequence below and see how the model responds:")
    extreme_seq = st.text_input(
        "Extreme test sequence:",
        value="AAAAAAAAAAAAAAAA",
        key="extreme_seq",
    )
    if st.button("Predict Extreme Sequence", key="extreme_predict"):
        try:
            cleaned_ext = validate_sequence(extreme_seq)
            ext_feats = extract_features(cleaned_ext)
            X_ext = np.array([[
                ext_feats["Count_A"], ext_feats["Count_T"],
                ext_feats["Count_G"], ext_feats["Count_C"],
                ext_feats["Seq_Length"], ext_feats["GC_Content_%"],
            ]])
            pred_ext = clf.predict(X_ext)[0]
            prob_ext = clf.predict_proba(X_ext)[0]

            st.info(f"**Predicted:** {pred_ext}")
            for lbl, prob in zip(clf.classes_, prob_ext):
                st.write(f"- {lbl}: {prob:.4f}")
            st.warning(
                "⚠️ Notice that the model assigns a group with seemingly high confidence, "
                "even though this sequence may be nothing like the training data. "
                "This illustrates the **OOD problem** discussed above."
            )
        except ValueError as e:
            st.error(str(e))

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: grey; padding: 1rem;'>
    🧬 <strong>DNA Sequence Classification & Genome Pattern Predictor</strong><br>
    Introduction to Computational Biology · AI/ML-based Classification<br>
    Built with Streamlit · scikit-learn · pandas · plotly
</div>
""", unsafe_allow_html=True)
