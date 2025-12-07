import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV
import io
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Genomic Data Analysis Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Analysis functions
# --------------------------

def preprocess_data(X, y, test_size=0.25, random_state=42):
    """Split data into train and test sets and scale features"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_and_evaluate_models(X_train, X_test, y_train, y_test, use_rf, use_svm, use_nn,
                               rf_n_estimators=100, svm_kernel='rbf', nn_hidden_layers=(100,50),
                               random_state=42):
    """Train and evaluate selected machine learning models"""
    # Initialize models dictionary and results list
    models = {}
    results = []
    
    # Add selected models
    if use_rf:
        st.write("Training Random Forest...")
        models['Random Forest'] = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=random_state)
    
    if use_svm:
        st.write("Training Support Vector Machine...")
        models['SVM'] = SVC(kernel=svm_kernel, probability=True, random_state=random_state)
    
    if use_nn:
        st.write("Training Neural Network...")
        models['Neural Network'] = MLPClassifier(hidden_layer_sizes=nn_hidden_layers,
                                                 max_iter=500, random_state=random_state)
    
    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        st.write(f"**{name}** - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return models, results_df

def plot_model_comparison(results_df):
    """Plot comparison of model performance metrics"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Set up bar positions
    models = results_df['Model']
    x = np.arange(len(models))
    width = 0.2
    
    # Create bars for each metric
    ax.bar(x - width*1.5, results_df['Accuracy'], width, label='Accuracy', color='#3498db')
    ax.bar(x - width/2, results_df['Precision'], width, label='Precision', color='#2ecc71')
    ax.bar(x + width/2, results_df['Recall'], width, label='Recall', color='#e74c3c')
    ax.bar(x + width*1.5, results_df['F1 Score'], width, label='F1 Score', color='#f39c12')
    
    # Add labels and title
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrices(models, X_test, y_test):
    """Plot confusion matrices for all models"""
    num_models = len(models)
    
    if num_models == 0:
        return None
    
    fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 5))
    
    # Make axes iterable even when there's only one model
    if num_models == 1:
        axes = [axes]
    
    # Get unique classes
    classes = np.unique(y_test)
    if len(classes) == 2:
        class_labels = ['Healthy', 'Disease']
    else:
        class_labels = []
        for i in range(len(classes)):
            if i == 0:
                class_labels.append("Healthy")
            else:
                class_labels.append(f"Disease Type {i}")
    
    for ax, (name, model) in zip(axes, models.items()):
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicted labels', fontsize=12)
        ax.set_ylabel('True labels', fontsize=12)
        ax.set_title(f'Confusion Matrix - {name}', fontsize=14)
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
    
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance for Random Forest model"""
    if not hasattr(model, 'feature_importances_'):
        st.write("Feature importance plot is only available for Random Forest model.")
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(range(len(indices)), importances[indices], color='#9b59b6')
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
    ax.set_xlabel('Features', fontsize=14)
    ax.set_ylabel('Importance', fontsize=14)
    ax.set_title('Feature Importance (Random Forest)', fontsize=16)
    plt.tight_layout()
    return fig

def plot_roc_curves(models, X_test, y_test):
    """Plot ROC curves for all models"""
    # Check if binary classification
    if len(np.unique(y_test)) != 2:
        st.write("ROC curves are only available for binary classification problems.")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    color_index = 0
    
    for name, model in models.items():
        # Get predicted probabilities
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color=colors[color_index % len(colors)], lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
        color_index += 1
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    
    plt.tight_layout()
    return fig

def train_disease_prediction_model(X_train, y_train, disease_names, model_type='rf', random_state=42):
    """Train a disease prediction model"""
    # Create the base classifier based on user selection
    if model_type == 'rf':
        base_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model_name = "Random Forest"
    elif model_type == 'svm':
        base_model = SVC(probability=True, random_state=random_state)
        model_name = "SVM"
    elif model_type == 'nn':
        base_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state)
        model_name = "Neural Network"
    else:
        base_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model_name = "Random Forest"
    
    # Use calibration to get better probability estimates
    model = CalibratedClassifierCV(base_model, cv=5)
    
    # Train the model
    st.write(f"Training {model_name} for disease prediction...")
    model.fit(X_train, y_train)
    
    # Create disease mapping
    disease_mapping = {i: name for i, name in enumerate(disease_names)}
    
    st.write(f"Disease prediction model trained successfully!")
    
    return model, disease_mapping

def predict_disease_risk(model, X_new, disease_mapping, top_n=3):
    """Predict risk of diseases for new genomic data"""
    # Get probability predictions
    y_probs = model.predict_proba(X_new)
    
    # Create a list to store results
    results = []
    
    # For each sample
    for i, probs in enumerate(y_probs):
        # Get the disease indices and probabilities
        disease_indices = np.argsort(probs)[::-1][:min(top_n, len(probs))]
        disease_probs = probs[disease_indices]
        
        # Map indices to disease names and create results
        sample_results = [
            {
                'disease': disease_mapping.get(idx, f"Disease {idx}"),
                'risk_percentage': prob * 100
            }
            for idx, prob in zip(disease_indices, disease_probs)
        ]
        
        results.append({
            'sample_id': i+1,
            'predictions': sample_results
        })
    
    return results

def visualize_disease_risk(results):
    """Create visualizations for disease risk predictions"""
    # Create a bar chart for each sample
    charts = []
    for result in results:
        sample_id = result['sample_id']
        predictions = result['predictions']
        
        # Extract data for plotting
        diseases = [pred['disease'] for pred in predictions]
        risks = [pred['risk_percentage'] for pred in predictions]
        
        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(diseases, risks, color=plt.cm.viridis(np.linspace(0, 0.8, len(diseases))))
        
        # Add risk percentages as text labels
        for bar, risk in zip(bars, risks):
            ax.text(min(risk + 1, 95), bar.get_y() + bar.get_height()/2,
                    f"{risk:.1f}%", va='center', fontweight='bold')
        
        # Add labels and title
        ax.set_xlabel('Risk Percentage (%)', fontsize=12)
        ax.set_title(f'Predicted Disease Risk - Sample {sample_id}', fontsize=14)
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        charts.append(fig)
    
    # Create summary of results
    st.write("## Disease Risk Prediction Summary")
    
    for idx, result in enumerate(results):
        sample_id = result['sample_id']
        predictions = result['predictions']
        
        st.write(f"### Sample {sample_id}")
        
        # Display bar chart
        st.pyplot(charts[idx])
        
        # Create a markdown table
        df_results = pd.DataFrame(predictions)
        st.table(df_results)
        
        # FIXED PREDICTION LOGIC
        # Check if "Healthy" is in the predictions
        healthy_pred = next((pred for pred in predictions if pred['disease'] == 'Healthy'), None)
        disease_pred = next((pred for pred in predictions if pred['disease'] != 'Healthy'), None)
        
        if healthy_pred:
            healthy_risk = healthy_pred['risk_percentage']
            
            # If Healthy prediction is high (low disease risk)
            if healthy_risk >= 75:
                risk_level = "Very Low"
                advice = "General health maintenance recommended"
            elif healthy_risk >= 50:
                risk_level = "Low"
                advice = "Standard health precautions recommended"
            elif healthy_risk >= 25:
                risk_level = "Moderate"
                advice = "Regular screening and preventive measures advised"
            else:
                # If Healthy prediction is quite low (high disease risk)
                risk_level = "High"
                advice = "Medical consultation recommended"
        else:
            # No "Healthy" in top predictions, likely high disease risk
            top_disease = predictions[0]['disease']
            top_risk = predictions[0]['risk_percentage']
            
            if top_risk >= 75:
                risk_level = "High"
                advice = "Immediate medical consultation recommended"
            elif top_risk >= 50:
                risk_level = "Moderate"
                advice = "Regular screening and preventive measures advised"
            elif top_risk >= 25:
                risk_level = "Low"
                advice = "Standard health precautions recommended"
            else:
                risk_level = "Very Low"
                advice = "General health maintenance recommended"
        
        st.write(f'''
**Risk Level:** {risk_level}

**Recommendation:** {advice}

**Note:** These predictions are based on machine learning analysis of genomic markers and should be considered as screening tools only. Always consult healthcare professionals for proper diagnosis and treatment.
        ''')

def run_disease_prediction(df_data, test_samples=3, random_state=42, model_type='rf'):
    """Run the disease prediction workflow"""
    try:
        st.write("# Disease Risk Prediction")
        
        # Display data preview
        st.write("## Data Overview")
        st.dataframe(df_data.head())
        
        # Data preparation
        X = df_data.iloc[:, :-1].values
        y = df_data.iloc[:, -1].values
        feature_names = df_data.columns[:-1].tolist()
        
        # Get unique classes and create disease names
        unique_classes = np.unique(y)
        
        # Create appropriate disease names based on the number of classes
        if len(unique_classes) == 2:
            disease_names = ["Healthy", "Disease"]
        else:
            disease_names = []
            for i in range(len(unique_classes)):
                if i == 0:
                    disease_names.append("Healthy")
                else:
                    disease_names.append(f"Disease Type {i}")
        
        st.write(f"**Detected Disease Classes:** {len(disease_names)}")
        for i, name in enumerate(disease_names):
            st.write(f"- Class {i}: {name}")
        
        # Split data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(
            X, y, test_size=0.25, random_state=random_state
        )
        
        # Train disease prediction model
        model, disease_mapping = train_disease_prediction_model(
            X_train, y_train, disease_names, model_type, random_state
        )
        
        # Select a few test samples for prediction demonstration
        st.write("## Sample Predictions")
        st.write("Selecting a few samples to demonstrate prediction capabilities:")
        
        num_samples = min(test_samples, len(X_test))
        sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
        X_samples = X_test[sample_indices]
        
        # Make predictions
        prediction_results = predict_disease_risk(model, X_samples, disease_mapping)
        
        # Visualize results
        visualize_disease_risk(prediction_results)
        
        # Add disclaimer
        st.write('''
## Important Disclaimer

The disease risk predictions shown above are for demonstration purposes only. They are based on machine learning models trained on the provided genomic data and should not be used for actual medical diagnosis or treatment decisions.

In a real-world application:
- Models would be trained on much larger, clinically validated datasets
- Multiple biomarkers and clinical variables would be included
- Rigorous validation and regulatory approval would be required
- Interpretation by healthcare professionals would be necessary

This tool demonstrates the potential of machine learning in genomic medicine, but actual implementation requires extensive clinical validation and regulatory oversight.
        ''')
        
    except Exception as e:
        st.error(f"**Error in disease prediction:** {str(e)}")
        st.error("Please ensure your data is properly formatted and contains sufficient samples for each disease class.")

def run_genomic_analysis(df_data, use_rf=True, use_svm=True, use_nn=True,
                         test_size=0.25, random_state=42,
                         rf_n_estimators=100, svm_kernel='rbf', nn_hidden_layers=(100,50),
                         include_disease_prediction=True, disease_model='rf',
                         test_samples=3):
    """Run full genomic data analysis with all components"""
    try:
        # Display data preview
        st.write("## Data Preview")
        st.dataframe(df_data.head())
        
        # Data info
        st.write("## Data Information")
        st.write(f"**Number of Samples:** {df_data.shape[0]}")
        st.write(f"**Number of Features:** {df_data.shape[1] - 1}")
        
        # Assuming the last column is the target
        target_col = df_data.columns[-1]
        st.write(f"**Number of Classes:** {len(df_data[target_col].unique())}")
        st.write(f"**Class Distribution:**")
        st.write(df_data[target_col].value_counts())
        
        # Data preparation
        X = df_data.iloc[:, :-1].values
        y = df_data.iloc[:, -1].values
        feature_names = df_data.columns[:-1].tolist()
        
        # Check if target is binary or multi-class
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            st.warning("**Warning:** The target variable has only one class. Please ensure your data has at least two classes for classification.")
            return
        
        # Preprocess data
        st.write("## Data Preprocessing")
        st.write(f"Splitting data into {(1-test_size)*100:.0f}% training and {test_size*100:.0f}% testing sets...")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, test_size, random_state)
        st.write(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train and evaluate models
        st.write("## Model Training and Evaluation")
        models, results_df = train_and_evaluate_models(
            X_train, X_test, y_train, y_test,
            use_rf, use_svm, use_nn,
            rf_n_estimators, svm_kernel, nn_hidden_layers,
            random_state
        )
        
        # Display results
        st.write("## Model Performance")
        st.dataframe(results_df)
        
        # Create visualizations
        st.write("## Visualizations")
        
        st.write("### 1. Model Performance Comparison")
        fig_comparison = plot_model_comparison(results_df)
        st.pyplot(fig_comparison)
        
        st.write("### 2. Confusion Matrices")
        fig_cm = plot_confusion_matrices(models, X_test, y_test)
        if fig_cm:
            st.pyplot(fig_cm)
        
        if use_rf:
            st.write("### 3. Feature Importance")
            rf_model = models.get('Random Forest')
            if rf_model:
                fig_importance = plot_feature_importance(rf_model, feature_names)
                if fig_importance:
                    st.pyplot(fig_importance)
        
        # ROC curves for binary classification
        if len(np.unique(y_test)) == 2:
            st.write("### 4. ROC Curves")
            fig_roc = plot_roc_curves(models, X_test, y_test)
            if fig_roc:
                st.pyplot(fig_roc)
        
        # Summary section
        st.write("## Analysis Summary")
        best_model = results_df.loc[results_df['F1 Score'].idxmax(), 'Model']
        st.write(f"Based on F1 Score, the best performing model is: **{best_model}**")
        
        # General description
        st.write('''
### Interpretation of Results

This analysis evaluated different machine learning models on genomic data to predict the target phenotype.

**Key observations:**
- Machine learning models can identify patterns in genomic data that correlate with the target phenotype
- The feature importance plot highlights which genetic markers (SNPs) have the strongest association with the phenotype
- The confusion matrices show how well each model distinguishes between the classes

**Potential applications:**
- Identifying genetic markers associated with specific traits or diseases
- Building predictive models for personalized medicine
- Understanding the genetic basis of the target phenotype

**Next steps:**
- Consider feature selection to focus on the most important genetic markers
- Try additional machine learning algorithms or hyperparameter tuning
- Validate findings on independent datasets
        ''')
        
        # Run disease prediction if requested
        if include_disease_prediction:
            run_disease_prediction(
                df_data,
                test_samples=test_samples,
                random_state=random_state,
                model_type=disease_model
            )
        
    except Exception as e:
        st.error(f"**Error analyzing data:** {str(e)}")
        st.error("Please ensure your CSV file has the correct format. The last column should be the target variable.")

def create_sample_dataset():
    """Create a sample genomic dataset"""
    np.random.seed(42)
    
    # Generate header
    header = ["SNP_" + str(i+1) for i in range(20)] + ["phenotype"]
    
    # Generate data
    data = []
    for _ in range(100):
        # Generate random SNP values (0, 1, 2)
        snps = np.random.randint(0, 3, 20)
        
        # Calculate phenotype based on SNPs (simple model)
        snp_sum = sum(snps)
        # 0 = Healthy, 1 = Disease
        phenotype = 1 if snp_sum > 30 else 0
        
        # Add row
        data.append(np.append(snps, phenotype))
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=header)
    return df

def create_multiclass_sample():
    """Create a sample multi-class genomic dataset"""
    np.random.seed(42)
    
    # Generate header
    header = ["SNP_" + str(i+1) for i in range(20)] + ["disease_type"]
    
    # Generate data
    data = []
    for _ in range(100):
        # Generate random SNP values (0, 1, 2)
        snps = np.random.randint(0, 3, 20)
        
        # Generate disease type (0, 1, 2, 3)
        snp_sum = np.sum(snps)
        if snp_sum < 25:
            disease = 0  # Healthy
        elif snp_sum < 35:
            disease = 1  # Disease type 1
        elif snp_sum < 45:
            disease = 2  # Disease type 2
        else:
            disease = 3  # Disease type 3
        
        # Add row
        data.append(np.append(snps, disease))
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=header)
    return df

# Function to convert uploaded file to dataframe
def convert_uploaded_file(uploaded_file):
    # Read CSV file
    content = uploaded_file.getvalue().decode('utf-8')
    return pd.read_csv(StringIO(content))

# Main app structure
def main():
    st.title("🧬 Genomic Data Analysis Tool")
    st.write("Upload your genomic data and analyze it with machine learning models")
    
    # Sidebar options
    st.sidebar.title("Analysis Options")
    
    # Data options
    st.sidebar.header("Data")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ("Upload your own CSV", "Use binary sample data", "Use multi-class sample data")
    )
    
    if data_option == "Upload your own CSV":
        uploaded_file = st.sidebar.file_uploader("Upload genomic data CSV", type="csv")
        if uploaded_file is not None:
            df = convert_uploaded_file(uploaded_file)
        else:
            st.info("Please upload a CSV file or select a sample dataset.")
            st.write('''
### Expected Data Format

Your CSV file should contain:
- Multiple columns representing genomic features (e.g., SNPs)
- The last column should be the target phenotype or disease classification
- For binary classification: use 0 (healthy) and 1 (disease)
- For multi-class classification: use 0 (healthy), 1, 2, 3, etc. for different disease types

Example header:
            ''')
            
            # Add more info about the tool
            st.write('''
## About This Tool

This Genomic Data Analysis Tool provides an easy-to-use interface for analyzing genomic data using machine learning techniques. It allows you to:

1. Upload your genomic data in CSV format
2. Visualize and explore the data characteristics
3. Train multiple machine learning models on genomic data
4. Compare model performance using various metrics
5. Identify important genomic markers using feature importance analysis
6. Predict disease risk based on genomic profiles

### Disclaimer

This tool is intended for educational and research purposes only. The predictions and analyses provided should not be used for clinical diagnoses or treatment decisions without proper validation by healthcare professionals.
            ''')
            return
    elif data_option == "Use binary sample data":
        df = create_sample_dataset()
        st.sidebar.success("Using binary sample data!")
    elif data_option == "Use multi-class sample data":
        df = create_multiclass_sample()
        st.sidebar.success("Using multi-class sample data!")
    
    # Model selection
    st.sidebar.header("Models")
    use_rf = st.sidebar.checkbox("Random Forest", value=True)
    use_svm = st.sidebar.checkbox("Support Vector Machine", value=True)
    use_nn = st.sidebar.checkbox("Neural Network", value=True)
    include_disease = st.sidebar.checkbox("Disease Prediction", value=True)
    
    # Disease prediction options
    if include_disease:
        st.sidebar.header("Disease Prediction")
        disease_model = st.sidebar.radio(
            "Model:",
            options=["rf", "svm", "nn"],
            format_func=lambda x: {"rf": "Random Forest", "svm": "SVM", "nn": "Neural Network"}[x],
            index=0
        )
        
        test_samples = st.sidebar.slider("Test Samples:", 1, 10, 3)
    else:
        disease_model = "rf"
        test_samples = 3
    
    # Advanced options
    st.sidebar.header("Advanced Options")
    
    test_size = st.sidebar.slider("Test Size:", 0.1, 0.5, 0.25, 0.05)
    rf_estimators = st.sidebar.slider("RF Estimators:", 10, 500, 100, 10)
    svm_kernel = st.sidebar.selectbox("SVM Kernel:", ["rbf", "linear", "poly", "sigmoid"])
    
    nn_layers_input = st.sidebar.text_input("NN Layers (comma-separated):", "100,50")
    try:
        nn_hidden_tuple = tuple(int(x.strip()) for x in nn_layers_input.split(','))
    except:
        nn_hidden_tuple = (100, 50)
        st.sidebar.warning("Invalid format. Using default (100,50).")
    
    random_state = st.sidebar.number_input("Random State:", 0, 100, 42)
    
    # Run analysis button
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Running analysis..."):
            run_genomic_analysis(
                df,
                use_rf=use_rf,
                use_svm=use_svm,
                use_nn=use_nn,
                test_size=test_size,
                random_state=random_state,
                rf_n_estimators=rf_estimators,
                svm_kernel=svm_kernel,
                nn_hidden_layers=nn_hidden_tuple,
                include_disease_prediction=include_disease,
                disease_model=disease_model,
                test_samples=test_samples
            )

if __name__ == "__main__":
    main()
