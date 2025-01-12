# Dubai Real Estate Prediction Project

This project aims to predict property rental and sale prices in Dubai's real estate market by analyzing transaction data and macroeconomic factors. The pipeline includes exploratory data analysis, feature engineering, machine learning modeling, and strategic recommendations.

---

## **Project Structure**

18_dubai_real_estate_prediction
├── .env                                     # Environment variables (not included in the repo)
├── .gitignore                               # Files and directories to exclude from version control
├── LICENSE                                  # License File
├── README.md                                # Project overview
├── requirements.txt                         # Project dependencies
├── configs                                  # Configuration files
│   └── config.yaml                          # Paths and settings
├── datasets                                 # Raw and processed data (excluded from the repo)
│   ├── raw                                  # Raw datasets
│   └── processed                            # Cleaned and prepared datasets
├── docs                                     # Documentation
│   ├── reports                              # Reports and analysis outputs
│   │   ├── figures
│   │   └── final_reports.md
│   ├── high_level_architecture.md
│   ├── api_reference.md
│   └── workflow.md                          # Workflow details
├── models                                   # Model artifacts and checkpoints
│   ├── checkpoints
│   └── final
├── notebooks                                # Jupyter notebooks for EDA, modeling, and analysis
│   ├── 01_data_loading.ipynb
│   ├── 02_data_processing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_model_evaluate_tune.ipynb
│   └── 06_models_interpretation_lime_shap.ipynb
├── tests
│   └── test_config.py
└── utils                                     # Utility scripts (data loading, preprocessing, etc.)
    └── (utility scripts)

---

## **Getting Started**

### **1. Clone the Repo and Set up the Virtual Environment**
git clone https://github.com/okpoEkpenyong/DubaiRealEstatePredictiveModel
cd 18_dubai_real_estate_prediction

python3 -m venv dubai_env
source dubai_env/bin/activate  # On Windows, use `dubai_env\Scripts\activate`

### **2. Configure Paths and Project Files**
Ensure the datasets are placed in the datasets/raw directory as outlined in configs/config.yaml

Update the paths in `config/config.yaml` if necessary.

### **3. Install Python Dependencies**
   pip install -r requirements.txt

### **4. Run Notebooks**
Navigate to the `notebooks/` directory and run the Jupyter notebooks in sequence.


- `01_data_loading.ipynb`
- `02_data_processing.ipynb`
- `03_feature_engineering.ipynb`
- `04_eda.ipynb`
- `05_model_evaluate_tune.ipynb`
- `06_models_interpretation_lime_shap.ipynb`

To start Jupyter, use:
   ```bash
   jupyter notebook
   ```
---

## **Key Features**
- **Exploratory Data Analysis:** Insights into Dubai's real estate trends.
- **Machine Learning Models:** Predictive models for rental and sale prices.
- **Strategic Recommendations:** Data-driven suggestions for investors.
- **Integration of Macroeconomic Factors:** Influence of GDP, population, tourism, and more.

---

### **4.Running the Tests**

Testing is crucial to ensure project integrity and correctness. Tests cover:
1. **Configuration File Validity**:
   - Validates `config.yaml` for syntax and correctness.
   - Ensures all directories and files listed in the config exist.

2. **Dataset Headers**:
   - Confirms that required columns exist in the dataset files.

### Steps to Run the Tests:

1. Install `pytest`:
   ```bash
   pip install pytest
   ```

2. Run the test suite:
   ```bash
   pytest tests/
   ```

3. Output:
   - If all tests pass, you'll see a success message:
     ```plaintext
     ================== test session starts ==================
     platform linux -- Python 3.x, pytest-7.x.x
     collected X items

     tests/test_config.py .......                               [100%]
     =================== X passed in 0.XXs ====================
     ```
   - If tests fail, details about the failures will be displayed.

### Debugging Tests
If a test fails, check the debug messages in the test logs to identify missing files, incorrect paths, or column issues.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing

We welcome contributions! Please fork the repository and submit a pull request for review.

---

## Contact

For questions, reach out to [okpo.ekpenyong@gmail.com].

---

