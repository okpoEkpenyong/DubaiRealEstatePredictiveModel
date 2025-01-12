18_dubai_real_estate_prediction
├── .env
├── .gitignore
├── requirements.txt
├── utils
│   ├── data_loader.py            # Script for dataset loading and preprocessing
│   └── other_utility_files.py
├── models
│   ├── checkpoints               # Model checkpoints for intermediate saves
│   └── final                     # Finalized trained models
├── notebooks
│   ├── 01_data_loading.ipynb
│   ├── 02_data_processing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_mode_evaluate_tune.ipynb
│   └── 06_models_interpration_lime_shap.ipynb
├── config
│   └── config.yaml               # Configuration file for paths and parameters
├── datasets
│   ├── processed
│   │   ├── test
│   │   └── train
│   └── raw
│       ├── Consumer Price Index
│       ├── Current Strength
│       ├── Population
│       ├── Rents & Transportation
│       ├── Tourism
│       └── World Development Indicators
├── docs
│   ├── reports
│   │   ├── figures
│   │   ├── prediction
│   │   └── final_reports.md
│   ├── api_reference.md
│   └── workflow.md


# High-Level Architecture

Below is the LaTeX representation of the architecture used for the Dubai Real Estate Prediction project:

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows}

\tikzstyle{startstop} = [rectangle, rounded corners, minimum width=3cm, minimum height=1cm,text centered, draw=black, fill=red!30]
\tikzstyle{process} = [rectangle, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=blue!30]
\tikzstyle{decision} = [diamond, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=green!30]
\tikzstyle{arrow} = [thick,->,>=stealth]

\begin{document}
\begin{tikzpicture}[node distance=2cm]

\node (start) [startstop] {Start};
\node (load) [process, below of=start] {Load Raw Datasets};
\node (clean) [process, below of=load] {Clean and Validate Data};
\node (integrate) [process, below of=clean] {Integrate Macroeconomic Factors};
\node (eda) [process, right of=clean, xshift=5cm] {Exploratory Data Analysis};
\node (model) [process, below of=integrate] {Train Machine Learning Models};
\node (evaluate) [process, below of=model] {Evaluate Model Performance};
\node (recommend) [process, below of=evaluate] {Generate Strategic Recommendations};
\node (end) [startstop, below of=recommend] {End};

\draw [arrow] (start) -- (load);
\draw [arrow] (load) -- (clean);
\draw [arrow] (clean) -- (integrate);
\draw [arrow] (integrate) -- (model);
\draw [arrow] (model) -- (evaluate);
\draw [arrow] (evaluate) -- (recommend);
\draw [arrow] (recommend) -- (end);
\draw [arrow] (clean) -- (eda);
\draw [arrow] (eda) -- (model);

\end{tikzpicture}
\end{document}

