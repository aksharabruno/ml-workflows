STAGE_MAP = {

    # ── 1. Library Loading ──────────────────────────────────────────────────

    # ── 2. Data Loading ─────────────────────────────────────────────────────
    "pandas.io.api.read_csv": "data_loading",
    "pandas.io.api.read_excel": "data_loading",
    "keras.src.datasets.mnist.load_data": "data_loading",

    # ── 3. Data Preparation ─────────────────────────────────────────────────
    "pandas.core.frame.DataFrame.fillna": "data_preparation",
    "pandas.core.frame.DataFrame.dropna": "data_preparation",
    "pandas.core.frame.DataFrame.drop": "data_preparation",
    "lightgbm.Dataset": "data_preparation",
    "numpy.lib._shape_base_impl.expand_dims": "data_preparation",  
    
    # ── 4. Exploratory Data Analysis ────────────────────────────────────────
    "seaborn.categorical.boxplot": "exploratory_data_analysis",
    "seaborn.categorical.countplot": "exploratory_data_analysis",
    "seaborn.axisgrid.pairplot": "exploratory_data_analysis",
    "seaborn.distributions.distplot": "exploratory_data_analysis",
    "seaborn.distributions.kdeplot": "exploratory_data_analysis",
    "seaborn.matrix.heatmap": "exploratory_data_analysis",
    "matplotlib.pyplot.show": "exploratory_data_analysis",
    "matplotlib.pyplot.subplots": "exploratory_data_analysis",
    "matplotlib.pyplot.figure": "exploratory_data_analysis",

    # ── 5. Data Cleaning ────────────────────────────────────────────────────

    # ── 6. Feature Engineering ──────────────────────────────────────────────

    # ── 7. Feature Transformation ───────────────────────────────────────────
    "sklearn.base.TransformerMixin.fit_transform": "feature_transformation",
    "sklearn.preprocessing._data.StandardScaler": "feature_transformation",
    "sklearn.preprocessing._data.StandardScaler.transform": "feature_transformation",
    "sklearn.preprocessing._data.RobustScaler": "feature_transformation",
    "sklearn.preprocessing._data.RobustScaler.transform": "feature_transformation",
    "keras.src.utils.numerical_utils.to_categorical": "feature_transformation",  

    # ── 8. Feature Selection ────────────────────────────────────────────────

    # ── 9. Model Building ───────────────────────────────────────────────────
    "sklearn.ensemble._forest.RandomForestClassifier": "model_building",
    "xgboost.sklearn.XGBClassifier": "model_building",
    "lightgbm.train": "model_building",
    "keras.src.models.sequential.Sequential": "model_building",
    "keras.src.layers.core.input_layer.Input": "model_building",
    "keras.src.layers.convolutional.conv2d.Conv2D": "model_building",
    "keras.src.layers.pooling.max_pooling2d.MaxPooling2D": "model_building",
    "keras.src.layers.reshaping.flatten.Flatten": "model_building",
    "keras.src.layers.regularization.dropout.Dropout": "model_building",
    "keras.src.layers.core.dense.Dense": "model_building",
    "keras.src.trainers.trainer.Trainer.compile": "model_building",
    

    # ── 10. Train-Test Splitting ────────────────────────────────────────────
    "sklearn.model_selection._split.train_test_split": "train_test_splitting",

    # ── 11. Model Training ──────────────────────────────────────────────────
    "sklearn.ensemble._forest.BaseForest.fit": "model_training",
    "xgboost.sklearn.XGBClassifier.fit": "model_training",
    "lightgbm.train": "model_training", 
    "keras.src.backend.tensorflow.trainer.TensorFlowTrainer.fit": "model_training",


    # ── 12. Model Parameter Tuning ──────────────────────────────────────────

    # ── 13. Model Validation / Evaluation ─────────────────────────────────
    "sklearn.ensemble._forest.ForestClassifier.predict": "model_validation",
    "sklearn.metrics._classification.accuracy_score": "model_validation",
    "sklearn.metrics._classification.classification_report": "model_validation",
    "sklearn.metrics._classification.confusion_matrix": "model_validation",
    "sklearn.metrics._regression.mean_squared_error": "model_validation",
    "xgboost.sklearn.XGBClassifier.predict": "model_validation",
    "keras.src.backend.tensorflow.trainer.TensorFlowTrainer.evaluate": "model_validation",
    

}