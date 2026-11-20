# Model artifact

This GitHub-ready package intentionally excludes the large serialized model to keep the download lightweight.

Running `Project1_House_Price_Prediction_Code_Only.ipynb` in Colab trains Linear Regression and Random Forest and exports the selected trained pipeline to the results archive. You can add the `.joblib` artifact later if required.

Only load `.joblib` files you trust; pickle-based deserialization can execute arbitrary code.
