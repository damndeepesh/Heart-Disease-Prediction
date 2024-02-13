import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from warnings import warn

# Load dataset
df = pd.read_csv("heart-disease.csv")

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Exploratory Data Analysis", "Model Training"])

if page == "Exploratory Data Analysis":
    st.title("UCI Heart Disease Dataset Analysis")
    # Display dataframe
    st.subheader("Heart Disease DataFrame")
    st.write(df)

    # Graphs/Plots
    st.subheader("Graphical Representation of Heart Disease DataFrame")

    # Bar plot for target value counts
    st.subheader("Target Value Counts")
    target_counts = df["target"].value_counts()
    st.bar_chart(target_counts, use_container_width=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("---")

    # Heart disease frequency for sex
    st.subheader("Heart Disease Frequency for Sex")
    sex_counts = pd.crosstab(df.target, df.sex)
    sex_counts.plot(kind='bar')
    plt.xlabel("Heart Disease")
    plt.ylabel("Count")
    plt.title("Heart Disease Frequency for Sex")
    st.pyplot()
    plt.close()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("---")

    # Scatter plot for age vs max heart rate
    st.subheader("Age vs Max Heart Rate by Disease")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='age', y='thalach', hue='target', palette='coolwarm', ax=ax)
    ax.set_xlabel('Age')
    ax.set_ylabel('Max Heart Rate')
    ax.legend(title='Disease', loc='upper right')
    st.pyplot(fig)
    plt.close()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("---")

    # Heart disease frequency per chest pain type
    st.subheader("Heart Disease Frequency per Chest Pain Type")
    cp_counts = pd.crosstab(df.cp, df.target)
    cp_counts.plot(kind='bar')
    plt.xlabel("Chest Pain Type")
    plt.ylabel("Count")
    plt.title("Heart Disease Frequency per Chest Pain Type")
    st.pyplot()
    plt.close()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("---")

    # Correlation matrix heatmap
    st.subheader("Correlation Matrix Heatmap")
    corr_matrix = df.corr()
    sns.set(font_scale=1.2)
    heatmap_fig, heatmap_ax = plt.subplots(figsize=(15,10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=heatmap_ax)
    heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), rotation=45)
    heatmap_ax.set_yticklabels(heatmap_ax.get_yticklabels(), rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.title("Correlation Matrix Heatmap")
    st.pyplot(heatmap_fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.close()



elif page == "Model Training":
    st.title("Model Training")

    # Model selection
    selected_models = st.multiselect("Select Model", ["Random Forest", "Logistic Regression", "K Nearest Neighbors"])

    # Feature selection
    st.subheader("Feature Selection")
    selected_features = {}
    for feature in df.columns[:-1]:  # Exclude target column
        value = st.number_input(f"Enter value for {feature}", value=0)
        selected_features[feature] = value

    # Data preprocessing
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training and evaluation
    for model_name in selected_models:
        st.subheader(model_name)

        if model_name == "Random Forest":
            # Hyperparameter tuning
            rf_grid = {"n_estimators": np.arange(10, 100, 10),
                       "max_depth": [None, 3, 5, 10],
                       "min_samples_split": np.arange(2, 10, 2),
                       "min_samples_leaf": np.arange(1, 5, 1)}

            rf = RandomForestClassifier()
            rf_rs = RandomizedSearchCV(rf, rf_grid, cv=5, n_iter=20, verbose=0)
            rf_rs.fit(X_train, y_train)

            # Model evaluation
            st.write("Best Parameters:", rf_rs.best_params_)
            st.write("Test Score:", rf_rs.score(X_test, y_test))

            # Cross-validation metrics
            cv_recall = np.mean(cross_val_score(rf_rs, X, y, cv=5, scoring='recall'))
            cv_precision = np.mean(cross_val_score(rf_rs, X, y, cv=5, scoring='precision'))
            cv_accuracy = np.mean(cross_val_score(rf_rs, X, y, cv=5, scoring='accuracy'))
            cv_f1 = np.mean(cross_val_score(rf_rs, X, y, cv=5, scoring='f1'))
            st.write("Cross-Validation Recall:", cv_recall)
            st.write("Cross-Validation Precision:", cv_precision)
            st.write("Cross-Validation Accuracy:", cv_accuracy)
            st.write("Cross-Validation F1-score:", cv_f1)

            # Predict if the person has heart disease or not
            prediction = rf_rs.predict([list(selected_features.values())])
            if prediction[0] == 1:
                st.write("Prediction: The person has heart disease.")
            else:
                st.write("Prediction: The person does not have heart disease.")

        elif model_name == "Logistic Regression":
            log_reg_grid = {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}
            log_reg = LogisticRegression()
            log_reg_rs = RandomizedSearchCV(log_reg, log_reg_grid, cv=5, n_iter=20, verbose=0)
            log_reg_rs.fit(X_train, y_train)

            # Model evaluation
            st.write("Best Parameters:", log_reg_rs.best_params_)
            st.write("Test Score:", log_reg_rs.score(X_test, y_test))

            # Cross-validation metrics
            cv_recall = np.mean(cross_val_score(log_reg_rs, X, y, cv=5, scoring='recall'))
            cv_precision = np.mean(cross_val_score(log_reg_rs, X, y, cv=5, scoring='precision'))
            cv_accuracy = np.mean(cross_val_score(log_reg_rs, X, y, cv=5, scoring='accuracy'))
            cv_f1 = np.mean(cross_val_score(log_reg_rs, X, y, cv=5, scoring='f1'))
            st.write("Cross-Validation Recall:", cv_recall)
            st.write("Cross-Validation Precision:", cv_precision)
            st.write("Cross-Validation Accuracy:", cv_accuracy)
            st.write("Cross-Validation F1-score:", cv_f1)
            st.divider()

            # Predict if the person has heart disease or not
            prediction = log_reg_rs.predict([list(selected_features.values())])
            if prediction[0] == 1:
                st.write("Prediction: The person has heart disease.")
            else:
                st.write("Prediction: The person does not have heart disease.")

        elif model_name == "K Nearest Neighbors":
            knn = KNeighborsClassifier()
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            # Hyperparameter tuning for KNN
            st.subheader("Hyperparameter Tuning for K Nearest Neighbors")
            # neighbors = st.slider("Select number of neighbors", min_value=1, max_value=20, step=1)
            # knn = KNeighborsClassifier(n_neighbors=neighbors)
            knn.fit(X_train, y_train)
            train_score = knn.score(X_train, y_train)
            test_score = knn.score(X_test, y_test)

            # Plotting training and test scores for different values of K
            # st.subheader("Training and Test Scores for Different Values of K")
            neighbors_range = range(1, 21)
            train_scores = []
            test_scores = []
            for k in neighbors_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                train_scores.append(knn.score(X_train, y_train))
                test_scores.append(knn.score(X_test, y_test))

            plt.plot(neighbors_range, train_scores, label="Train Score")
            plt.plot(neighbors_range, test_scores, label="Test Score")
            plt.xlabel("Number of Neighbors (K)")
            plt.ylabel("Model Score")
            plt.title("Training and Test Scores for Different Values of K")
            plt.legend()
            st.pyplot()
            plt.close()
            
            # Model evaluation
            st.write("Test Score:", knn.score(X_test, y_test))

            # Cross-validation metrics
            cv_recall = np.mean(cross_val_score(knn, X, y, cv=5, scoring='recall'))
            cv_precision = np.mean(cross_val_score(knn, X, y, cv=5, scoring='precision'))
            cv_accuracy = np.mean(cross_val_score(knn, X, y, cv=5, scoring='accuracy'))
            cv_f1 = np.mean(cross_val_score(knn, X, y, cv=5, scoring='f1'))
            st.write("Cross-Validation Recall:", cv_recall)
            st.write("Cross-Validation Precision:", cv_precision)
            st.write("Cross-Validation Accuracy:", cv_accuracy)
            st.write("Cross-Validation F1-score:", cv_f1)

            # Predict if the person has heart disease or not
            prediction = knn.predict([list(selected_features.values())])
            if prediction[0] == 1:
                st.write("Prediction: The person has heart disease.")
            else:
                st.write("Prediction: The person does not have heart disease.")