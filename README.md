# 🏋️‍♂️ Workout Recommendation System

A machine learning–based Workout Recommendation System built on top of the **MegaGymDataset (megaGymDataset.csv)**.  
The system predicts **exercise difficulty level** and recommends **personalized workouts** based on user inputs like:

- Target body part  
- Available equipment  
- Workout level (Beginner / Intermediate / Expert)

It also includes **multi-model comparison**, **confusion matrix heatmaps**, and **training curves for XGBoost**.

---

## 📂 Project Structure

```text
Workout_Recommendation_System/
├── app.py                # Web UI for user input & workout recommendations
├── main_ml_script.py     # ML training, evaluation & recommendation logic
├── megaGymDataset.csv    # Gym exercise dataset (titles, types, body parts, equipment, levels, ratings)
└── (generated at runtime)
    ├── classification_metrics_summary.csv
    ├── regression_metrics_summary.csv
    ├── classification_accuracy_comparison.png
    ├── confusion_matrix_xgboost_heatmap.png
    ├── xgb_training_loss_curve.png
    ├── xgb_validation_accuracy_curve.png
```
## 🌐 Run the Web App (Streamlit)

To launch the workout recommendation interface, simply run:


streamlit run app.py
