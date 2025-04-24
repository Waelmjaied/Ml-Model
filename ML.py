import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Chargement et préparation des données
df = pd.read_csv('projects.csv')

# Feature engineering: Calcul de la durée totale estimée
df['total_duration'] = df['num_tasks'] * df['avg_duration']

# Sélection des caractéristiques
features = ['num_tasks', 'avg_duration', 'team_size', 'complexity', 'total_duration']
X = df[features]
y = df['total_cost']

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Création d'un pipeline de prétraitement
scaler = StandardScaler()

# 3. Entraînement des modèles
# Modèle linéaire avec normalisation
lr_model = make_pipeline(
    scaler,
    LinearRegression()
)
lr_model.fit(X_train, y_train)

# Random Forest (ne nécessite pas de normalisation)
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)
rf_model.fit(X_train, y_train)


# 4. Évaluation
def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)

    print(f"\n{model_name} Performance:")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} €")
    # Version sécurisée avec gestion des divisions par zéro
    with np.errstate(divide='ignore', invalid='ignore'):
        errors = np.abs((y_test - y_pred) / y_test)
        errors[y_test == 0] = 0  # Gérer les cas où y_test = 0

    print(f"Erreur relative moyenne: {np.nanmean(errors) * 100:.2f}%")

    return y_pred


y_pred_lr = evaluate_model(lr_model, "Régression Linéaire")
y_pred_rf = evaluate_model(rf_model, "Random Forest")

# 5. Visualisations améliorées
plt.figure(figsize=(20, 8))

# A. Relation durée totale/coût réel
plt.subplot(1, 3, 1)
plt.scatter(X['total_duration'], y, alpha=0.6)
plt.xlabel('Durée Totale Estimée (jours)')
plt.ylabel('Coût Réel (€)')
plt.title('Relation Durée/Coût')
plt.grid(True)

# B. Comparaison des modèles
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.6, label='Random Forest')
plt.scatter(y_test, y_pred_lr, alpha=0.6, label='Régression Linéaire')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.legend()
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')
plt.title('Comparaison des Prédictions')

# C. Importance des caractéristiques
plt.subplot(1, 3, 3)
importances = pd.Series(rf_model.feature_importances_, index=features)
importances.sort_values().plot.barh(color='skyblue')
plt.title('Importance des Caractéristiques (Random Forest)')

plt.tight_layout()
plt.show()


# 6. Prédiction personnalisée avec interface utilisateur
def predict_project():
    print("\nEntrez les détails du nouveau projet:")

    num_tasks = int(input("Nombre de tâches: "))
    avg_duration = float(input("Durée moyenne par tâche (jours): "))
    team_size = int(input("Taille de l'équipe: "))
    complexity = int(input("Complexité (1-5): "))

    total_duration = num_tasks * avg_duration

    new_project = pd.DataFrame([[num_tasks, avg_duration, team_size, complexity, total_duration]],
                               columns=features)

    cost_pred = rf_model.predict(new_project)

    print(f"\nEstimation du projet:")
    print(f"- Durée totale: {total_duration:.1f} jours")
    print(f"- Coût estimé: {cost_pred[0]:.2f} DT")
    print(f"- Coût/jour: {cost_pred[0] / total_duration:.2f} DT")


# Exécuter la prédiction
predict_project()