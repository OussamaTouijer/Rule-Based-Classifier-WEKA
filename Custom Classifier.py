"""
Classificateur Intelligent par Règles avec Visualisation Avancée
"""

# -----------------------------------------------------------------------------
# LIBRAIRIES
# -----------------------------------------------------------------------------
import csv
import random
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
plt.style.use('ggplot')
sns.set_palette("pastel")
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


RULES = [
    # Format : {conditions: {attributs}, class: 'high'}

    {
        'conditions': {'bread and cake': 't', 'frozen foods': 't', 'pet foods': 't', 'tissues-paper prd': 't', 'cheese': 't'},
        'class': 'high'
    },
    {
        'conditions': {'bread and cake': 't', 'biscuits': 't', 'frozen foods': 't', 'tissues-paper prd': 't', 'cheese': 't', 'margarine': 't'},
        'class': 'high'
    },
    {
        'conditions': {'bread and cake': 't', 'biscuits': 't', 'frozen foods': 't', 'pet foods': 't', 'tissues-paper prd': 't', 'fruit': 't'},
        'class': 'high'
    },
    {
        'conditions': {'bread and cake': 't', 'frozen foods': 't', 'pet foods': 't', 'tissues-paper prd': 't', 'margarine': 't', 'vegetables': 't'},
        'class': 'high'
    },
    {
        'conditions': {'biscuits': 't', 'frozen foods': 't', 'pet foods': 't', 'tissues-paper prd': 't', 'margarine': 't'},
        'class': 'high'
    },
    {
        'conditions': {'biscuits': 't', 'frozen foods': 't', 'laundry needs': 't', 'tissues-paper prd': 't', 'margarine': 't'},
        'class': 'high'
    },
    {
        'conditions': {'bread and cake': 't', 'biscuits': 't', 'frozen foods': 't', 'pet foods': 't', 'margarine': 't', 'fruit': 't'},
        'class': 'high'
    },
    {
        'conditions': {'biscuits': 't', 'frozen foods': 't', 'tissues-paper prd': 't', 'cheese': 't', 'fruit': 't', 'vegetables': 't'},
        'class': 'high'
    },
    {
        'conditions': {'biscuits': 't', 'frozen foods': 't', 'laundry needs': 't', 'tissues-paper prd': 't', 'vegetables': 't'},
        'class': 'high'
    },
    {
        'conditions': {'biscuits': 't', 'breakfast food': 't', 'tissues-paper prd': 't', 'margarine': 't', 'vegetables': 't'},
        'class': 'high'
    },
    {
        'conditions': {'frozen foods': 't', 'pet foods': 't', 'laundry needs': 't', 'tissues-paper prd': 't'},
        'class': 'high'
    },
    {
        'conditions': {'biscuits': 't', 'breakfast food': 't', 'frozen foods': 't', 'pet foods': 't', 'vegetables': 't'},
        'class': 'high'
    },
    {
        'conditions': {'frozen foods': 't', 'laundry needs': 't', 'tissues-paper prd': 't', 'margarine': 't', 'vegetables': 't'},
        'class': 'high'
    },
    {
        'conditions': {'pet foods': 't', 'laundry needs': 't', 'tissues-paper prd': 't', 'margarine': 't'},
        'class': 'high'
    },
    {
        'conditions': {'bread and cake': 't', 'biscuits': 't', 'frozen foods': 't', 'tissues-paper prd': 't', 'cheese': 't', 'fruit': 't'},
        'class': 'high'
    },
    {
        'conditions': {'bread and cake': 't', 'biscuits': 't', 'frozen foods': 't', 'tissues-paper prd': 't', 'cheese': 't', 'vegetables': 't'},
        'class': 'high'
    },
    {
        'conditions': {'biscuits': 't', 'frozen foods': 't', 'pet foods': 't', 'tissues-paper prd': 't', 'fruit': 't', 'vegetables': 't'},
        'class': 'high'
    },
    {
        'conditions': {'frozen foods': 't', 'laundry needs': 't', 'tissues-paper prd': 't', 'cheese': 't'},
        'class': 'high'
    },
    {
        'conditions': {'bread and cake': 't', 'biscuits': 't', 'frozen foods': 't', 'pet foods': 't', 'tissues-paper prd': 't', 'vegetables': 't'},
        'class': 'high'
    },
    {
        'conditions': {'bread and cake': 't', 'biscuits': 't', 'frozen foods': 't', 'tissues-paper prd': 't', 'dairy foods': 't'},
        'class': 'high'
    },
    {
        'conditions': {'bread and cake': 't', 'pet foods': 't', 'tissues-paper prd': 't', 'cheese': 't', 'vegetables': 't'},
        'class': 'high'
    }
]


# -----------------------------------------------------------------------------
# FONCTIONS CORE
# -----------------------------------------------------------------------------
def load_dataset(file_path: str) -> List[Dict[str, str]]:
    """Charge et nettoie le dataset"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [{k.strip(" '\""): v.strip() for k, v in row.items()}
                    for row in csv.DictReader(f)]
    except FileNotFoundError:
        print(f"Erreur: Fichier '{file_path}' introuvable!")
        exit()

class RuleClassifier:
    """Classificateur à règles avec suivi d'utilisation"""
    def __init__(self, rules: List[Dict]):
        self.rules = rules
        self.rule_usage = {str(r['conditions']): 0 for r in rules}
    
    def predict(self, instance: Dict) -> str:
        """Effectue une prédiction avec suivi des règles"""
        for rule in self.rules:
            if all(instance.get(k) == v for k, v in rule['conditions'].items()):
                self.rule_usage[str(rule['conditions'])] += 1
                return rule['class']
        return 'low'

def split_data(data: List[Dict], test_size: float = 0.3) -> Tuple[List, List]:
    """Divise les données de manière aléatoire"""
    shuffled = data.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_size))
    return shuffled[:split_idx], shuffled[split_idx:]

# -----------------------------------------------------------------------------
# ÉVALUATION
# -----------------------------------------------------------------------------
def calculate_metrics(predictions: List[str], actuals: List[str]) -> Dict:
    """Calcule les métriques de performance"""
    cm = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    
    for pred, true in zip(predictions, actuals):
        if pred == 'high' and true == 'high': cm['TP'] += 1
        elif pred == 'high' and true == 'low': cm['FP'] += 1
        elif pred == 'low' and true == 'low': cm['TN'] += 1
        else: cm['FN'] += 1

    total = len(predictions)
    precision = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 0
    recall = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0
    
    return {
        'accuracy': (cm['TP'] + cm['TN']) / total * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': 2 * (precision * recall) / (precision + recall) * 100 if (precision + recall) > 0 else 0,
        'confusion_matrix': cm,
        'total': total
    }

def cross_validate(data: List[Dict], classifier: RuleClassifier, k: int = 10) -> Dict:
    """Validation croisée k-fold"""
    data_copy = data.copy()
    random.shuffle(data_copy)
    fold_size = len(data_copy) // k
    all_preds, all_actuals = [], []
    
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else len(data_copy)
        test_set = data_copy[test_start:test_end]
        
        preds = [classifier.predict(x) for x in test_set]
        actuals = [x['total'] for x in test_set]
        
        all_preds.extend(preds)
        all_actuals.extend(actuals)
    
    return calculate_metrics(all_preds, all_actuals)

# -----------------------------------------------------------------------------
# VISUALISATION
# -----------------------------------------------------------------------------
def visualize_performance(metrics: Dict, rule_usage: Dict):
    """Génère un dashboard de visualisation"""
    plt.figure(figsize=(18, 12))
    
    # Matrice de Confusion
    plt.subplot(2, 2, 1)
    cm = metrics['confusion_matrix']
    sns.heatmap([[cm['TP'], cm['FN']], [cm['FP'], cm['TN']]], 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['High', 'Low'], 
                yticklabels=['High', 'Low'])
    plt.title('Matrice de Confusion', fontsize=14)
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies Valeurs')
    
    # Métriques Clés
    plt.subplot(2, 2, 2)
    metrics_data = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    bars = plt.barh(metrics_data, values, color=COLORS)
    plt.bar_label(bars, fmt='%.2f%%', padding=5)
    plt.xlim(0, 100)
    plt.title('Métriques de Performance', fontsize=14)
    
    # Utilisation des Règles
    plt.subplot(2, 1, 2)
    rules_df = (pd.DataFrame.from_dict(rule_usage, orient='index')
                .reset_index()
                .rename(columns={'index': 'Règle', 0: 'Utilisations'})
                .sort_values('Utilisations', ascending=False)
                .head(10))
    sns.barplot(x='Utilisations', y='Règle', data=rules_df, palette=COLORS)
    plt.title('Top 10 des Règles les Plus Utilisées', fontsize=14)
    plt.xlabel('Nombre d\'Utilisations')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# EXÉCUTION PRINCIPALE
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Chargement des données
    data = load_dataset("csupESp.csv")
    classifier = RuleClassifier(RULES)
    
    # Interface Utilisateur
    print("⚡ Classificateur Intelligent par Règles ⚡\n")
    mode = input("Choisissez le mode d'évaluation:\n1. Train-Test Split\n2. Validation Croisée\n→ ")
    
    if mode == '1':
        # Mode Train-Test Split
        train, test = split_data(data, 0.6)
        predictions = [classifier.predict(x) for x in test]
        actuals = [x['total'] for x in test]
        metrics = calculate_metrics(predictions, actuals)
    elif mode == '2':
        # Mode Validation Croisée
        k = int(input("Nombre de folds (k): ") or 10)
        metrics = cross_validate(data, classifier, k)
    else:
        print("Mode non reconnu - Utilisation du mode par défaut (Train-Test 70/30)")
        train, test = split_data(data)
        predictions = [classifier.predict(x) for x in test]
        actuals = [x['total'] for x in test]
        metrics = calculate_metrics(predictions, actuals)
    
    # Affichage des Résultats
    print(f"\n🔍 Performance du Modèle 🔍")
    print(f"Exactitude: {metrics['accuracy']:.2f}%")
    print(f"Précision: {metrics['precision']:.2f}%")
    print(f"Rappel: {metrics['recall']:.2f}%")
    print(f"Score F1: {metrics['f1']:.2f}%")
    print(f"Échantillons Analysés: {metrics['total']}")
    
    # Génération des Visualisations
    visualize_performance(metrics, classifier.rule_usage)