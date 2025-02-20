### **📌 Rule-Based Classifier with WEKA**  

**Un classificateur intelligent basé sur les règles d’association 🔍**  

📊 Ce projet implémente un **classificateur basé sur les règles d’association (CAR - Classification Association Rules)** en utilisant des règles extraites par **l’algorithme Apriori** sous **WEKA**. L'objectif est de comparer cette approche à des méthodes classiques de classification comme **K-Nearest Neighbors (KNN)**.  

---

## **🚀 Fonctionnalités**  
✅ Extraction et utilisation des **règles d’association** pour la classification  
✅ Évaluation du modèle avec **Train/Test Split et Validation Croisée**  
✅ Visualisation des performances : **Matrice de confusion, scores de précision, F1-score**  
✅ Comparaison avec **KNN** pour valider la pertinence des règles d’association  

---

## **🛠️ Technologies utilisées**  
- **Python** (pandas, seaborn, matplotlib)  
- **WEKA** (pour l'extraction des règles d’association)  
- **CSV Dataset** (*supermarket transactions*)  

---

## **📂 Structure du projet**  
```
📦 Rule-Based-Classifier-WEKA
├── 📄 Custom Classifier.py  # Code principal du classificateur
├── 📄 csupESp.csv  # Jeu de données (transactions supermarché)
├── 📄 README.md  # Documentation du projet
```

---

## **📥 Installation et exécution**  

### **1️⃣ Cloner le repository**  
```bash
git clone https://github.com/ton_username/Rule-Based-Classifier-WEKA.git
cd Rule-Based-Classifier-WEKA
```

### **2️⃣ Installer les dépendances**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Lancer le classificateur**  
```bash
python classifier.py
```

Vous pouvez choisir entre :  
- **Train/Test Split** (60% entraînement, 40% test)  
- **Validation croisée (k-fold)**  

---

## **📊 Résultats et comparaison avec KNN**  

| Modèle | Exactitude (%) | Précision (%) | Rappel (%) | F1-score (%) |  
|---------|--------------|-------------|----------|------------|  
| **CAR (Apriori)** | **79.95** | **79.95** | **61.29** | **72.90** |  
| **KNN (k=7, WEKA)** | 51.41 | 51.38 | 36.10 | 43.02 |  

💡 **Le classificateur basé sur les règles d’association dépasse largement KNN, notamment en précision et en F1-score.**  

---

## **📊 Visualisation des performances**  
Le script génère plusieurs graphiques :  
📌 **Matrice de confusion**  
📌 **Scores de performance (Accuracy, Precision, Recall, F1-Score)**  
📌 **Top 10 des règles les plus utilisées**  

---

## **📝 À propos**  
👨‍💻 **Développé par :** [Oussama Touijer](https://github.com/oussamavou)  
📧 **Contact :** oussama.touijer@um5r.ac.ma  

🔗 **WEKA Documentation :** [https://www.cs.waikato.ac.nz/ml/weka/](https://www.cs.waikato.ac.nz/ml/weka/)  

💡 **N’hésitez pas à ⭐ le repository et à contribuer !** 🚀
