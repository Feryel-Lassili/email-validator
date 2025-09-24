# -*- coding: utf-8 -*-
"""
Application Flask de validation d'emails
"""

import re
import numpy as np
import pandas as pd
import io
import csv
from flask import Flask, render_template, request, jsonify, send_file
from difflib import SequenceMatcher
import os
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# CONFIGURATION FLASK
# -----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'csv'}
app.config['SECRET_KEY'] = 'votre_cle_secrete_ici'

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------
# MODÈLE DE VALIDATION
# -----------------------------

# Dataset d'entraînement
data = {
    "email": [
        "user@gmail.com", "test@yahoo.com", "contact@outlook.com",
        "john.doe@hotmail.com", "alice123@protonmail.com",
        "abc@gmail.com", "xyz@yahoo.fr", "name@esprit.tn",
        "bademail@", "noatsymbol.com", "hello@@gmail.com",
        "test@.com", "@gmail.com", "valid@example.org",
        "test.email+tag@gmail.com", "contact@company.com.uk"
    ],
    "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Domaines courants
common_domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", 
                  "protonmail.com", "aol.com", "icloud.com", "example.org"]

def similar(a, b):
    """Calcule la similarité entre deux chaînes"""
    return SequenceMatcher(None, a, b).ratio()

def extract_features(email):
    """Extrait les caractéristiques d'un email"""
    features = []
    
    # 1. Syntaxe de base
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    syntax_ok = 1 if re.match(pattern, email) else 0
    features.append(syntax_ok)
    
    # 2. Longueur
    features.append(len(email))
    
    # 3. Nombre de @
    features.append(email.count('@'))
    
    # 4. Nombre de parties
    parts = email.split('@')
    features.append(len(parts) if '@' in email else 0)
    
    # 5. Similarité du domaine
    domain = parts[-1] if len(parts) > 1 else ""
    if domain:
        similarities = [similar(domain, d) for d in common_domains]
        features.append(max(similarities) if similarities else 0)
    else:
        features.append(0)
    
    # 6. Caractères suspects
    suspicious = len(re.findall(r'[\!\#\$\%\^\&\*\(\)\=\+\{\}\[\]\:\;\'\<\>\?\/\\\|]', email))
    features.append(suspicious)
    
    return features

# Entraînement du modèle
print("Entraînement du modèle en cours...")
X = np.array([extract_features(e) for e in df["email"]])
y = np.array(df["label"])

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X, y)
print("Modèle entraîné avec succès!")

# -----------------------------
# FONCTIONS UTILITAIRES
# -----------------------------

def valider_email_unique(email):
    """Valide un email unique et retourne le résultat"""
    features = extract_features(email)
    proba = model.predict_proba([features])[0]
    prediction = model.predict([features])[0]
    confidence = max(proba)
    
    return {
        'email': email,
        'valide': bool(prediction == 1),
        'confiance': round(confidence * 100, 2),
        'probabilite_valide': round(proba[1] * 100, 2),
        'probabilite_invalide': round(proba[0] * 100, 2)
    }

def valider_fichier(fichier_path):
    """Valide un fichier d'emails et retourne les résultats"""
    emails_valides = []
    emails_invalides = []
    resultats_detailles = []
    
    try:
        # Lire le fichier selon son extension
        if fichier_path.endswith('.csv'):
            df = pd.read_csv(fichier_path)
            # Chercher la colonne email
            colonne_email = None
            for col in df.columns:
                if 'email' in col.lower():
                    colonne_email = col
                    break
            if colonne_email is None and len(df.columns) > 0:
                colonne_email = df.columns[0]
            
            if colonne_email:
                emails = df[colonne_email].dropna().astype(str).str.strip().tolist()
            else:
                emails = []
        else:  # Fichier texte
            with open(fichier_path, 'r', encoding='utf-8') as f:
                emails = [line.strip() for line in f if line.strip()]
        
        # Valider chaque email
        for email in emails:
            if email and len(email) > 3:
                resultat = valider_email_unique(email)
                resultats_detailles.append(resultat)
                
                if resultat['valide']:
                    emails_valides.append(email)
                else:
                    emails_invalides.append(email)
        
        return {
            'total': len(emails),
            'valides': len(emails_valides),
            'invalides': len(emails_invalides),
            'taux_validite': round(len(emails_valides) / len(emails) * 100, 2) if emails else 0,
            'emails_valides': emails_valides,
            'emails_invalides': emails_invalides,
            'resultats_detailles': resultats_detailles
        }
    except Exception as e:
        raise Exception(f"Erreur lecture fichier: {str(e)}")

def creer_fichier_resultat(emails_valides, format_sortie='csv'):
    """Crée un fichier de résultat avec les emails valides"""
    if format_sortie == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['email_valide'])
        for email in emails_valides:
            writer.writerow([email])
        content = output.getvalue()
        return content, 'emails_valides.csv'
    else:  # Format texte
        content = '\n'.join(emails_valides)
        return content, 'emails_valides.txt'

def allowed_file(filename):
    """Vérifie l'extension du fichier"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# -----------------------------
# ROUTES FLASK
# -----------------------------

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/valider-email', methods=['POST'])
def valider_email_api():
    """API pour valider un email unique"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'erreur': 'Données JSON requises'}), 400
            
        email = data.get('email', '').strip()
        
        if not email:
            return jsonify({'erreur': 'Email requis'}), 400
        
        resultat = valider_email_unique(email)
        return jsonify(resultat)
    except Exception as e:
        return jsonify({'erreur': str(e)}), 500

@app.route('/importer-fichier', methods=['POST'])
def importer_fichier():
    """API pour importer et valider un fichier d'emails"""
    if 'fichier' not in request.files:
        return jsonify({'erreur': 'Aucun fichier uploadé'}), 400
    
    fichier = request.files['fichier']
    
    if fichier.filename == '':
        return jsonify({'erreur': 'Aucun fichier sélectionné'}), 400
    
    # Vérifier l'extension du fichier
    if not allowed_file(fichier.filename):
        return jsonify({
            'erreur': 'Type de fichier non autorisé. Seuls les fichiers .txt et .csv sont acceptés.',
            'type_fichier': fichier.filename.split('.')[-1].lower() if '.' in fichier.filename else 'inconnu'
        }), 400
    
    if fichier and allowed_file(fichier.filename):
        try:
            filename = secure_filename(fichier.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            fichier.save(filepath)
            
            resultats = valider_fichier(filepath)
            
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify(resultats)
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'erreur': f'Erreur lors du traitement: {str(e)}'}), 500
    else:
        return jsonify({'erreur': 'Type de fichier non autorisé. Utilisez .txt ou .csv'}), 400

@app.route('/telecharger-resultat', methods=['POST'])
def telecharger_resultat():
    """API pour télécharger le fichier résultat"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'erreur': 'Données JSON requises'}), 400
            
        emails_valides = data.get('emails_valides', [])
        format_sortie = data.get('format', 'csv')
        
        if not emails_valides:
            return jsonify({'erreur': 'Aucun email valide à exporter'}), 400
        
        contenu, nom_fichier = creer_fichier_resultat(emails_valides, format_sortie)
        
        return jsonify({
            'contenu': contenu,
            'nom_fichier': nom_fichier,
            'format': format_sortie
        })
    except Exception as e:
        return jsonify({'erreur': str(e)}), 500

@app.route('/telecharger-fichier', methods=['POST'])
def telecharger_fichier():
    """Télécharge le fichier généré"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'erreur': 'Données JSON requises'}), 400
            
        contenu = data.get('contenu', '')
        nom_fichier = data.get('nom_fichier', 'resultat.txt')
        
        output = io.BytesIO()
        output.write(contenu.encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output,
            as_attachment=True,
            download_name=nom_fichier,
            mimetype='text/csv' if nom_fichier.endswith('.csv') else 'text/plain'
        )
    except Exception as e:
        return jsonify({'erreur': str(e)}), 500

# -----------------------------
# LANCEMENT DE L'APPLICATION
# -----------------------------

if __name__ == '__main__':
    print("🌐 Lancement de l'application de validation d'emails...")
    print("📧 Accédez à l'interface sur: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)