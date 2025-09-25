# -*- coding: utf-8 -*-
"""
Application Flask de validation d'emails avec correcteur intelligent
"""

import re
import numpy as np
import pandas as pd
import io
import csv
from flask import Flask, render_template, request, jsonify, send_file
from difflib import SequenceMatcher, get_close_matches
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
# NOUVEAU MODULE : CORRECTEUR INTELLIGENT
# -----------------------------

class IntelligentEmailCorrector:
    def __init__(self):
        self.common_domains = [
            'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
            'protonmail.com', 'icloud.com', 'aol.com', 'yandex.com'
        ]
        
        self.common_typos = {
            'gmial.com': 'gmail.com',
            'gmail.cm': 'gmail.com',
            'gmail.con': 'gmail.com',
            'yaho.com': 'yahoo.com',
            'yahoo.cm': 'yahoo.com',
            'outlook.cm': 'outlook.com',
            'hotmail.cm': 'hotmail.com',
            'hotmail.co': 'hotmail.com'
        }
        
        self.common_username_typos = {
            'gmail': 'gmail',
            'gmai': 'gmail',
            'yahoo': 'yahoo', 
            'yaho': 'yahoo',
            'outlook': 'outlook',
            'outlok': 'outlook'
        }

    def suggest_corrections(self, email):
        """Propose des corrections intelligentes pour un email"""
        suggestions = []
        
        # Vérification basique de format
        if '@' not in email:
            return self._suggest_missing_at(email)
        
        username, domain = email.split('@', 1)
        
        # Suggestions pour le domaine
        domain_suggestions = self._suggest_domain_corrections(domain)
        # Suggestions pour le nom d'utilisateur
        username_suggestions = self._suggest_username_corrections(username)
        # Suggestions structurelles
        structural_suggestions = self._suggest_structural_corrections(email)
        
        # Combiner toutes les suggestions
        all_suggestions = domain_suggestions + username_suggestions + structural_suggestions
        
        # Éliminer les doublons et garder les meilleures
        unique_suggestions = list(set(all_suggestions))
        unique_suggestions.sort(key=lambda x: self._calculate_confidence(x))
        
        return unique_suggestions[:3]  # Retourne les 3 meilleures suggestions

    def _suggest_domain_corrections(self, domain):
        """Suggère des corrections pour le domaine"""
        suggestions = []
        
        # Correction des fautes de frappe courantes
        if domain in self.common_typos:
            suggestions.append(self.common_typos[domain])
        
        # Recherche de domaines similaires
        close_matches = get_close_matches(domain, self.common_domains, n=2, cutoff=0.7)
        suggestions.extend(close_matches)
        
        # Ajout d'extensions manquantes
        if '.' not in domain and len(domain) > 3:
            for common_domain in self.common_domains:
                main_domain = common_domain.split('.')[0]
                if main_domain in domain:
                    suggestions.append(common_domain)
        
        return [f"@{suggestion}" for suggestion in suggestions]

    def _suggest_username_corrections(self, username):
        """Suggère des corrections pour le nom d'utilisateur"""
        suggestions = []
        
        # Vérification des caractères invalides
        invalid_chars = re.findall(r'[^a-zA-Z0-9._%+-]', username)
        if invalid_chars:
            cleaned = re.sub(r'[^a-zA-Z0-9._%+-]', '', username)
            if cleaned:
                suggestions.append(cleaned)
        
        return suggestions

    def _suggest_structural_corrections(self, email):
        """Suggère des corrections structurelles"""
        suggestions = []
        
        # Double @
        if '@@' in email:
            suggestions.append(email.replace('@@', '@'))
        
        # Espaces
        if ' ' in email:
            suggestions.append(email.replace(' ', ''))
            suggestions.append(email.replace(' ', '.'))
        
        # Points mal placés
        if email.startswith('.') or email.endswith('.'):
            suggestions.append(email.strip('.'))
        
        return suggestions

    def _suggest_missing_at(self, email):
        """Suggère des corrections quand @ est manquant"""
        suggestions = []
        
        # Cherche des patterns qui ressemblent à des emails sans @
        for common_domain in self.common_domains:
            if common_domain in email:
                # Remplace le domaine par @domaine
                suggestion = email.replace(common_domain, f"@{common_domain}")
                suggestions.append(suggestion)
        
        return suggestions

    def _calculate_confidence(self, suggestion):
        """Calcule un score de confiance pour la suggestion"""
        score = 0
        
        # Bonus pour les domaines communs
        domain = suggestion.split('@')[-1] if '@' in suggestion else ''
        if domain in self.common_domains:
            score += 10
        
        # Bonus pour la longueur raisonnable
        if 5 <= len(suggestion) <= 50:
            score += 5
        
        # Malus pour les caractères spéciaux excessifs
        special_chars = len(re.findall(r'[^a-zA-Z0-9@._-]', suggestion))
        score -= special_chars * 2
        
        return score

# Initialisation du correcteur
corrector = IntelligentEmailCorrector()

# -----------------------------
# MODÈLE DE VALIDATION AMÉLIORÉ
# -----------------------------

# Dataset d'entraînement étendu
data = {
    "email": [
        "user@gmail.com", "test@yahoo.com", "contact@outlook.com",
        "john.doe@hotmail.com", "alice123@protonmail.com",
        "abc@gmail.com", "xyz@yahoo.fr", "name@esprit.tn",
        "bademail@", "noatsymbol.com", "hello@@gmail.com",
        "test@.com", "@gmail.com", "valid@example.org",
        "test.email+tag@gmail.com", "contact@company.com.uk",
        "user@gmial.com", "test@yahoesprit.tn", "contact@outlok.com"
    ],
    # Correction: "abc@gmail.com" should be valid (1 instead of 0)
    "label": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
}


df = pd.DataFrame(data)

# Domaines courants (étendu)
common_domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", 
                  "protonmail.com", "aol.com", "icloud.com", "example.org"]

def similar(a, b):
    """Calcule la similarité entre deux chaînes"""
    return SequenceMatcher(None, a, b).ratio()

def extract_features(email):
    """Extrait les caractéristiques d'un email (version améliorée)"""
    features = []
    
    # 1. Syntaxe de base
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    syntax_ok = 1 if re.match(pattern, email) else 0
    features.append(syntax_ok)
    
    # 2. Longueur totale
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
    
    # 7. Score de qualité (basé sur correcteur) – mais seulement si domaine n’est pas déjà valide
    suggestions = []
    if domain not in common_domains:
        suggestions = corrector.suggest_corrections(email)
    quality_score = max([corrector._calculate_confidence(sugg) for sugg in suggestions]) if suggestions else 10
    features.append(quality_score)
    
    # 8. Fautes de frappe connues
    has_common_typo = 1 if any(typo in email for typo in corrector.common_typos.keys()) else 0
    features.append(has_common_typo)
    
    return features


# Entraînement du modèle
print("Entraînement du modèle amélioré en cours...")
X = np.array([extract_features(e) for e in df["email"]])
y = np.array(df["label"])

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X, y)
print("Modèle amélioré entraîné avec succès!")

# -----------------------------
# FONCTIONS UTILITAIRES AMÉLIORÉES
# -----------------------------

def valider_email_unique(email):
    """Valide un email unique et retourne le résultat amélioré"""
    features = extract_features(email)
    proba = model.predict_proba([features])[0]
    prediction = model.predict([features])[0]
    confidence = max(proba)
    
    # Suggestions seulement si prediction invalide
    suggestions = corrector.suggest_corrections(email) if prediction == 0 else []
    
    quality_score = max([corrector._calculate_confidence(sugg) for sugg in suggestions]) if suggestions else 10
    
    return {
        'email': email,
        'valide': bool(prediction == 1),
        'confiance': round(confidence * 100, 2),
        'probabilite_valide': round(proba[1] * 100, 2),
        'probabilite_invalide': round(proba[0] * 100, 2),
        'suggestions_correction': suggestions[:3],
        'score_qualite': quality_score,
        'correction_automatique': suggestions[0] if suggestions and quality_score > 8 else None
    }


def valider_fichier(fichier_path):
    """Valide un fichier d'emails et retourne les résultats améliorés"""
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
        
        # NOUVEAU: Statistiques améliorées
        scores_qualite = [r['score_qualite'] for r in resultats_detailles]
        suggestions_total = sum(1 for r in resultats_detailles if r['suggestions_correction'])
        
        return {
            'total': len(emails),
            'valides': len(emails_valides),
            'invalides': len(emails_invalides),
            'taux_validite': round(len(emails_valides) / len(emails) * 100, 2) if emails else 0,
            'emails_valides': emails_valides,
            'emails_invalides': emails_invalides,
            'resultats_detailles': resultats_detailles,
            # NOUVELLES STATISTIQUES
            'score_qualite_moyen': round(np.mean(scores_qualite), 2) if scores_qualite else 0,
            'emails_corrigeables': suggestions_total,
            'taux_corrigeable': round(suggestions_total / len(emails) * 100, 2) if emails else 0
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
# NOUVELLE ROUTE POUR LA CORRECTION
# -----------------------------

@app.route('/corriger-email', methods=['POST'])
def corriger_email():
    """API dédiée à la correction d'emails"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'erreur': 'Données JSON requises'}), 400
            
        email = data.get('email', '').strip()
        
        if not email:
            return jsonify({'erreur': 'Email requis'}), 400
        
        suggestions = corrector.suggest_corrections(email)
        best_suggestion = suggestions[0] if suggestions else None
        
        return jsonify({
            'email_original': email,
            'suggestions': suggestions,
            'meilleure_suggestion': best_suggestion,
            'score_confiance': corrector._calculate_confidence(best_suggestion) if best_suggestion else 0,
            'nombre_suggestions': len(suggestions)
        })
    except Exception as e:
        return jsonify({'erreur': str(e)}), 500

# -----------------------------
# ROUTES FLASK EXISTANTES (MODIFIÉES)
# -----------------------------

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/valider-email', methods=['POST'])
def valider_email_api():
    """API pour valider un email unique (version améliorée)"""
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
    """API pour importer et valider un fichier d'emails (version améliorée)"""
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
    print("🌐 Lancement de l'application de validation d'emails intelligente...")
    print("📧 Accédez à l'interface sur: http://localhost:5000")
    print("🤖 Correcteur intelligent activé avec détection de fautes de frappe")
    app.run(debug=True, host='0.0.0.0', port=5000)