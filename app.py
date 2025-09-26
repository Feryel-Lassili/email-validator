# -*- coding: utf-8 -*-
"""
Application Flask de validation d'emails avec correcteur intelligent AMÉLIORÉ
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
from collections import Counter

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
# MODULE CORRECTEUR INTELLIGENT AMÉLIORÉ
# -----------------------------

class IntelligentEmailCorrector:
    def __init__(self):
        self.common_domains = [
            'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
            'protonmail.com', 'icloud.com', 'aol.com', 'yandex.com',
            'esprit.tn', 'example.com', 'company.org'
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
        
        self.common_usernames = [
            'user', 'test', 'contact', 'admin', 'info', 'support',
            'john', 'alice', 'bob', 'david', 'sarah', 'michael',
            'service', 'hello', 'newsletter', 'noreply'
        ]

    def suggest_corrections(self, email):
        """Propose des corrections intelligentes pour un email"""
        suggestions = []
        
        # Cas 1: Email ne contient que le domaine (@hotmail.com)
        if email.startswith('@') and '.' in email:
            suggestions.extend(self._suggest_username_for_domain(email))
        
        # Cas 2: Email ne contient que le nom d'utilisateur (hhh@)
        elif email.endswith('@') and len(email) > 1:
            suggestions.extend(self._suggest_domain_for_username(email))
        
        # Cas 3: Aucun @ présent
        elif '@' not in email:
            suggestions.extend(self._suggest_missing_at(email))
        
        # Cas 4: Email complet mais avec erreurs
        else:
            try:
                username, domain = email.split('@', 1)
                suggestions.extend(self._suggest_domain_corrections(domain, username))
                suggestions.extend(self._suggest_username_corrections(username, domain))
                suggestions.extend(self._suggest_structural_corrections(email))
            except ValueError:
                # Cas où l'email a plusieurs @
                suggestions.extend(self._suggest_structural_corrections(email))
        
        # Éliminer les doublons et garder les meilleures
        unique_suggestions = list(set([s for s in suggestions if s]))
        unique_suggestions.sort(key=lambda x: self._calculate_confidence(x), reverse=True)
        
        return unique_suggestions[:5]  # Retourne les 5 meilleures suggestions

    def _suggest_username_for_domain(self, domain_email):
        """Suggère des noms d'utilisateur pour un domaine donné"""
        suggestions = []
        domain = domain_email[1:]  # Enlever le @
        
        for username in self.common_usernames:
            suggestions.append(f"{username}{domain_email}")  # user@hotmail.com
            suggestions.append(f"{username}.test{domain_email}")  # user.test@hotmail.com
            suggestions.append(f"{username}123{domain_email}")  # user123@hotmail.com
        
        # Ajouter des suggestions basées sur le domaine
        if 'hotmail' in domain:
            suggestions.append(f"contact{domain_email}")
            suggestions.append(f"support{domain_email}")
        
        return suggestions

    def _suggest_domain_for_username(self, username_email):
        """Suggère des domaines pour un nom d'utilisateur donné"""
        suggestions = []
        username = username_email[:-1]  # Enlever le @
        
        for domain in self.common_domains:
            suggestions.append(f"{username_email}{domain}")  # hhh@gmail.com
            suggestions.append(f"{username}@{domain}")  # hhh@gmail.com (format propre)
        
        return suggestions

    def _suggest_missing_at(self, email):
        """Suggère des corrections quand @ est manquant"""
        suggestions = []
        
        # Cherche des patterns qui ressemblent à des emails sans @
        for common_domain in self.common_domains:
            if common_domain in email:
                # Trouver où commence le domaine
                domain_index = email.find(common_domain)
                if domain_index > 0:
                    username = email[:domain_index]
                    domain = email[domain_index:]
                    suggestions.append(f"{username}@{domain}")
        
        # Si ça ressemble à "nomdomaine.com"
        if '.' in email and len(email.split('.')[-1]) >= 2:
            parts = email.split('.')
            if len(parts) >= 2:
                # Essayer différentes combinaisons
                suggestions.append(f"user@{email}")
                suggestions.append(f"contact@{email}")
                if len(parts[0]) > 2:  # Si la première partie semble être un username
                    suggestions.append(f"{parts[0]}@{'.'.join(parts[1:])}")
        
        return suggestions

    def _suggest_domain_corrections(self, domain, username=""):
        """Suggère des corrections pour le domaine"""
        suggestions = []
        
        # Correction des fautes de frappe courantes
        if domain in self.common_typos:
            suggestions.append(f"{username}@{self.common_typos[domain]}")
        
        # Recherche de domaines similaires
        close_matches = get_close_matches(domain, self.common_domains, n=3, cutoff=0.6)
        for match in close_matches:
            suggestions.append(f"{username}@{match}")
        
        # Ajout d'extensions manquantes
        if '.' not in domain and len(domain) > 3:
            for common_domain in self.common_domains:
                main_domain = common_domain.split('.')[0]
                if main_domain in domain:
                    suggestions.append(f"{username}@{common_domain}")
        
        return suggestions

    def _suggest_username_corrections(self, username, domain=""):
        """Suggère des corrections pour le nom d'utilisateur"""
        suggestions = []
        
        # Vérification des caractères invalides
        invalid_chars = re.findall(r'[^a-zA-Z0-9._%+-]', username)
        if invalid_chars:
            cleaned = re.sub(r'[^a-zA-Z0-9._%+-]', '', username)
            if cleaned:
                suggestions.append(f"{cleaned}@{domain}")
        
        # Suggestions de noms d'utilisateur communs
        if len(username) < 3 or not username.replace('.', '').replace('_', '').isalnum():
            for common_user in self.common_usernames:
                suggestions.append(f"{common_user}@{domain}")
                if username:  # Essayer de combiner avec l'username original
                    suggestions.append(f"{common_user}.{username}@{domain}")
        
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
        
        # @ au début ou à la fin
        if email.startswith('@'):
            suggestions.append(f"user{email}")
        if email.endswith('@'):
            suggestions.append(f"{email}gmail.com")
        
        return suggestions

    def _calculate_confidence(self, suggestion):
        """Calcule un score de confiance pour la suggestion"""
        if not suggestion:
            return -100
            
        score = 0
        
        # Vérifier que c'est un email valide
        if '@' not in suggestion or '.' not in suggestion:
            return -10
        
        try:
            parts = suggestion.split('@')
            if len(parts) != 2:
                return -10
            
            username, domain = parts
            
            # Bonus pour les domaines communs
            if domain in self.common_domains:
                score += 20
            
            # Bonus pour la longueur raisonnable du username
            if 3 <= len(username) <= 30:
                score += 10
            elif len(username) == 0:
                score -= 15  # Pénalité si username vide
            
            # Bonus pour username alphanumérique
            if username.replace('.', '').replace('_', '').replace('-', '').isalnum():
                score += 5
            
            # Malus pour les caractères spéciaux excessifs
            special_chars = len(re.findall(r'[^a-zA-Z0-9@._-]', suggestion))
            score -= special_chars * 3
            
            return score
        except:
            return -10

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
        "user@gmial.com", "test@yaho.com", "contact@outlok.com",
        "hhh@", "@hotmail.com", "usergmail.com"  # Nouveaux cas de test
    ],
    "label": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

# Domaines courants (étendu)
common_domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", 
                  "protonmail.com", "aol.com", "icloud.com", "example.org",
                  "esprit.tn"]

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
    
    # 7. Score de qualité (basé sur correcteur)
    suggestions = corrector.suggest_corrections(email)
    quality_score = max([corrector._calculate_confidence(sugg) for sugg in suggestions]) if suggestions else 0
    features.append(quality_score)
    
    # 8. Fautes de frappe connues
    has_common_typo = 1 if any(typo in email for typo in corrector.common_typos.keys()) else 0
    features.append(has_common_typo)
    
    # 9. Parties manquantes
    missing_username = 1 if email.startswith('@') else 0
    missing_domain = 1 if email.endswith('@') else 0
    missing_at = 1 if '@' not in email else 0
    features.extend([missing_username, missing_domain, missing_at])
    
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

def _get_email_diagnostic(email):
    """Retourne un diagnostic détaillé de l'email"""
    diagnostics = []
    
    if email.startswith('@'):
        diagnostics.append("❌ Nom d'utilisateur manquant avant le @")
    elif email.endswith('@'):
        diagnostics.append("❌ Domaine manquant après le @")
    elif '@' not in email:
        diagnostics.append("❌ Symbole @ manquant")
    else:
        username, domain = email.split('@', 1)
        if not username:
            diagnostics.append("❌ Nom d'utilisateur vide")
        if not domain:
            diagnostics.append("❌ Domaine vide")
        if '.' not in domain:
            diagnostics.append("❌ Extension de domaine manquante")
        if len(username) < 3:
            diagnostics.append("⚠️ Nom d'utilisateur trop court")
    
    return diagnostics

def valider_email_unique(email):
    """Valide un email unique et retourne le résultat amélioré"""
    features = extract_features(email)
    proba = model.predict_proba([features])[0]
    prediction = model.predict([features])[0]
    confidence = max(proba)
    
    # Toujours générer des suggestions
    suggestions = corrector.suggest_corrections(email)
    
    # Filtrer les suggestions valides
    valid_suggestions = []
    for suggestion in suggestions:
        if suggestion:  # Vérifier que la suggestion n'est pas vide
            suggestion_features = extract_features(suggestion)
            suggestion_pred = model.predict([suggestion_features])[0]
            if suggestion_pred == 1:  # Suggestion valide
                valid_suggestions.append(suggestion)
    
    quality_score = max([corrector._calculate_confidence(sugg) for sugg in valid_suggestions]) if valid_suggestions else 0
    
    return {
        'email': email,
        'valide': bool(prediction == 1),
        'confiance': round(confidence * 100, 2),
        'probabilite_valide': round(proba[1] * 100, 2),
        'probabilite_invalide': round(proba[0] * 100, 2),
        'suggestions_correction': valid_suggestions[:3],
        'score_qualite': quality_score,
        'correction_automatique': valid_suggestions[0] if valid_suggestions and quality_score > 5 else None,
        'diagnostic': _get_email_diagnostic(email)
    }

def analyser_statistiques_avancees(emails):
    """Analyse statistique avancée des emails - MODE EXPERT"""
    if not emails:
        return {}
    
    domains = []
    usernames = []
    patterns_erreurs = {
        'manque_arobase': 0,
        'domaine_manquant': 0,
        'username_manquant': 0,
        'double_arobase': 0,
        'caracteres_invalides': 0
    }
    
    for email in emails:
        # Analyse des domaines
        if '@' in email:
            parts = email.split('@')
            if len(parts) == 2:
                domains.append(parts[1])
                usernames.append(parts[0])
        
        # Détection des patterns d'erreurs
        if '@' not in email:
            patterns_erreurs['manque_arobase'] += 1
        elif email.startswith('@'):
            patterns_erreurs['username_manquant'] += 1
        elif email.endswith('@'):
            patterns_erreurs['domaine_manquant'] += 1
        elif email.count('@') > 1:
            patterns_erreurs['double_arobase'] += 1
        
        if re.search(r'[^a-zA-Z0-9@._-]', email):
            patterns_erreurs['caracteres_invalides'] += 1
    
    # Statistiques des domaines
    domain_stats = {}
    if domains:
        domain_counter = Counter(domains)
        domain_stats = dict(domain_counter.most_common(5))
    
    # Longueur moyenne des usernames
    avg_username_len = round(np.mean([len(u) for u in usernames]), 1) if usernames else 0
    
    # Score de taux de correction 
    emails_corrigeables = sum(1 for email in emails if corrector.suggest_corrections(email))
    taux_correction = round((emails_corrigeables / len(emails)) * 100, 1) if emails else 0

    return {
    'total_emails': len(emails),
    'domaines_populaires': domain_stats,
    'longueur_moyenne_username': avg_username_len,
    'patterns_erreurs': patterns_erreurs,
    'taux_correction': taux_correction  # ← Nouvelle métrique utile
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
            if email and len(email) > 0:
                resultat = valider_email_unique(email)
                resultats_detailles.append(resultat)
                
                if resultat['valide']:
                    emails_valides.append(email)
                else:
                    emails_invalides.append(email)
        
        # Statistiques améliorées
        scores_qualite = [r['score_qualite'] for r in resultats_detailles if r['score_qualite'] > 0]
        suggestions_total = sum(1 for r in resultats_detailles if r['suggestions_correction'])
        
        # AJOUT DU MODE EXPERT - Statistiques avancées
        statistiques_avancees = analyser_statistiques_avancees(emails)
        
        return {
            'total': len(emails),
            'valides': len(emails_valides),
            'invalides': len(emails_invalides),
            'taux_validite': round(len(emails_valides) / len(emails) * 100, 2) if emails else 0,
            'emails_valides': emails_valides,
            'emails_invalides': emails_invalides,
            'resultats_detailles': resultats_detailles,
            'score_qualite_moyen': round(np.mean(scores_qualite), 2) if scores_qualite else 0,
            'emails_corrigeables': suggestions_total,
            'taux_corrigeable': round(suggestions_total / len(emails) * 100, 2) if emails else 0,
            # NOUVEAU: Statistiques Mode Expert
            'statistiques_avancees': statistiques_avancees
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
            'suggestions': suggestions[:5],  # Limiter à 5 suggestions
            'meilleure_suggestion': best_suggestion,
            'score_confiance': corrector._calculate_confidence(best_suggestion) if best_suggestion else 0,
            'nombre_suggestions': len(suggestions),
            'diagnostic': _get_email_diagnostic(email)
        })
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
    
    if not allowed_file(fichier.filename):
        return jsonify({
            'erreur': 'Type de fichier non autorisé. Seuls les fichiers .txt et .csv sont acceptés.'
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
        return jsonify({'erreur': 'Type de fichier non autorisé'}), 400

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
    print("💡 Fonctionnalités améliorées :")
    print("   - Correction des emails avec parties manquantes")
    print("   - Diagnostic détaillé des erreurs")
    print("   - Suggestions intelligentes contextuelles")
    print("   - 🔍 MODE EXPERT avec statistiques avancées")
    app.run(debug=True, host='0.0.0.0', port=5000)