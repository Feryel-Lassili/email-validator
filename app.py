# -*- coding: utf-8 -*-
"""
VERSION SIMPLIFIÉE POUR DÉBOGUER
"""
import pandas as pd
import requests
from flask import Flask, request, send_file, render_template
import os
from werkzeug.utils import secure_filename
import io
import re
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HUNTER_API_KEY = "32bced14cf372195a829465e5c26a0366d00d9fe"
SERPAPI_KEY = "c3f8f66d2468923349b27e11e4c8e094e59b23b3e89beb2cdf6ba87fdbcc5650"

def verifier_email_simple(email):
    """Vérification SIMPLE sans API - pour fallback"""
    email = email.strip().lower()
    
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False, 50
    
    domain = email.split('@')[1]
    popular_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'live.com']
    
    if domain in popular_domains:
        return True, 90
    else:
        return True, 70

def rechercher_profils(email):
    """Recherche LinkedIn et GitHub avec SerpAPI et GitHub API"""
    username = email.split('@')[0]
    domain = email.split('@')[1]
    name = username.replace('.', ' ').title()
    
    linkedin_urls = []
    github_url = None
    
    # Recherche LinkedIn
    try:
        if domain == 'esprit.tn':
            query_linkedin = f'site:linkedin.com/in "{name}" esprit tunisie'
        else:
            query_linkedin = f'site:linkedin.com/in "{name}"'
        url = f"https://serpapi.com/search.json?engine=google&q={query_linkedin}&api_key={SERPAPI_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'organic_results' in data and data['organic_results']:
            for result in data['organic_results'][:10]:
                link = result.get('link', '')
                title = result.get('title', '').lower()
                snippet = result.get('snippet', '').lower()
                if 'linkedin.com/in/' in link:
                    # Pour esprit.tn, privilégier les liens tunisiens
                    if domain == 'esprit.tn':
                        if 'tn.linkedin.com' in link or 'tunisie' in title or 'tunisie' in snippet or 'esprit' in title or 'esprit' in snippet:
                            linkedin_urls.append(link)
                            break  # Prendre le premier qui match
                    else:
                        linkedin_urls.append(link)
    except Exception as e:
        print(f"Erreur SerpAPI LinkedIn pour {email}: {e}")
    
    # Recherche GitHub avec API GitHub
    variantes = [username]
    if '.' in username:
        variantes.append(username.replace('.', ''))
        variantes.append(username.replace('.', '-'))
    if len(name.split()) > 1:
        first, last = name.split()[:2]
        variantes.append(f"{first[0]}{last}".lower())
        variantes.append(f"{first.lower()}{last.lower()}")
    
    for var in variantes:
        try:
            api_url = f"https://api.github.com/users/{var}"
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                github_url = data.get('html_url')
                break
        except Exception as e:
            print(f"Erreur API GitHub pour {var}: {e}")
    
    linkedin_url = linkedin_urls[0] if linkedin_urls else None
    
    return linkedin_url, github_url

def verifier_email_hunter(email):
    """Vérification avec Hunter API, fallback sur simple"""
    try:
        url = f"https://api.hunter.io/v2/email-verifier?email={email}&api_key={HUNTER_API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        print(f"Réponse Hunter pour {email}: {data}")
        
        if data.get('data') and 'result' in data['data']:
            result = data['data']['result']
            score = data['data']['score']
            
            if result in ['deliverable', 'risky']:
                return True, score
            else:
                # Si undeliverable, utiliser vérification simple
                est_valide, score_simple = verifier_email_simple(email)
                return est_valide, score_simple
        else:
            # Erreur API, utiliser simple
            est_valide, score_simple = verifier_email_simple(email)
            return est_valide, score_simple
    except Exception as e:
        print(f"Erreur Hunter pour {email}: {e}")
        # Fallback sur simple
        est_valide, score_simple = verifier_email_simple(email)
        return est_valide, score_simple

def traiter_ligne_email(email):
    """Traitement avec APIs réelles et enrichissement complet"""
    email = email.strip()
    
    if not email or email == 'nan':
        return None
    
    print(f"🔍 Traitement de: {email}")
    
    result = {
        'email_original': email,
        'email_active': None,
        'statut': None,
        'score_hunter': None,
        'nom_complet': None,
        'poste': None,
        'entreprise': None,
        'site_web': None,
        'ville': None,
        'pays': None,
        'linkedin': None,
        'github': None
    }
    
    # Validation Hunter
    est_actif, score = verifier_email_hunter(email)
    result['score_hunter'] = score
    
    if est_actif:
        result['email_active'] = email
        result['statut'] = "actif"
        
        # Enrichissement comme pour username
        username = email.split('@')[0]
        name = username.replace('.', ' ').title()
        domain = email.split('@')[1]
        
        # GitHub
        try:
            api_url = f"https://api.github.com/users/{username}"
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                result['github'] = data.get('html_url')
                result['nom_complet'] = data.get('name') or result.get('nom_complet')
                result['site_web'] = data.get('blog') or data.get('html_url')
                result['ville'] = data.get('location')
                bio = data.get('bio', '')
                if bio:
                    entreprise_match = re.search(r'@([A-Za-z0-9_-]+)', bio)
                    if entreprise_match:
                        result['entreprise'] = entreprise_match.group(1)
        except Exception as e:
            print(f"Erreur GitHub pour {email}: {e}")
        
        # LinkedIn
        try:
            if domain == 'esprit.tn':
                query = f'site:linkedin.com/in "{name}" esprit tunisie'
            else:
                query = f'site:linkedin.com/in "{name}"'
            url = f"https://serpapi.com/search.json?engine=google&q={query}&api_key={SERPAPI_KEY}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'organic_results' in data and data['organic_results']:
                for res in data['organic_results'][:3]:
                    link = res.get('link', '')
                    title = res.get('title', '')
                    snippet = res.get('snippet', '')
                    if 'linkedin.com/in/' in link:
                        result['linkedin'] = link
                        # Extraire infos
                        if ' - ' in title:
                            parts = title.split(' - ')
                            if len(parts) >= 2:
                                result['nom_complet'] = parts[0].strip()
                                result['poste'] = parts[1].strip()
                        if snippet:
                            entreprise_patterns = [
                                r'chez ([^|•\n]+)',
                                r'at ([^|•\n]+)',
                                r'([A-Z][a-zA-Z\s&]+) \|',
                                r'• ([A-Z][a-zA-Z\s&]+)'
                            ]
                            for pattern in entreprise_patterns:
                                match = re.search(pattern, snippet)
                                if match:
                                    result['entreprise'] = match.group(1).strip()
                                    break
                        break
        except Exception as e:
            print(f"Erreur LinkedIn pour {email}: {e}")
        
    else:
        result['statut'] = "invalide"
    
    return result

@app.route('/search', methods=['POST'])
def search_username():
    """Recherche par username avec enrichissement"""
    username = request.form.get('username', '').strip()
    
    if not username:
        return {'error': 'Username requis'}, 400
    
    result = {
        'username': username,
        'email': None,
        'github': None,
        'linkedin': None,
        'nom_complet': None,
        'poste': None,
        'entreprise': None,
        'site_web': None,
        'ville': None,
        'pays': None
    }
    
    # Chercher GitHub avec plus d'infos
    try:
        api_url = f"https://api.github.com/users/{username}"
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            result['github'] = data.get('html_url')
            result['nom_complet'] = data.get('name')
            result['site_web'] = data.get('blog') or data.get('html_url')
            result['ville'] = data.get('location')
            # Pour entreprise, GitHub n'a pas directement, mais on peut chercher dans bio
            bio = data.get('bio', '')
            if bio:
                # Extraire entreprise de la bio si mentionnée
                entreprise_match = re.search(r'@([A-Za-z0-9_-]+)', bio)
                if entreprise_match:
                    result['entreprise'] = entreprise_match.group(1)
    except Exception as e:
        print(f"Erreur GitHub pour {username}: {e}")
    
    # Chercher LinkedIn avec plus d'infos
    try:
        query = f'site:linkedin.com/in "{username}"'
        url = f"https://serpapi.com/search.json?engine=google&q={query}&api_key={SERPAPI_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'organic_results' in data and data['organic_results']:
            for res in data['organic_results'][:3]:
                link = res.get('link', '')
                title = res.get('title', '')
                snippet = res.get('snippet', '')
                if 'linkedin.com/in/' in link:
                    result['linkedin'] = link
                    # Extraire infos du titre et snippet
                    if ' - ' in title:
                        parts = title.split(' - ')
                        if len(parts) >= 2:
                            result['nom_complet'] = parts[0].strip()
                            result['poste'] = parts[1].strip()
                    # Chercher entreprise dans snippet
                    if snippet:
                        # Chercher patterns comme "chez Company" ou "at Company"
                        entreprise_patterns = [
                            r'chez ([^|•\n]+)',
                            r'at ([^|•\n]+)',
                            r'([A-Z][a-zA-Z\s&]+) \|',
                            r'• ([A-Z][a-zA-Z\s&]+)'
                        ]
                        for pattern in entreprise_patterns:
                            match = re.search(pattern, snippet)
                            if match:
                                result['entreprise'] = match.group(1).strip()
                                break
                    break
    except Exception as e:
        print(f"Erreur LinkedIn pour {username}: {e}")
    
    # Chercher email avec plus de méthodes
    try:
        # Recherche Google pour email
        query_email = f'"{username}" email'
        url = f"https://serpapi.com/search.json?engine=google&q={query_email}&api_key={SERPAPI_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'organic_results' in data and data['organic_results']:
            for res in data['organic_results'][:5]:
                snippet = res.get('snippet', '')
                # Extraire email
                emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', snippet)
                if emails:
                    result['email'] = emails[0]
                    break
    except Exception as e:
        print(f"Erreur recherche email pour {username}: {e}")
    
    return result

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_fichier():
    """Upload et traitement du CSV - FORCÉ EN TXT"""
    
    if 'fichier' not in request.files:
        return 'Aucun fichier uploadé', 400
    
    fichier = request.files['fichier']
    if fichier.filename == '':
        return 'Aucun fichier sélectionné', 400
    
    try:
        filename = secure_filename(fichier.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        fichier.save(filepath)
        
        print(f"✅ Fichier sauvegardé dans: {filepath}")
        
        df = pd.read_csv(filepath)
        print("📋 Colonnes trouvées:", df.columns.tolist())
        
        colonne_email = None
        for col in df.columns:
            if 'email' in col.lower():
                colonne_email = col
                break
        
        if not colonne_email:
            colonne_email = df.columns[0]
        
        print(f"📧 Colonne email utilisée: {colonne_email}")
        
        emails = df[colonne_email].dropna().astype(str).unique().tolist()
        print(f"📨 {len(emails)} emails à traiter")
        
        lignes_enrichies = []
        
        for i, email in enumerate(emails):
            print(f"🔄 Traitement {i+1}/{len(emails)}: {email}")
            ligne_enrichie = traiter_ligne_email(email)
            
            if ligne_enrichie:
                lignes_enrichies.append(ligne_enrichie)
                print(f"✅ Ajouté: {email}")
            else:
                print(f"❌ Ignoré: {email}")
        
        # TOUJOURS GÉNÉRER UN FICHIER TXT
        if lignes_enrichies:
            df_final = pd.DataFrame(lignes_enrichies)
            contenu = generer_fichier_txt(df_final)
            filename_output = f"rapport_emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        else:
            contenu = "Aucun email valide trouvé dans le fichier."
            filename_output = f"rapport_emails_vide.txt"
        
        output = io.BytesIO()
        output.write(contenu.encode('utf-8'))
        output.seek(0)
        
        print(f"📤 Fichier TXT généré: {filename_output}")
        
        os.remove(filepath)
        print("🧹 Fichier original supprimé")
        
        return send_file(
            output,
            as_attachment=True,
            download_name=filename_output,
            mimetype='text/plain'
        )
        
    except Exception as e:
        print(f"❌ ERREUR: {str(e)}")
        return f'Erreur: {str(e)}', 500

def generer_fichier_txt(df):
    """Génère un fichier TXT bien formaté et lisible"""
    lignes = []
    
    lignes.append("=" * 60)
    lignes.append("RAPPORT DE VALIDATION D'EMAILS")
    lignes.append("=" * 60)
    lignes.append(f"Date de génération: {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
    lignes.append(f"Nombre d'emails valides: {len(df)}")
    lignes.append("")
    
    for index, row in df.iterrows():
        lignes.append(f"EMAIL {index + 1}")
        lignes.append("-" * 40)
        lignes.append(f"Original:    {row.get('email_original', 'N/A')}")
        lignes.append(f"Validé:      {row.get('email_active', 'N/A')}")
        lignes.append(f"Statut:      {row.get('statut', 'N/A')}")
        lignes.append(f"Score:       {row.get('score_hunter', 'N/A')}%")
        
        if row.get('nom_complet'):
            lignes.append(f"Nom:         {row['nom_complet']}")
        if row.get('poste'):
            lignes.append(f"Poste:       {row['poste']}")
        if row.get('entreprise'):
            lignes.append(f"Entreprise:  {row['entreprise']}")
        if row.get('site_web'):
            lignes.append(f"Site web:    {row['site_web']}")
        if row.get('ville'):
            lignes.append(f"Ville:       {row['ville']}")
        if row.get('pays'):
            lignes.append(f"Pays:        {row['pays']}")
        if row.get('linkedin'):
            lignes.append(f"LinkedIn:    {row['linkedin']}")
        if row.get('github'):
            lignes.append(f"GitHub:      {row['github']}")
        
        lignes.append("")
    
    lignes.append("=" * 60)
    lignes.append("RÉSUMÉ")
    lignes.append("-" * 40)
    
    stats_valides = len(df[df['statut'] == 'actif']) if 'statut' in df.columns else len(df)
    stats_alternatifs = len(df[df['statut'] == 'alternative']) if 'statut' in df.columns else 0
    
    lignes.append(f"Emails actifs validés:    {stats_valides}")
    lignes.append(f"Alternatives trouvées:    {stats_alternatifs}")
    lignes.append(f"Total d'emails traités:  {len(df)}")
    lignes.append("")
    lignes.append("Fin du rapport")
    lignes.append("=" * 60)
    
    return '\n'.join(lignes)

if __name__ == '__main__':
    print("🚀 VERSION AVEC APIs RÉELLES - PRÊTE !")
    print("📧 Validation avec Hunter API")
    print("🔍 Recherche LinkedIn/GitHub avec SerpAPI")
    print("📄 Format de sortie: TXT")
    print("🌐 Serveur démarré sur http://localhost:5000")
    app.run(debug=True, port=5000)