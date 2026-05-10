"""
Point d'entrée principal du démonstrateur KYC.

Exemple d'utilisation pour la conférence!
"""

import os
import re
import sys

os.environ["LANGCHAIN_OPENAI_TCP_KEEPALIVE"] = "0"

from dotenv import load_dotenv

from chains.llm_chain import KYCDocumentChain
from pipeline import KYCPipeline

# Charger les variables d'environnement depuis .env
load_dotenv()


def demo_document_unique(image_path: str):
    """
    Démo: traiter un seul document.

    Args:
        image_path: Chemin vers l'image du document
    """
    print("\n" + "=" * 70)
    print("🎯 DÉMO: Classification et extraction d'un document unique")
    print("=" * 70)

    chain = KYCDocumentChain()
    result = chain.process_document(image_path)

    if result.extraction_reussie:
        # Afficher selon le type
        if result.carte_identite:
            cni = result.carte_identite
            # Accord du participe passé selon le sexe
            ne_text = "Né.e"
            if cni.sexe:
                if cni.sexe.value == "M":
                    ne_text = "Né"
                elif cni.sexe.value == "F":
                    ne_text = "Née"
                print(f"📄 Carte d'Identité de {cni.prenom} {cni.nom} ({cni.sexe.value})")
            else:
                print(f"📄 Carte d'Identité de {cni.prenom} {cni.nom}")
            print(f"   {ne_text} le: {cni.date_naissance}")
            print(f"   N° document: {cni.numero_document}")
            print(f"   Valide jusqu'au: {cni.date_expiration}")
            print(f"   Statut: {'✓ Valide' if cni.est_valide else '✗ Expirée'}")

        elif result.passeport:
            pp = result.passeport
            # Civilité
            if pp.sexe:
                civilite = (
                    "M." if pp.sexe.value == "M" else ("Mme" if pp.sexe.value == "F" else "X")
                )
                ne_text = (
                    "Né" if pp.sexe.value == "M" else ("Née" if pp.sexe.value == "F" else "Né(e)")
                )
                print(f"📕 Passeport de {civilite} {pp.prenom} {pp.nom} ({pp.sexe.value})")
            else:
                ne_text = "Né(e)"
                print(f"📕 Passeport de {pp.prenom} {pp.nom}")
            # Date et lieu de naissance
            print(
                f"   {ne_text} le: {pp.date_naissance} à {pp.lieu_naissance if pp.lieu_naissance else '?'}"
            )
            # Nationalité
            print(f"   Nationalité: {pp.nationalite}")
            # Statut marital
            if pp.statut_marital:
                print(f"   Statut marital: {pp.statut_marital}")
            # Adresse (si présente)
            if pp.adresse:
                print(f"   Adresse: {pp.adresse}")
            # Numéro, dates, autorité
            print(f"   N° passeport: {pp.numero_passeport}")
            print(f"   Valide jusqu'au: {pp.date_expiration}")
            print(f"   Statut: {'✓ Valide' if pp.est_valide else '✗ Expiré'}")

        elif result.permis_conduire:
            pc = result.permis_conduire
            print(f"🚗 Permis de Conduire de {pc.prenom} {pc.nom}")
            print(f"   Catégories: {', '.join(pc.categories)}")
            print("   ⭐ POINT IMPORTANT: ces catégories ont été détectées")
            print("      visuellement par le LLM en regardant les cases du permis!")
            print("      Avant = deep learning avec bounding boxes pour chaque case")
            print("      Maintenant = prompt simple!")

        elif result.justificatif_domicile:
            jd = result.justificatif_domicile
            print("🏠 Justificatif de Domicile")
            print(f"   Type: {jd.type_document.value}")
            print(f"   Titulaire: {jd.nom_complet}")
            print(f"   Adresse: {jd.adresse_ligne1}")
            if jd.adresse_ligne2:
                print(f"            {jd.adresse_ligne2}")
            print(f"            {jd.code_postal} {jd.ville}")
            print(f"   Date: {jd.date_document}")
            if jd.emetteur:
                print(f"   Émetteur: {jd.emetteur}")
            print(f"   Statut: {'✓ Récent (< 3 mois)' if jd.est_recent else '✗ Trop ancien'}")

        elif result.rib:
            rib = result.rib
            print("🏦 RIB")
            print(f"   Titulaire: {rib.nom_titulaire}")
            print(f"   IBAN: {rib.iban or '⚠️  Non détecté'}")
            if rib.bic:
                print(f"   BIC: {rib.bic}")
            if rib.nom_banque:
                print(f"   Banque: {rib.nom_banque}")
            if rib.iban:
                if rib.iban_valide:
                    print("   Checksum IBAN: ✓ Valide")
                    print("   ⭐ L'IBAN a été validé avec l'algorithme modulo 97!")
                else:
                    print("   Checksum IBAN: ✗ Invalide")

        elif result.formulaire_inscription:
            form = result.formulaire_inscription
            print("📋 FORMULAIRE D'INSCRIPTION")
            print(f"   N° formulaire: {form.numero_formulaire}")
            print()

            # Statistiques impressionnantes pour la démo
            nb_cases = form.nombre_cases_cochees
            print(f"   ⭐ CASES COCHÉES DÉTECTÉES: {nb_cases}")
            print("      Le LLM a analysé visuellement ~120 cases possibles!")
            print()

            # Section 1: État civil
            ec = form.etat_civil
            if ec.civilite or ec.nom_naissance:
                print("   👤 ÉTAT CIVIL")
                if ec.civilite:
                    print(f"      Civilité: {ec.civilite.value}")
                if ec.nom_naissance:
                    print(f"      Nom: {ec.nom_naissance}")
                if ec.prenoms:
                    print(f"      Prénom(s): {ec.prenoms}")
                if ec.date_naissance:
                    print(f"      Né(e) le: {ec.date_naissance}")
                if ec.nationalite:
                    print(f"      Nationalité: {ec.nationalite.value}")

            # Section 3: Situation pro
            sp = form.situation_professionnelle
            if sp.statut:
                print()
                print("   💼 SITUATION PROFESSIONNELLE")
                print(f"      Statut: {sp.statut.value}")
                if sp.employeur:
                    print(f"      Employeur: {sp.employeur}")
                if sp.experience:
                    print(f"      Expérience: {sp.experience.value}")

            # Section 5: Formation souhaitée
            fs = form.formation_souhaitee
            if fs.domaines:
                print()
                print("   🎓 FORMATION SOUHAITÉE")
                print(f"      Domaines: {', '.join(d.value for d in fs.domaines)}")
                if fs.niveau_vise:
                    print(f"      Niveau visé: {fs.niveau_vise.value}")
                if fs.modalite:
                    print(f"      Modalité: {fs.modalite.value}")

            # Section 6: Financement
            fin = form.financement
            if fin.types_financement:
                print()
                print("   💰 FINANCEMENT")
                print(f"      Types: {', '.join(f.value for f in fin.types_financement)}")
                if fin.statut_accord:
                    print(f"      Statut: {fin.statut_accord.value}")

            # Section 7: Compétences (tableau)
            comp = form.competences
            competences_remplies = []
            if comp.programmation:
                competences_remplies.append(f"Programmation: {comp.programmation.value}")
            if comp.bases_de_donnees:
                competences_remplies.append(f"BDD: {comp.bases_de_donnees.value}")
            if comp.cloud:
                competences_remplies.append(f"Cloud: {comp.cloud.value}")
            if comp.machine_learning:
                competences_remplies.append(f"ML/IA: {comp.machine_learning.value}")
            if comp.devops:
                competences_remplies.append(f"DevOps: {comp.devops.value}")
            if competences_remplies:
                print()
                print("   📊 AUTO-ÉVALUATION (tableau 7×6 cases)")
                for c in competences_remplies:
                    print(f"      {c}")

            # Section 10: Engagements
            eng = form.engagements
            engagements_coches = sum(
                [
                    eng.exactitude_informations,
                    eng.reglement_interieur,
                    eng.conditions_generales,
                    eng.autorisation_contact,
                    eng.accepte_autres_formations,
                ]
            )
            if engagements_coches > 0:
                print()
                print(f"   ✅ ENGAGEMENTS: {engagements_coches}/5 cases cochées")
                if eng.signature_presente:
                    print("   ✍️  Signature: PRÉSENTE")
                if eng.date_signature:
                    print(f"   📅 Date: {eng.date_signature}")

            # Message final impressionnant
            print()
            print("   " + "=" * 50)
            print("   🎯 POINT CLÉ DE LA DÉMO:")
            print("      Avec du deep learning classique, il aurait fallu:")
            print("      - Annoter ~120 cases × milliers de formulaires")
            print("      - Entraîner un détecteur pour chaque case")
            print("      - Classifier coché/non coché pour chacune")
            print("      = Des MOIS de travail!")
            print()
            print("      Avec un LLM multimodal:")
            print("      - 1 prompt bien écrit")
            print("      - 1 schéma Pydantic")
            print("      = Quelques HEURES de travail!")
            print("   " + "=" * 50)

    else:
        print("\n❌ ÉCHEC DE L'EXTRACTION\n")
        if result.erreurs:
            for erreur in result.erreurs:
                _print_simplified_errors(erreur)


def _print_simplified_errors(erreur: str):
    """Affiche les erreurs de validation Pydantic de façon simplifiée."""
    # Extraire le nombre d'erreurs
    match = re.match(r"(\d+) validation errors? for (\w+)", erreur)
    if not match:
        print(f"   - {erreur}")
        return

    nb_errors, model_name = match.group(1), match.group(2)
    print(f"   {nb_errors} erreurs de validation ({model_name}):\n")

    # Parser chaque erreur individuelle
    # Format: "champ\n  message [type=..., input_value=..., input_type=...]\n    For further..."
    lines = erreur.split("\n")
    i = 1  # Skip la première ligne (déjà affichée)
    while i < len(lines):
        line = lines[i].strip()
        # Skip les lignes "For further information"
        if line.startswith("For further information"):
            i += 1
            continue
        # Ligne de champ (pas d'indentation ou indentation simple)
        if line and not line.startswith("Input should") and not line.startswith("For "):
            field = line
            # Lire la ligne de message suivante
            if i + 1 < len(lines):
                msg_line = lines[i + 1].strip()
                # Extraire input_value et les valeurs attendues
                val_match = re.search(r"input_value=(.+?), input_type=(\w+)", msg_line)
                recu = val_match.group(1) if val_match else "?"
                type_recu = val_match.group(2) if val_match else "?"
                # Identifier le type d'erreur et formater
                if "bool" in msg_line:
                    print(f"   {field}: reçu {recu} ({type_recu}) → attendu true/false (bool)")
                else:
                    expected_match = re.search(r"Input should be (.+?) \[type=", msg_line)
                    if expected_match and val_match:
                        attendu_raw = expected_match.group(1)
                        options = re.findall(r"'[^']*'", attendu_raw)
                        if len(options) <= 3:
                            attendu = " | ".join(options)
                        else:
                            attendu = " | ".join(options[:2]) + f" | ... ({len(options)} options)"
                        print(f"   {field}: reçu {recu} ({type_recu}) → attendu {attendu}")
                    else:
                        print(f"   {field}: {msg_line}")
                i += 2
                continue
        i += 1


def demo_dossier_complet(folder_path: str):
    """
    Démo: traiter un dossier KYC complet.

    Args:
        folder_path: Chemin vers le dossier contenant les documents
    """
    pipeline = KYCPipeline()
    dossier = pipeline.process_folder(folder_path)

    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DU DOSSIER KYC")
    print("=" * 70 + "\n")

    # Récapitulatif
    print(f"Dossier complet: {'✓ OUI' if dossier.dossier_complet else '✗ NON'}")
    print(f"Cohérence identité: {'✓ OUI' if dossier.coherence_identite else '✗ NON'}")

    if dossier.erreurs_validation:
        print("\n⚠️  Erreurs de validation:")
        for erreur in dossier.erreurs_validation:
            print(f"   - {erreur}")

    print("\n" + "=" * 70)
    print("💡 Ce qui a changé depuis l'époque du deep learning:")
    print("=" * 70)
    print("""
AVANT (Deep Learning classique):
❌ Créer des datasets avec milliers d'images annotées
❌ Annoter des bounding boxes pour chaque champ
❌ Entraîner des modèles CNN pour classification
❌ Entraîner des modèles pour détection de cases cochées
❌ OCR post-processing complexe
❌ Règles hardcodées pour extraction de chaque format
❌ Maintenance cauchemardesque à chaque nouveau format

MAINTENANT (LLM multimodal):
✅ Quelques prompts bien rédigés
✅ Schémas Pydantic pour structurer la sortie
✅ Validation Python pour les règles métier
✅ Zéro annotation, zéro entraînement
✅ Fonctionne out-of-the-box sur nouveaux formats
✅ Temps de développement: quelques heures vs plusieurs mois!

🎯 Le LLM comprend:
   - La structure visuelle du document
   - Le contexte (ex: "facture" vs "justificatif")
   - Les cases cochées visuellement
   - Les formats de dates variables
   - Les variations de mise en page
   """)


def main():
    """Point d'entrée principal."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <chemin_document>        # Traiter un document")
        print("  python main.py --folder <chemin_dossier> # Traiter un dossier complet")
        return

    if sys.argv[1] == "--folder":
        if len(sys.argv) < 3:
            print("Erreur: spécifiez le chemin du dossier")
            return
        demo_dossier_complet(sys.argv[2])
    else:
        demo_document_unique(sys.argv[1])


if __name__ == "__main__":
    main()
