"""
Point d'entrée principal du démonstrateur KYC.

Exemple d'utilisation pour la conférence!
"""

import sys

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
        print("\n✅ EXTRACTION RÉUSSIE\n")

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
    else:
        print("\n❌ ÉCHEC DE L'EXTRACTION\n")
        if result.erreurs:
            for erreur in result.erreurs:
                print(f"   - {erreur}")


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
