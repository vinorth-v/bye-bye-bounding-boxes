"""Tests pour les schémas KYC."""

from datetime import date, timedelta

from chains.schemas import (
    RIB,
    CarteIdentite,
    DossierKYC,
    JustificatifDomicile,
    Passeport,
    Sexe,
    TypeJustificatifDomicile,
)


class TestCarteIdentite:
    """Tests pour le schéma CarteIdentite."""

    def test_cni_valide(self):
        """Test d'une CNI valide."""
        cni = CarteIdentite(
            numero_document="123456789012",
            nom="MARTIN",
            prenom="Jean",
            sexe=Sexe.M,
            date_naissance=date(1990, 5, 15),
            lieu_naissance="Paris (75)",
            date_emission=date(2020, 1, 1),
            date_expiration=date(2030, 1, 1),
            nationalite="FRA",
        )

        assert cni.est_valide is True
        assert cni.numero_document == "123456789012"

    def test_cni_expiree(self):
        """Test d'une CNI expirée."""
        cni = CarteIdentite(
            numero_document="123456789012",
            nom="MARTIN",
            prenom="Jean",
            sexe=Sexe.M,
            date_naissance=date(1990, 5, 15),
            lieu_naissance="Paris (75)",
            date_emission=date(2010, 1, 1),
            date_expiration=date(2020, 1, 1),
            nationalite="FRA",
        )

        assert cni.est_valide is False

    def test_nom_uppercase(self):
        """Test que le nom est automatiquement mis en majuscules."""
        cni = CarteIdentite(
            numero_document="123456789012",
            nom="martin",
            prenom="Jean",
            date_naissance=date(1990, 5, 15),
            date_emission=date(2020, 1, 1),
            date_expiration=date(2030, 1, 1),
            nationalite="FRA",
        )

        assert cni.nom == "MARTIN"


class TestPasseport:
    """Tests pour le schéma Passeport."""

    def test_passeport_champs_optionnels(self):
        """Test des champs optionnels du passeport (statut marital, adresse, lieu de délivrance)."""
        pp = Passeport(
            numero_passeport="24AX12345",
            nom="MARTIN",
            prenom="Jean",
            sexe=Sexe.M,
            date_naissance=date(1990, 5, 15),
            lieu_naissance="Paris",
            nationalite="FRA",
            statut_marital="marié",
            date_emission=date(2020, 1, 1),
            date_expiration=date(2030, 1, 1),
            autorite_emission="Préfecture de Paris",
            lieu_delivrance="Paris",
            adresse="10 rue de la Paix, 75001 Paris",
            mrz_ligne1="P<FRAMARTIN<<JEAN<<<<<<<<<<<<<<<<<<<<<<<<<",
            mrz_ligne2="24AX12345FRA9005159M3001012<<<<<<<<<<<<<<00",
        )
        assert pp.statut_marital == "marié"
        assert pp.adresse == "10 rue de la Paix, 75001 Paris"
        assert pp.lieu_delivrance == "Paris"
        assert pp.mrz_ligne1.startswith("P<FRAMARTIN")
        assert pp.mrz_ligne2.startswith("24AX12345FRA")

    def test_passeport_valide(self):
        """Test d'un passeport valide."""
        pp = Passeport(
            numero_passeport="24AX12345",
            nom="MARTIN",
            prenom="Jean",
            sexe=Sexe.M,
            date_naissance=date(1990, 5, 15),
            lieu_naissance="Paris",
            nationalite="FRA",
            date_emission=date(2020, 1, 1),
            date_expiration=date(2030, 1, 1),
            autorite_emission="Préfecture de Paris",
        )

        assert pp.numero_passeport == "24AX12345"
        assert pp.est_valide is True

    def test_passeport_expire(self):
        """Test d'un passeport expiré."""
        pp = Passeport(
            numero_passeport="24AX12345",
            nom="MARTIN",
            prenom="Jean",
            date_naissance=date(1990, 5, 15),
            nationalite="FRA",
            date_emission=date(2010, 1, 1),
            date_expiration=date(2020, 1, 1),
        )

        assert pp.est_valide is False


class TestJustificatifDomicile:
    """Tests pour le schéma JustificatifDomicile."""

    def test_justificatif_recent(self):
        """Test d'un justificatif récent (< 3 mois)."""
        date_recente = date.today() - timedelta(days=30)

        justif = JustificatifDomicile(
            type_document=TypeJustificatifDomicile.UTILITY_BILL,
            nom_complet="Jean MARTIN",
            adresse_ligne1="10 rue de la Paix",
            code_postal="75001",
            ville="Paris",
            date_document=date_recente,
            emetteur="EDF",
        )

        assert justif.est_recent is True

    def test_justificatif_ancien(self):
        """Test d'un justificatif trop ancien (> 3 mois)."""
        date_ancienne = date.today() - timedelta(days=100)

        justif = JustificatifDomicile(
            type_document=TypeJustificatifDomicile.UTILITY_BILL,
            nom_complet="Jean MARTIN",
            adresse_ligne1="10 rue de la Paix",
            code_postal="75001",
            ville="Paris",
            date_document=date_ancienne,
            emetteur="EDF",
        )

        assert justif.est_recent is False

    def test_justificatif_tax_notice(self):
        """Test avec un avis d'imposition."""
        justif = JustificatifDomicile(
            type_document=TypeJustificatifDomicile.TAX_NOTICE,
            nom_complet="Jean MARTIN",
            adresse_ligne1="10 rue de la Paix",
            code_postal="75001",
            ville="Paris",
            date_document=date.today() - timedelta(days=10),
        )

        assert justif.type_document == TypeJustificatifDomicile.TAX_NOTICE


class TestRIB:
    """Tests pour le schéma RIB."""

    def test_rib_creation(self):
        """Test de création d'un RIB."""
        rib = RIB(
            nom_titulaire="MARTIN",
            numero_compte="00001234567",
            code_guichet="00810",
            iban="FR7610278060740002014820115",
            bic="BNPAFRPP",
            nom_banque="BNP Paribas",
        )

        assert rib.iban == "FR7610278060740002014820115"
        assert rib.nom_titulaire == "MARTIN"
        assert rib.iban_valide is True

    def test_iban_normalisation(self):
        """Test normalisation de l'IBAN avec espaces."""
        rib = RIB(
            nom_titulaire="MARTIN",
            iban="FR76 1027 8060 7400 0201 4820 115",
            nom_banque="BNP Paribas",
        )

        assert rib.iban == "FR7610278060740002014820115"

    def test_iban_invalide_checksum(self):
        """Test qu'un IBAN avec mauvais checksum est marqué invalide."""
        rib = RIB(
            nom_titulaire="MARTIN",
            iban="FR0000000000000000000000000",
            nom_banque="BNP Paribas",
        )

        assert rib.iban_valide is False


class TestDossierKYC:
    """Tests pour le schéma DossierKYC complet."""

    def test_dossier_creation(self):
        """Test de création d'un dossier KYC complet."""
        cni = CarteIdentite(
            numero_document="123456789012",
            nom="MARTIN",
            prenom="Jean",
            sexe=Sexe.M,
            date_naissance=date(1990, 5, 15),
            lieu_naissance="Paris (75)",
            date_emission=date(2020, 1, 1),
            date_expiration=date(2030, 1, 1),
            nationalite="FRA",
        )

        justif = JustificatifDomicile(
            type_document=TypeJustificatifDomicile.UTILITY_BILL,
            nom_complet="Jean MARTIN",
            adresse_ligne1="10 rue de la Paix",
            code_postal="75001",
            ville="Paris",
            date_document=date.today() - timedelta(days=30),
            emetteur="EDF",
        )

        rib = RIB(
            nom_titulaire="MARTIN",
            numero_compte="00001234567",
            code_guichet="00810",
            iban="FR7610278060740002014820115",
            bic="BNPAFRPP",
            nom_banque="BNP Paribas",
        )

        dossier = DossierKYC(document_identite=cni, justificatif_domicile=justif, rib=rib)

        assert dossier.document_identite.nom == "MARTIN"
        assert dossier.justificatif_domicile.est_recent is True
        assert dossier.statut_kyc == "PENDING"

    def test_dossier_statut_rejet(self):
        """Test d'un dossier avec statut rejeté."""
        cni = CarteIdentite(
            numero_document="123456789012",
            nom="MARTIN",
            prenom="Jean",
            date_naissance=date(1990, 5, 15),
            date_emission=date(2020, 1, 1),
            date_expiration=date(2030, 1, 1),
            nationalite="FRA",
        )

        justif = JustificatifDomicile(
            type_document=TypeJustificatifDomicile.UTILITY_BILL,
            nom_complet="Jean DUPONT",  # Nom différent
            adresse_ligne1="10 rue de la Paix",
            code_postal="75001",
            ville="Paris",
            date_document=date.today() - timedelta(days=30),
        )

        rib = RIB(
            nom_titulaire="MARTIN",
            iban="FR7610278060740002014820115",
            nom_banque="BNP Paribas",
        )

        dossier = DossierKYC(
            document_identite=cni,
            justificatif_domicile=justif,
            rib=rib,
            statut_kyc="REJECTED",
            raisons_rejet=["Incohérence de nom entre les documents"],
        )

        assert dossier.statut_kyc == "REJECTED"
        assert len(dossier.raisons_rejet) > 0
