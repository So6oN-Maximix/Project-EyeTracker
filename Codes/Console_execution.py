import re
import sys
import shutil
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("Logger_Report.log", mode="w", encoding="utf8")],
)
logger = logging.getLogger("EyeTracker_Console")

PROTECH_PATH = Path(__file__).resolve().parent.parent


def verifier_fichier_video(chemin, type):
    """
    Verifie si le chemin du fichier en quesiton est soit existant, soit dans le format demandé par l'argument

    Args :
        chemin (str) : Le chemin auquel est situé le fichier que l'on souhaite vérifier
        type (str) : Type de donnée que doit être notre fichier

    Returns
        str : Etat du test effectué, avec l'erreur associé le cas échéant
    """
    fichier = Path(chemin)

    if not fichier.exists():
        return "Introuvable"
    if type == "video" or type == "vidéo":
        if fichier.suffix != ".mp4":
            return "Format Incorrect - Demandé MP4"
    elif type == "csv":
        if fichier.suffix != ".csv":
            return "Format Incorrect - Demandé CSV"
    return "Valide"


def clear_folder(folder_path):
    dossier = Path(folder_path)

    if not dossier.exists():
        logger.warning(f"\nLe dossier {dossier} n'existe pas, création en cours...")
        dossier.mkdir(parents=True, exist_ok=True)
        # On crée le sous-dossier Graphs immédiatement
        (dossier / "Graphs").mkdir(parents=True, exist_ok=True)
        return

    logger.info(f"\nNettoyage du dossier : {dossier}")
    for item in dossier.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            logger.error(f"Impossible de supprimer {item.name} : {e}")

    dossier_graphs = dossier / "Graphs"
    dossier_graphs.mkdir(parents=True, exist_ok=True)


def verifier_format_timecode(tc):
    pattern = r"^\d{2}:\d{2}:\d{2}:\d{3}$"
    if not re.match(pattern, tc):
        return False
    return True


def verifier_existence_fichier(chemin):
    p = Path(chemin)
    if not p.exists() or not p.is_file():
        logger.error(f"Erreur : Le fichier '{chemin}' n'existe pas.")
        return False
    return True


def demander_saisie(message, validateur=None, erreur_msg="Entrée invalide"):
    """Fonction utilitaire pour demander une saisie utilisateur interactive"""
    while True:
        valeur = input(message).strip()
        if not valeur:
            logger.error("Erreur : La valeur ne peut pas être vide.")
            continue

        if validateur:
            if validateur(valeur):
                return valeur
            else:
                print(erreur_msg)
        else:
            return valeur


def main():
    print(
        "\n================================ CONFIGURATION ================================"
    )

    video_path = demander_saisie(
        "Entrez le chemin du fichier Vidéo (.mp4) : ",
        verifier_existence_fichier,
        "Fichier introuvable.",
    )
    csv_path = demander_saisie(
        "Entrez le chemin du fichier Gaze (.csv) : ",
        verifier_existence_fichier,
        "Fichier introuvable.",
    )

    name_mp4 = Path(video_path).stem
    name_csv = Path(csv_path).stem
    logger.debug(f"Egalite des noms : {name_mp4==name_csv}")

    if name_mp4 != name_csv:
        print(f"\nATTENTION : Les noms diffèrent ({name_mp4} vs {name_csv}).")
        confirm = input("Voulez-vous continuer quand même ? (o/n) : ")
        if confirm.lower() != "o":
            logger.info("\nAnnulation.")
            sys.exit(0)

    tc_start = demander_saisie(
        "Timecode Début (HH:MM:SS:ms) : ",
        verifier_format_timecode,
        "Format invalide. Attendu : XX:XX:XX:XXX",
    )
    tc_end = demander_saisie(
        "Timecode Fin (HH:MM:SS:ms) : ",
        verifier_format_timecode,
        "Format invalide. Attendu : XX:XX:XX:XXX",
    )
    csv_tc_path = demander_saisie(
        "Entrez le chemin du fichier TimeCodes Timeline (.csv) : ",
        verifier_existence_fichier,
        "Fichier introuvable.",
    )

    # --- Nettoyage et Lancement ---
    try:
        # Adaptation du chemin Outputs comme dans l'original
        output_path = PROTECH_PATH / "Outputs"
        clear_folder(output_path)
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des dossiers : {e}")
        # On continue quand même ou on arrête selon le besoin

    print(
        "\n================================ DEBUT DE L'ANALYSE ================================"
    )
    print(f"\nVidéo   : {Path(video_path).name}")
    print(f"\nGaze    : {Path(csv_path).name}")
    print(f"\nTC CSV  : {Path(csv_tc_path).name}")
    print(f"\nSegment : {tc_start} -> {tc_end}")
    print(
        "\n===================================================================================="
    )

    # Retourne les noms de fichiers comme demandé par la structure de l'ancien script
    return (
        Path(video_path).name,
        Path(csv_path).name,
        [tc_start, tc_end],
        Path(csv_tc_path).name,
    )
