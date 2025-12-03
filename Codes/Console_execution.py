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


def verifier_fichier(chemin, type):
    """
    Verifies if the file gives as an argument is existing and if the format is the one that we are searching for

    Args :
        chemin (str) : Path of the file we are verifying
        type (str) : Supposed type of the file we are verifying

    Returns
        str : State of the test, and the reason if it's not complete
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
    """
    Clean the folder given as an argument

    Args :
        folder_path (str) : Path of the file we want to clean
    """
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
    """
    Verify that the timecode is in the correct format by using a regular expression (REGEX)

    Args :
        tc (str): Timcode that we want to verify the format

    Returns :
        Boolean : State of the verification
    """
    pattern = r"^\d{2}:\d{2}:\d{2}:\d{3}$"
    if not re.match(pattern, tc):
        return False
    return True


def demander_saisie(message, validateur=None, type=None, erreur_msg="Entrée invalide"):
    """
    Utilitary function that will asks different parameters to the user and use then to execute Code_V4.0.py

    Args :
        message (str) : Message that will appear on the terminal
        valideur (None) : function that can be used to verify a format
        erreur_msg (str) : Message that will be show if an error appears

    Returns :
        str : The answer of the user
    """
    while True:
        valeur = input(message).strip()
        if not valeur:
            logger.error("Erreur : La valeur ne peut pas être vide.")
            continue

        if validateur and type:
            if validateur(valeur, type):
                return valeur
            else:
                print(erreur_msg)
        else:
            return valeur


def main():
    """
    Function that will be used in Code_V4.0.py to take as input values that this function returns

    Returns :
        str : Name of the MP4 file
        str : Name of the CSV file
        [str, str] : List of timecode (beginning and ending)
        str : Name of the CSV file for the timeline
    """
    print(
        "\n================================ CONFIGURATION ================================"
    )

    video_path = demander_saisie(
        "Entrez le chemin du fichier Vidéo (.mp4) : ",
        verifier_fichier,
        "video",
        "Fichier introuvable.",
    )
    csv_path = demander_saisie(
        "Entrez le chemin du fichier Gaze (.csv) : ",
        verifier_fichier,
        "csv",
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
        verifier_fichier,
        "csv",
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
