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
    Verifie que le fichier donné comme argument existe et, le cas échéant, vérifie si le format est bien celui attendu

    Args :\n
        chemin (str) : Chemin vers le fichier dont on souhaite vérifier le format\n
        type (str) : Type supposé du fichier analysé

    Returns\n
        str : Etat du test, et la raison si jamais le test n'est pas concluant
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
    Nettoie le dossier donné en argument

    Args :\n
        folder_path (str) : Chemin vers le dossier que l'on veut nettoyer
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
    Verifie que le timecode est bien dans le format souhaité en utilisant une "regular expression" (REGEX)

    Args :\n
        tc (str): Timcode dont on veut vérifier le format

    Returns :\n
        Boolean : Etat de la vérification
    """
    pattern = r"^\d{2}:\d{2}:\d{2}:\d{3}$"
    if not re.match(pattern, tc):
        return False
    return True


def demander_saisie(message, validateur=None, type=None, erreur_msg="Entrée invalide"):
    """
    Fonction utilisateur qui sert à la demande d'une donnée particulière

    Args :\n
        message (str) : Message qui apparaitera pour demander la donnée en question\n
        valideur (None) : Fonction qui va vérifier la cohérence de la donnée\n
        type (str) : Type de la donnée que l'on demande\n
        erreur_msg (str) : Message qui apparaitera en car d'erreur dans la saisie de la donnée\n

    Returns :\n
        str : Réponse que va donner l'utilisateur
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
    Fonction qui récupère chaque réponse de l'utilisateur avant de les donner à Code_V4.0.py qui va les utiliser comme entrée dans son code

    Returns :\n
        str : Nom du fichier vidéo à analyser\n
        str : Nom du fichier CSV (Fichier Gaze) à analyser\n
        [str, str] : Liste des timecode d'entrée et de sortie (pour la découpe vidéo)\n
        str : Nom du fichier CSV servant à la création de la timeline
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
        erreur_msg="Format invalide. Attendu : XX:XX:XX:XXX",
    )
    tc_end = demander_saisie(
        "Timecode Fin (HH:MM:SS:ms) : ",
        verifier_format_timecode,
        erreur_msg="Format invalide. Attendu : XX:XX:XX:XXX",
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
