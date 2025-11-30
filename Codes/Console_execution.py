import re
import sys
import shutil
from pathlib import Path

# On suppose que Test_timeline existe dans le même dossier
# Si ce module n'est pas présent, il faudra adapter l'import
try:
    import Test_timeline
    from Test_timeline import PROTECH_PATH
except ImportError:
    # Fallback pour le test si le module n'est pas là
    print("Attention: Module 'Test_timeline' introuvable. Utilisation d'un chemin par défaut.")
    PROTECH_PATH = Path("./ProTech")

def clear_folder(folder_path):
    dossier = Path(folder_path)

    if not dossier.exists():
        print(f"Le dossier {dossier} n'existe pas, création en cours...")
        dossier.mkdir(parents=True, exist_ok=True)
        # On crée le sous-dossier Graphs immédiatement
        (dossier / "Graphs").mkdir(parents=True, exist_ok=True)
        return

    print(f"Nettoyage du dossier : {dossier}")
    for item in dossier.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            print(f"Impossible de supprimer {item.name} : {e}")

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
        print(f"Erreur : Le fichier '{chemin}' n'existe pas.")
        return False
    return True

def demander_saisie(message, validateur=None, erreur_msg="Entrée invalide"):
    """Fonction utilitaire pour demander une saisie utilisateur interactive"""
    while True:
        valeur = input(message).strip()
        if not valeur:
            print("Erreur : La valeur ne peut pas être vide.")
            continue

        if validateur:
            if validateur(valeur):
                return valeur
            else:
                print(erreur_msg)
        else:
            return valeur

def main():
    print("================================ CONFIGURATION ================================")

    video_path = demander_saisie("Entrez le chemin du fichier Vidéo (.mp4) : ",verifier_existence_fichier,"Fichier introuvable.")
    csv_path = demander_saisie("Entrez le chemin du fichier Gaze (.csv) : ",verifier_existence_fichier,"Fichier introuvable.")

    name_mp4 = Path(video_path).stem
    name_csv = Path(csv_path).stem

    if name_mp4 != name_csv:
        print(f"ATTENTION : Les noms diffèrent ({name_mp4} vs {name_csv}).")
        confirm = input("Voulez-vous continuer quand même ? (o/n) : ")
        if confirm.lower() != 'o':
            print("Annulation.")
            sys.exit(0)

    tc_start = demander_saisie("Timecode Début (HH:MM:SS:ms) : ",verifier_format_timecode,"Format invalide. Attendu : XX:XX:XX:XXX")
    tc_end = demander_saisie("Timecode Fin (HH:MM:SS:ms) : ",verifier_format_timecode,"Format invalide. Attendu : XX:XX:XX:XXX")
    csv_tc_path = demander_saisie("Entrez le chemin du fichier TimeCodes Timeline (.csv) : ",verifier_existence_fichier,"Fichier introuvable.")

    # --- Nettoyage et Lancement ---
    try:
        # Adaptation du chemin Outputs comme dans l'original
        output_path = PROTECH_PATH / "Outputs"
        clear_folder(output_path)
    except Exception as e:
        print(f"Erreur lors du nettoyage des dossiers : {e}")
        # On continue quand même ou on arrête selon le besoin

    print("\n================================ DEBUT DE L'ANALYSE ================================")
    print(f"Vidéo   : {Path(video_path).name}")
    print(f"Gaze    : {Path(csv_path).name}")
    print(f"TC CSV  : {Path(csv_tc_path).name}")
    print(f"Segment : {tc_start} -> {tc_end}")
    print("====================================================================================")

    # Retourne les noms de fichiers comme demandé par la structure de l'ancien script
    return Path(video_path).name, Path(csv_path).name, tc_start, tc_end, Path(csv_tc_path).name
