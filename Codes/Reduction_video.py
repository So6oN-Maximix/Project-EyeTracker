from pathlib import Path
import csv
import subprocess
import imageio_ffmpeg


def prendre_extrait(entree, t1, t2, sortie):
    """
    Permet de découper la vidéo en fonction des timecodes donnés

    Args :\n
        entree (str) : Chemin vers la vidéo de base\n
        t1, t2 (float) : Timecodes de début et de fin en secondes\n
        sortie (str) : Chemin où la vidéo découpée sera sauvegardée
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        entree,
        "-ss",
        str(t1),
        "-to",
        str(t2),
        "-c",
        "copy",
        sortie,
        "-y",
    ]
    subprocess.run(command, check=True)
    print(f"Video MP4 enregistrée : {sortie}")


def lecture(file_name, file):
    """
    Permet la lecture d'un fichier CSV dans un format particulier (celui données par le logiciel)

    Args :\n
        file_name (str) : Nom du fichier CSV d'entrée\n
        file (str) : Nom du dossier dans lequel il fait aller chercher le fichier donné

    Returns :\n
        Liste : Retourne une liste de str avec toutes les données du CSV
    """
    datas = []
    with open(PROTECH_PATH / file / file_name, "r", encoding="latin-1") as f:
        doc = csv.reader(f)
        if "decoupe" not in file_name:
            for _ in range(6):
                next(doc)
        for line in doc:
            datas.append(line)
    return datas


def convertion_secounds(timecode):
    """
    Convertit un timecode XX:XX:XX:XXX en secondes

    Args :\n
        timecode (str) : Timecode à convertir
    """
    parts = timecode.split(":")
    return (
        3600 * int(parts[0]) + 60 * int(parts[1]) + int(parts[2]) + int(parts[3]) / 1000
    )


def extraction(tableau, tc1, tc2):
    """
    Sert à couper la CSV de départ en fonction de comment on coupe la vidéo

    Args :\n
        tableau (Liste) : Tableau CSV complet que l'on veut couper\n
        tc1, tc2 (float) : Timecodes d'entrée et de sortie  en secondes pour découper le CSV en fonction d'eux

    Returns :\n
        Liste : Tableau de données coupé\n
        int : Frame initiale\n
        int : Frame finale
    """
    start = 1
    end = 1
    for i in range(1, len(tableau)):
        if float(tableau[i][3][:5]) < tc1:
            start += 1
        if float(tableau[i][3][:5]) <= tc2:
            end += 1
    donnees = [tableau[0]]
    for line in tableau[start:end]:
        donnees.append(line)
    return donnees, int(tableau[start][2]), int(tableau[end][2])


def sauvegarde(file_name, lines):
    """
    Sert à sauvegarder un fichier CSV dans un dossier spécifique

    Args :\n
        file_name (str) : Nom du fichier CSV à sauvegarder\n
        lines (Liste) : Données du fichier
    """
    input_path = PROTECH_PATH / "Outputs" / file_name
    with open(input_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(lines)
    print(f"Fichier CSV enregistré : {PROTECH_PATH/'Outputs'/file_name}")


def reduction(video, fichier_csv, tc1, tc2):
    """
    Fonction main de découpage récupérant les données de la console

    Args :\n
        video (str) : Nom du fichier vidéo\n
        fichier_csv (str) : Nom du fichier CSV associé\n
        tc1, tc2 (str) : Timecodes de départ et de fin pour la découpe (au format XX:XX:XX:XXX)

    Returns :\n
        str : Nom du fichier CSV de sortie\n
        str : Nom du fichier vidéo de sortie
    """
    print("=========== DECOUPAGE DONNES VIDEO ===========")
    file = lecture(fichier_csv, "Datas")
    tc01, tc02 = convertion_secounds(tc1), convertion_secounds(tc2)

    output_name_csv = video.split(".")[0] + "_decoupe.csv"
    new_tabl, start_frame, end_frame = extraction(file, tc01, tc02)
    sauvegarde(output_name_csv, new_tabl)

    output_name_mp4 = video.split(".")[0] + "_decoupe.mp4"
    prendre_extrait(
        PROTECH_PATH / "Datas" / video,
        tc01,
        tc02,
        PROTECH_PATH / "Outputs" / output_name_mp4,
    )
    print("========= FIN DECOUPAGE DONNES VIDEO =========\n")
    return output_name_csv, output_name_mp4


PROTECH_PATH = Path(__file__).resolve().parent.parent
