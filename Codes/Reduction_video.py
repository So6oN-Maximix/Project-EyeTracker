from pathlib import Path
import csv
import subprocess
import imageio_ffmpeg


def prendre_extrait(entree, t1, t2, sortie):
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
    parts = timecode.split(":")
    return (
        3600 * int(parts[0]) + 60 * int(parts[1]) + int(parts[2]) + int(parts[3]) / 1000
    )


def extraction(tableau, tc1, tc2):
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
    input_path = PROTECH_PATH / "Outputs" / file_name
    with open(input_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(lines)
    print(f"Fichier CSV enregistré : {PROTECH_PATH/'Outputs'/file_name}")


def reduction(video, fichier_csv, tc1, tc2):
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
