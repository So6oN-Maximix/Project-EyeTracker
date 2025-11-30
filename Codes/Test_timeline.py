import matplotlib.pyplot as plt
import csv
import sys
from pathlib import Path
from moviepy import VideoFileClip

def lecture(file_name):
    datas=[]
    with open(file_name,"r",encoding="utf-8") as file:
        doc=csv.reader(file)
        for _ in range (3):
            next(doc)
        for line in doc:
            datas.append(line)
    datas01=datas[2][0].split("\n")
    video01, data_name01=datas01[0], datas01[1]
    index=3
    data02=datas[index][0].split("\n")
    while data02==[""]:
        index+=1
        data02=datas[index][0].split("\n")
    video02, data_name02=data02[0], data02[1]
    names=[data_name01,data_name02]

    clip01=VideoFileClip(PROTECH_PATH/"Datas"/video01)
    seconde01=clip01.duration
    clip02=VideoFileClip(PROTECH_PATH/"Datas"/video02)
    seconde02=clip02.duration
    secondes=[seconde01, seconde02]

    nb_datas=(len(datas[0])-2)//2
    titres=[datas[0][2*i+1] for i in range(nb_datas+1)]
    segments=[[] for _ in range(2*(nb_datas+1))]
    for i in range(nb_datas):
        for j in range (3, len(datas)):
            if datas[j][2*i+1]!="" or datas[j][2*i+2]!="":
                debut, fin=convertion_secounds(datas[j][2*i+1]), convertion_secounds(datas[j][2*i+2])
                if j<index:
                    segments[i].append((debut, round(fin-debut,3)))
                else:
                    segments[i+nb_datas+1].append((debut, round(fin-debut,3)))
    for i in range (3,len(datas)):
        if datas[i][2*nb_datas+1]!="":
            if i<index:
                segments[nb_datas].append(convertion_secounds(datas[i][2*nb_datas+1]))
            else:
                segments[2*(nb_datas+1)-1].append(convertion_secounds(datas[i][2*nb_datas+1]))

    return segments, titres, secondes, names

def convertion_secounds(timecode):
    parts=timecode.split(":")
    return 3600*int(parts[0])+60*int(parts[1])+int(parts[2])+int(parts[3])/1000

def list_to_timecode(positions):
    tc0=positions[0]
    labels=[f"{int(tc0//60):02d}:{int(tc0%60):02d}"]
    for i in range (1, len(positions)):
        tc1=positions[i]
        if tc1-tc0<=2:
            labels.append(f"{int(tc1//60):02d}:{int(tc1%60):02d}\n")
        else:
            labels.append(f"{int(tc1//60):02d}:{int(tc1%60):02d}\n")
        tc0=tc1
    return labels

def creation_timeline(file_name):
    all_segments, titles, secondes, names = lecture(PROTECH_PATH/"Datas"/file_name)
    nbr_datas=(len(all_segments)-2)//2

    segments=all_segments[:nbr_datas]+all_segments[nbr_datas+1:len(all_segments)-1]

    print("================== TIMELINES ==================")
    for i in range(nbr_datas):
        duree_total = 0
        for rectangle in segments[i]:
            duree_total += rectangle[1]
        duree_segment = f"{int(duree_total//60):02d}:{int(duree_total%60):02d}"
        print(f"Durée {titles[i]} -> {duree_segment}")
    print("")

    y_position = 1  # Ecart entre la timeline et l'axe
    bar_height = 5  # Hauteur de la timeline

    for i in range (len(names)):
        segments_gris = [(0, secondes[i])]
        annotation = all_segments[(nbr_datas+1)*(i+1)-1]

        fig, ax = plt.subplots(figsize=(30, 3))

        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_ylim(0, 20)
        ax.set_xlim(0, secondes[i])

        positions=[0]
        for j in range (nbr_datas):
            for temps in segments[2*i+j]:
                if temps[0] not in positions:
                    positions.append(temps[0])
                if temps[0]+temps[1] not in positions:
                    positions.append(round(temps[0]+temps[1],3))
        if secondes[i] not in positions:
            positions.append(secondes[i])
        ax.set_xticks(positions)
        ax.set_xticklabels(list_to_timecode(positions))
        ax.tick_params(axis='x', rotation=50, labelsize=9) # Rotation des timecodes de 90°

        color_list = ["#800080", "#008B8B", "#FF7F50", "#DAA520", "#B22222","#4682B4"]  # Violet, Turquoise, Corail, Ocre, Rouge brique, Bleu acier
        assert len(color_list) >= len(segments)  # Pour avoir assez de couleur à dessiner
        ax.broken_barh(segments_gris, (y_position, bar_height), facecolors="#D3D3D3")  # Gris
        for j in range(len(segments)//2):
            if segments[2*i+j]!=[]:
                ax.broken_barh(segments[2*i+j], (y_position, bar_height), facecolors=color_list[j])

        for annot in annotation:
            ax.annotate("Mise en forme", xy=(annot, y_position + bar_height), xytext=(annot, y_position + bar_height + 3),arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), ha='center')

        plt.tight_layout()
        plt.savefig(PROTECH_PATH/"Outputs"/f"Timeline_{names[i]}.png")
        plt.close()
        print(f"Image enregistrée : {PROTECH_PATH/'Outputs'/f'Timeline_{names[i]}.png'}")
    print("================ FIN TIMELINES ================\n")

PROTECH_PATH=Path(__file__).resolve().parent.parent

if __name__ == "__main__":
	creation_timeline(PROTECH_PATH/"Datas"/sys.argv[1])
