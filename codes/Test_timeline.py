import matplotlib.pyplot as plt
import csv
import sys

# Rajouter les times codes sur le graph final - Voir pour la lisibilité !!!

def lecture(file_name):
    datas=[]
    with open(file_name,"r",encoding="utf-8") as file:
        doc=csv.reader(file)
        for line in doc:
            datas.append(line)

    colonnes=[convertion(extraction_colonne(datas,i+1)) for i in range(len(datas[0])-1)]

    full_time=datas[1][-1].split(":")
    total_sec=int(full_time[0])*3600+int(full_time[1])*60+int(full_time[2])

    return colonnes,total_sec,datas[0]

def extraction_colonne(datas,colonne_num):
    assert colonne_num<=len(datas[0]) # La première colonne est la colonne 1.
    colonne_extracted=[]
    for data in datas:
        colonne_extracted.append(data[colonne_num-1])
    return colonne_extracted

def convertion(colonne):
    if len(colonne)%3 == 0:
        colonne.append("")
    converted_colonne=[]
    for i in range (1,len(colonne),3):
        if colonne[i]!="":
            split1=colonne[i].split(":")
            split2=colonne[i+1].split(":")
            seconde1=int(split1[0])*3600+int(split1[1])*60+int(split1[2])
            seconde2=int(split2[0])*3600+int(split2[1])*60+int(split2[2])
            converted_colonne.append((seconde1,seconde2-seconde1,colonne[i+2]))
    return converted_colonne

def definition_bloc(colonne,annot=None):
    bloc=[]
    if not annot:
        for zone in colonne:
            bloc.append((zone[0],zone[1]))
    else:
        for i in range (len(colonne)):
            for j in range (len(colonne[i])):
                if colonne[i][j][2] != "":
                    bloc.append((colonne[i][j][0]+colonne[i][j][1],colonne[i][j][2]))
    return bloc

columns,total_seconds,titles=lecture(sys.argv[1])

segments=[definition_bloc(columns[i]) for i in range(len(columns))]
segments_gris = [(0,total_seconds)]
annotation = definition_bloc(columns,True)

for i in range (len(segments)):
    duree_total=0
    for rectangle in segments[i]:
        duree_total+=rectangle[1]
    duree_segment=f"{duree_total//60}:{duree_total%60}"
    print(f"Durée {titles[i]}: {duree_segment}")

y_position = 1 # Ecart entre la timeline et l'axe
bar_height = 5 # Hauteur de la timeline

fig, ax = plt.subplots(figsize=(30, 3))

ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylim(0, 20)
ax.set_xlim(0, total_seconds)

ax.set_xticks([0, total_seconds])
ax.set_xticklabels(["00:00", "10:40"])

color_list=["#800080", "#008B8B", "#FF7F50", "#DAA520", "#B22222", "#4682B4"] # Violet, Turquoise, Corail, Ocre, Rouge brique, Bleu acier
assert len(color_list)>=len(segments) # Pour avoir assez de couleur à dessiner
ax.broken_barh(segments_gris, (y_position, bar_height), facecolors="#D3D3D3") # Gris
for i in range (len(segments)):
    ax.broken_barh(segments[i], (y_position, bar_height), facecolors=color_list[i])

for annot in annotation :
    ax.annotate(annot[1],xy=(annot[0], y_position + bar_height),xytext=(annot[0], y_position + bar_height + 3),arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),ha='center')

plt.tight_layout()

plt.savefig("outputs/timeline_exemple.png")
print("\nImage enregistrée !")
plt.show()
