#James Alford-Golojuch
#Round 2 Random Forest

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#Initializes training data and labels and testing data from given data files
data_train = np.genfromtxt('train.csv', delimiter=",", skip_header=1, dtype=None)
data_test = np.genfromtxt('test.csv', delimiter=",", skip_header=1, usecols=range(1,193), dtype='float64')
test_ids = np.genfromtxt('test.csv', delimiter=',', skip_header=1, usecols=0, dtype='int32')

tempData = []
for x in range(2,194):
    tempData.append(float(data_train[0][x]))
data = [tempData]

data_labels = [data_train[0][1]]
for x in range(1,data_train.size):
    tempLabel = [data_train[x][1]]
    data_labels= np.vstack((data_labels,tempLabel))
    temp = []
    for y in range(2,194):
        temp.append(float(data_train[x][y]))
    data = np.vstack((data,temp))
data_labels = np.ravel(data_labels)

#Initializes the neural network classifier settings
clf = KNeighborsClassifier(n_neighbors=9, weights='distance', algorithm='auto', 
      leaf_size=60, p=2, metric='minkowski')
clf.fit(data,data_labels)
pred = clf.predict_proba(data_test)

#Finalizes array to be written to file
temp1 = [test_ids[0]]
for x in range(0,99):
    temp1.append(pred[0][x])
pred2 = [temp1]
for x in range(1,594):
    temp = []
    temp.append(test_ids[x])
    for y in range(0,99):
        temp.append(pred[x][y])
    pred2 = np.vstack((pred2,temp))
    
np.savetxt("test_labels3.csv", pred2, delimiter=",", 
header = "id,Acer_Capillipes,Acer_Circinatum,Acer_Mono,Acer_Opalus,Acer_Palmatum,Acer_Pictum,Acer_Platanoids,Acer_Rubrum,Acer_Rufinerve,Acer_Saccharinum,Alnus_Cordata,Alnus_Maximowiczii,Alnus_Rubra,Alnus_Sieboldiana,Alnus_Viridis,Arundinaria_Simonii,Betula_Austrosinensis,Betula_Pendula,Callicarpa_Bodinieri,Castanea_Sativa,Celtis_Koraiensis,Cercis_Siliquastrum,Cornus_Chinensis,Cornus_Controversa,Cornus_Macrophylla,Cotinus_Coggygria,Crataegus_Monogyna,Cytisus_Battandieri,Eucalyptus_Glaucescens,Eucalyptus_Neglecta,Eucalyptus_Urnigera,Fagus_Sylvatica,Ginkgo_Biloba,Ilex_Aquifolium,Ilex_Cornuta,Liquidambar_Styraciflua,Liriodendron_Tulipifera,Lithocarpus_Cleistocarpus,Lithocarpus_Edulis,Magnolia_Heptapeta,Magnolia_Salicifolia,Morus_Nigra,Olea_Europaea,Phildelphus,Populus_Adenopoda,Populus_Grandidentata,Populus_Nigra,Prunus_Avium,Prunus_X_Shmittii,Pterocarya_Stenoptera,Quercus_Afares,Quercus_Agrifolia,Quercus_Alnifolia,Quercus_Brantii,Quercus_Canariensis,Quercus_Castaneifolia,Quercus_Cerris,Quercus_Chrysolepis,Quercus_Coccifera,Quercus_Coccinea,Quercus_Crassifolia,Quercus_Crassipes,Quercus_Dolicholepis,Quercus_Ellipsoidalis,Quercus_Greggii,Quercus_Hartwissiana,Quercus_Ilex,Quercus_Imbricaria,Quercus_Infectoria_sub,Quercus_Kewensis,Quercus_Nigra,Quercus_Palustris,Quercus_Phellos,Quercus_Phillyraeoides,Quercus_Pontica,Quercus_Pubescens,Quercus_Pyrenaica,Quercus_Rhysophylla,Quercus_Rubra,Quercus_Semecarpifolia,Quercus_Shumardii,Quercus_Suber,Quercus_Texana,Quercus_Trojana,Quercus_Variabilis,Quercus_Vulcanica,Quercus_x_Hispanica,Quercus_x_Turneri,Rhododendron_x_Russellianum,Salix_Fragilis,Salix_Intergra,Sorbus_Aria,Tilia_Oliveri,Tilia_Platyphyllos,Tilia_Tomentosa,Ulmus_Bergmanniana,Viburnum_Tinus,Viburnum_x_Rhytidophylloides,Zelkova_Serrata", comments='')