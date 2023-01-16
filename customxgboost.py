#Jonas Gabirot
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from operator import itemgetter

df = pd.read_csv("diabetes.csv")
df_train, df_test = train_test_split(df, test_size=0.3)
df_train_label = df_train["Outcome"]
df_train_data = df_train.drop("Outcome", axis=1)

df_test_label = df_test["Outcome"]
df_test_data = df_test.drop("Outcome", axis=1)

df_train_label = df_train_label.to_numpy()
df_train_data = df_train_data.to_numpy()


df_test_label = df_test_label.to_numpy()
df_test_data =df_test_data.to_numpy()

def calculg_h(predictlist,labellist):
    
    g_list = []
    h_list = []
    for predict, label in zip(predictlist,labellist):
        if (predict <= 0.00001) or (1-predict <= 0.00001):
            g = 0.00001
            h = 0.00001
        else:
            g = (-label/predict)+((1-label)/(1-predict))
            h = (label/predict**2)+((1-label)/(1-predict)**2)
        g_list.append(g)
        h_list.append(h)

    return g_list, h_list


def optimisation_noeud(index, p, listg,listh, value):
    feuille = True
    result_L = None
    result_R = None
    boundarydata = None
    
    sum_g = sum(listg)
    sum_h = sum(listh)
    

    if p>0:
        if value == None:
            value = (-1/2)*((sum_g)**2)/sum_h
        
        minimum = None
        minset = []
        minvalue = []
        
        
        for column_index, column in zip(range(8),df_train_data.T):
            
            max_col = int(np.ceil(max(column)))
            
            min_col = int(np.floor(min(column)))
            
            step = 1
            if column_index == 5:
                step = 0.1
            if column_index == 6:
                step = 0.001

            borne_list = [x*step for x in range(min_col,max_col)]
            
            for borne in borne_list:
                index_L = [x for x in index if column[x] > borne]
                index_R = [x for x in index if column[x] <= borne]
                if not index_R or not index_L:
                    continue
                listg_R = [listg[i] for i in index_R]
                listg_L = [listg[i] for i in index_L]
                listh_R = [listh[i] for i in index_R]
                listh_L = [listh[i] for i in index_L]
                
            
                value_R = (-1/2)*((sum(listg_R))**2)/sum(listh_R)
                value_L = (-1/2)*((sum(listg_L))**2)/sum(listh_L)

                result = value_R + value_L -value

                if result<0:
                    if minimum == None:
                        minimum = result
                        minset = [index_R, index_L]
                        minvalue = [value_R, value_L]
                        boundarydata = [column_index, borne]

                    elif result<minimum:
                        minimum = result
                        minset = [index_R,index_L]
                        minvalue = [value_R, value_L]
                        boundarydata = [column_index, borne]


        if minimum != None:
            #Pas une feuille, on sépare le noeud en 2. 
            p = p-1
            feuille = False
            result_R, newlistg_R, newlisth_R = optimisation_noeud(minset[0],p,listg,listh,minvalue[0])
            result_L, newlistg_L, newlisth_L = optimisation_noeud(minset[1],p,listg,listh,minvalue[1])
            enum_listg_R = list(zip(minset[0],newlistg_R))
            enum_listg_L = list(zip(minset[1],newlistg_L))
            enum_listh_R = list(zip(minset[0],newlisth_R))
            enum_listh_L = list(zip(minset[1],newlisth_L))
            
            newlistg = enum_listg_R + enum_listg_L
            newlisth = enum_listh_R + enum_listh_L
            
            newlistg = sorted(newlistg, key=itemgetter(0))
            newlistg = [i[1] for i in newlistg]

            newlisth = sorted(newlisth, key=itemgetter(0))
            newlisth = [i[1] for i in newlisth]
           
            
            
            
    
    zeros = 0
    ones = 0      
    for i in index:
        if df_train_label[i] == 0:
            zeros += 1
        else:
            ones += 1
    prob_1 = ones/(zeros +ones)

    if feuille:
        probs = [prob_1 for i in range(len(index))]
        label_feuille = [df_train_label[i] for i in index]
        newlistg, newlisth = calculg_h(probs,label_feuille)
        
        boundarydata = [None,None, prob_1]
        
    else:
        if prob_1 > 0.5:
            prob_1 = 1
        else:
            prob_1 = 0
        boundarydata.append(prob_1)
    

    
    tree = dict(decision = boundarydata, noeud_droite = result_R, noeud_gauche = result_L)
    return tree, newlistg, newlisth


glist = np.random.random(df_train_label.size)
hlist = np.random.random(df_train_label.size)



tree = optimisation_noeud(range(0,537),7,glist,hlist,None)[0]

def find_outcome(x, tree):
    variable, borne, prob = tree["decision"]
    
    if variable == None:
        return prob
    else:
        if x[variable] > borne:
            tree = tree["noeud_gauche"]
        else:
            tree = tree["noeud_droite"]
        return find_outcome(x, tree)



treelist = []
k=20
for l in range(k):
    output = optimisation_noeud(range(0,537),3,glist,hlist,None)
    treelist.append(output[0])
    glist = output[1]
    hlist = output[2]
print("Training Done")


all_predictions = [0 for i in range(df_test_label.size)]
for tree in treelist:
    tree_predictions = []
    for x in df_test_data:  
         
        prediction = find_outcome(x,tree)
        
        tree_predictions.append(prediction)
    
    all_predictions =  [sum(x) for x in zip(all_predictions, tree_predictions)] 

all_predictions = [x/k for x in all_predictions]

def loss(y_hat,y):
    loss = -(y*np.log(y_hat))-((1-y)*np.log(1-y_hat))
    return loss

total_loss = 0
for i, y in zip(range(0,df_test_label.size), df_test_label):
    y_hat = all_predictions[i]
    
    lossi = loss(y_hat,y)
    
    total_loss += lossi

print(total_loss)
print(total_loss/df_test_label.size)



