# Packages
import numpy as np
import ast

# Importing data
# 2 cases according to time-forward or time-forward/-backward
path_input = "/path/input"
typ = "dtwi"
bf = True
if not bf:
    mdir = path_input + "/trial1"
else:
    mdir = path_input

# Load the results inside a dictionary
res_dico = {}
for i in range(1,11):
    with open(mdir + "/res_user={0}_type={1}.txt".format(i,typ)) as file:
        data = []
        for line in file:
            # Treatment of the dict part
            dico = {}
            dic = line.split("}, [")[0]
            dic = dic.split(":")
            dico['0'] = eval(dic[1])[0]
            dico['1'] = eval(dic[2])[0]
            dico['2'] = eval(dic[3])
            # Treatment of the list part
            liste = []
            lst = line.split("}, [")[1].split(",")
            liste.append(float(lst[0]))
            liste.append(float(lst[1]))
            liste.append(float(lst[2].split("]]")[0]))
            # Finish
            data.append([dico,liste])
        res_dico[str(i)] = data

# Analysis of the results

# For each user left out, 2 cases
    # 1) Use the 1 nearest neighbour
    # 2) Use the 3 nearest neighbours
        # Compute the accuracy over 100 iterations
            # Then compute the average accuracy and its standard devitation
            
# functions definition
def check_3NN(lst,num):
    """
    Return True if at least 2 of the 3 neighbours correspond to the given number
    False otherwise
    """
    dic = lst[0]
    cnt = 0
    for klst in dic.values():
        if int(klst[1]) == num:
            cnt += 1
    if cnt > 1:
        return True
    else:
        return False
    
def check_1NN(lst,num):
    """
    Return True if the nearest neighbour corresponds to the given number
    False otherwise
    """
    dic = lst[0]
    l = np.array(lst[1])
    ind = np.where(l == min(l))[0][0]
    if int(dic[str(ind)][1]) == num:
        return True
    else:
        return False
            
# Overall values
accuracies_1 = []
accuracies_3 = []
            
for user in range(1,11):
    u_dic = res_dico[str(user)]
    nn1 = []
    nn3 = []
    for it in range(100):
        number = it // 10
        lst = u_dic[it]
        nn1.append(check_1NN(lst,number))
        nn3.append(check_3NN(lst,number))
    accuracies_1.append(round(sum(np.array(nn1))/100,4))
    accuracies_3.append(round(sum(np.array(nn3))/100,4))
    
print("Method: {0} - BF = {5}, average accuracies (1NN): {1} | (3NN): {2},\n std (1NN): {3} | (3NN): {4}"
      .format(typ, round(np.mean(accuracies_1),4), round(np.mean(accuracies_3),4), round(np.std(accuracies_1),4), round(np.std(accuracies_3),4), bf))