# Packages
import numpy as np
import os

# Path to txt files
path_to_d1 = ""

########
# Data #
########
# Preallocation
data = {}

# Loading .txt files
for i in range(1,1001):
  with open(path_to_d1 + 'Domain01/{0}.txt'.format(i)) as f:
    lines = f.readlines()
    # Retrieve domain, class and user ids
    dom = lines[0].split(" ")[-1][:1]
    cla = lines[1].split(" ")[-1].split("\n")[0]
    use = lines[2].split(" ")[-1].split("\n")[0]

    # Check for right domain
    if dom != "1":
      print("Found data for domain {0} in file {1}".format(dom,i))
    # Check if the order of data insertion is the same for all files
    if lines[4].split('\n')[0].split(',') != ["<x>","<y>","<z>","<t>"]:
      print("Different order found in data insertion at index {0}".format(i))

    # Retrieve x, y, z and t data
    lines = lines[5:]
    x = []
    y = []
    z = []
    t = []
    for line in lines:
      line = line.split('\n')[0].split(',')
      x.append(float(line[0]))
      y.append(float(line[1]))
      z.append(float(line[2]))
      t.append(float(line[3]))
    # Insert data in the main dico
    if use not in data.keys():
      data[use] = {}
      data[use][cla] = {'x':[x], 'y':[y], 'z':[z], 't':[t]}
    else:
      if cla not in data[use].keys():
        data[use][cla] = {'x':[x], 'y':[y], 'z':[z], 't':[t]}
      else:
        data[use][cla]['x'].append(x)
        data[use][cla]['y'].append(y)
        data[use][cla]['z'].append(z)
        data[use][cla]['t'].append(t)

#######
# DTW #
#######

# Dynamic time warping

def dtw(user='1',k=1,dist='dtwi',data=data):
  """
  INPUTS: user, k, dist, data
    user --> str corresponding to the user left out for cross-validation
    k --> int corresponding to the number of K nearest neighbours asked to provide
    for each sequence
    dist --> str = 'euclidian' OR 'dtwi' OR 'dtwd' according to the type of distance to use
    data --> a dict corresponding to the data to use in the model (involving the
    user to leave out). Order of the keys: user > class > coordinates
  OUTPUTS: list_out
  """
  ###########################
  # Complementary functions #
  ###########################

  def comp_dtwi(c_test,c_train):
    """
    INPUTS: c_test, c_train
    c_test --> list of int/float
    c_train --> list of int/float
    OUTPUTS: dtw distance between the two 1D time series
    """
    # Preallocation
    matrix = np.zeros((len(c_test)+1,len(c_train)+1))
    # Filling the fist line and the first column with infinity values
    matrix[0,0] = 0
    for i in range(1,len(c_test)+1):
      matrix[i,0] = np.inf
    for j in range(1,len(c_train)+1):
      matrix[0,j] = np.inf
    # Filling the rest of the matrix
    for i in range(1,len(c_test)+1):
      for j in range(1,len(c_train)+1):
        matrix[i,j] = (c_test[i-1]-c_train[j-1])**2 + np.min([matrix[i-1,j-1],matrix[i-1,j],matrix[i,j-1]])
    # Computing the value with the best backward path
    i = len(c_test)
    j = len(c_train)
    path = []
    while i > 1 or j > 1:
      path.append(matrix[i,j])
      next = np.min([matrix[i-1,j-1],matrix[i-1,j],matrix[i,j-1]])
      if next == matrix[i-1,j-1]:
        i -= 1
        j -= 1
      elif next == matrix[i,j-1]:
        j -= 1
      elif next == matrix[i-1,j]:
        i -= 1
    # Add last value
    path.append(matrix[1,1])
    # Return sum of the path
    return np.sqrt(sum(path))

  def comp_dtwd(c_test,c_train):
    """
    INPUTS: c_test, c_train
    c_test --> list of lists of int/float
    c_train --> list of lists of int/float
    OUTPUTS: dtwd distance between the two multi-dimensional time series
    """
    # Preallocation
    matrix = np.zeros((len(c_test[0])+1,len(c_train[0])+1))
    # Filling the fist line and the first column with infinity values
    matrix[0,0] = 0
    for i in range(1,len(c_test[0])+1):
      matrix[i,0] = np.inf
    for j in range(1,len(c_train[0])+1):
      matrix[0,j] = np.inf
    # Filling the rest of the matrix
    for i in range(1,len(c_test[0])+1):
      for j in range(1,len(c_train[0])+1):
        # d(i,j) is computed as the sum of the squared euclidian distance for all dimensions
        distij = []
        for m in range(len(c_test)):
          distij.append((c_test[m][i-1]-c_train[m][j-1])**2)
        matrix[i,j] = sum(distij) + np.min([matrix[i-1,j-1],matrix[i-1,j],matrix[i,j-1]])
    # Computing the value with the best backward path
    i = len(c_test)
    j = len(c_train)
    path = []
    while i > 1 or j > 1:
      path.append(matrix[i,j])
      next = np.min([matrix[i-1,j-1],matrix[i-1,j],matrix[i,j-1]])
      if next == matrix[i-1,j-1]:
        i -= 1
        j -= 1
      elif next == matrix[i,j-1]:
        j -= 1
      elif next == matrix[i-1,j]:
        i -= 1
    # Add last value
    path.append(matrix[i,i])
    # Return sum of the path
    return np.sqrt(sum(path))

  def samples_dist(type,c_test,c_train):
    """
    INPUTS: type, c_test, c_train
    type --> str = 'euclidian' OR 'dtwi' OR 'dtwd' according to the type of distance to compute
    c_test --> list of float [x,y,z] of testing sample
    c_train --> list of float [x,y,z] of training sample
    OUTPUT: overall distance between the two samples
    """
    if type == 'euclidian':
      # By definition, euclidian distance cannot take into account the differences
      # in lengths or in timesteps of the two time series. The test sample is set
      # as the refence sequence.
      dist = 0
      if len(c_test[0]) <= len(c_train[0]):
        for i in range(len(c_test[0])):
          dist += np.sqrt((c_test[0][i]-c_train[0][i])**2 +
                          (c_test[1][i]-c_train[1][i])**2 +
                          (c_test[2][i]-c_train[2][i])**2)
      else:
        for i in range(len(c_train[0])):
          dist += np.sqrt((c_test[0][i]-c_train[0][i])**2 +
                          (c_test[1][i]-c_train[1][i])**2 +
                          (c_test[2][i]-c_train[2][i])**2)
      return dist
    elif type == 'dtwi':
      # dtwi is the sum of the distances computed independently for all dimensions
      dtwi = []
      for coord in range(len(c_test)):
        dtwi.append(comp_dtwi(c_test[coord],c_train[coord]))
      return sum(dtwi)
    elif type == 'dtwd':
      return comp_dtwd(c_test,c_train)

  #############
  # Main part #
  #############

  # Preallocation
  dico = {}
  coord = ['x','y','z']
  users_train = []
  for u in data.keys():
    if u != user:
      users_train.append(u)
  # Number of iterations
  it_tot = 100
  it = 1
  # Preallocation of the output
  list_out = []

  # Iterate through the testing classes
  for key in data[user].keys():
    dico[key] = []
    # Iterate through the testing samples
    for samp in range(len(data[user][key]['x'])):
      x_test = data[user][key]['x'][samp]
      y_test = data[user][key]['y'][samp]
      z_test = data[user][key]['z'][samp]

      # Preallocation with fake values
      knn_d = []
      indices = {}
      for v in range(k):
        knn_d.append(1e10)
        indices[str(v)] = [-1,-1,-1]     # [user, class, samp]
      knn_dist = np.array(knn_d)

      # Iterate through the users of the training set
      for user_tr in users_train:
        # Iterate through the training classes
        for key_tr in data[user].keys():
          # Iterate through the training samples
          for iter in range(len(data[user_tr][key_tr]['x'])):

            # Compare training and testing samples forward and backward in time
            # At the end, only the best of the two will be compared to the current knn
            dtime = []
            compo = []
            for tim in range(2):
              # Training sample
              if tim == 0:
                x_train = data[user_tr][key_tr]['x'][iter]
                y_train = data[user_tr][key_tr]['y'][iter]
                z_train = data[user_tr][key_tr]['z'][iter]
              else:
                x_train = data[user_tr][key_tr]['x'][iter]
                x_train.reverse()
                y_train = data[user_tr][key_tr]['y'][iter]
                y_train.reverse()
                z_train = data[user_tr][key_tr]['z'][iter]
                z_train.reverse()
              # Compute the distance between the testing and the training samples
              dtime.append(samples_dist(dist,[x_test,y_test,z_test],[x_train,y_train,z_train]))
              if tim == 0:
                compo.append([user_tr,key_tr,str(iter),"forward"])
              else:
                compo.append([user_tr,key_tr,str(iter),"backward"])
            # Determine best output between forward and backward in time
            ind = np.where(np.array(dtime) == min(np.array(dtime)))[0][0]
            d = dtime[ind]
            comp = compo[ind]
            # Compare to the current knn
            if d < max(knn_dist):
              # If smaller distance, replace max value in knn
              ind2 = np.where(knn_dist == max(knn_dist))[0][0]
              knn_dist[ind2] = d
              indices[str(ind2)] = comp

      # Print the current progress
      print("Iteration {0}/{1} done.".format(it,it_tot))
      it += 1
      # Store the sample's outputs in the output list
      list_out.append([indices, knn_dist.tolist()])

  return list_out

#############
# Computing #
#############

# Output path
path_out = "path/out/"

for dist_type in ["euclidian", "dtwd", "dtwi"]:
    for u in range(1,11):
        user = str(u)
        result = dtw(user=user, k=3, dist=dist_type, data=data)

        # Writing the output
        with open(path_out + "res_user={0}_type={1}.txt".format(user,dist_type), "w") as output:
            output.write("[")
            for i in range(len(result)):
                if i != len(result)-1:
                    output.write(str(result[i])+",\n")
                else:
                    output.write(str(result[i]))
            output.write("]")
