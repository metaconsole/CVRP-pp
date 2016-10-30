from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

#r'C:\Users\Paul\Desktop\Python\Metaheuristics\Testcases\vrpnc1.txt'
# Open File and save Coordinates in CoordArr
with open(r'C:\Users\Nero\Master Mathematik\Metaheuristics\Projekt\cvrp_instances\vrpnc1.txt', 'r') as f:
    
    CoordArr = []
    l = list(map(int, f.readline().split()))
    NoCus, VeCap, RoLen, DroTm = l
    for lx in f:
        line = list(map(int, lx.split()))
        CoordArr.append(line)

CoordArr[0].append(0)
# calculate the Distance of two points
def Distance(Arr1,Arr2):
    return sqrt((Arr1[0] - Arr2[0])**2 + (Arr1[1] - Arr2[1])**2)



# Save Distances in n Array, with n-(i+1) entrys for the i-th costumor
DistArr = []
'''
for i in range(NoCus + 1):
    lis = []
    for j in range(i, NoCus + 1):
        if i != j:
            #not saving 0 distance to itself
            lis.append(Distance(CoordArr[i],CoordArr[j]))
    DistArr.append(lis)
'''
for i in range(NoCus + 1):
    lis = []
    for j in range(NoCus + 1):
        lis.append(Distance(CoordArr[i],CoordArr[j]))
    DistArr.append(lis)
    

def GetDist(i,j):
    if j < i:
        return DistArr[j][i]
    else:
        return DistArr[i][j]


#Minimal Distance to a given point
def GetMinDist(Arr):
    Min = Arr[0]
    for i in range(len(Arr)):
        if Arr[i] < Min:
            Min = Arr[i]
    return Min


#returns the nearest, customer that is NOT yet visited
#if two with same distance exist it takes the first
#auskommentiert weil falsch, war erster draft
'''
def NearestCus(i):
    nex = 0
    dis = DistArr[i]
    dis2 = [DistArr[l][i] for l in range(i)]
    SorDis = sorted(dis2+dis)
    k = 0
    nex = (dis2+dis).index(SorDis[k])
    while CusArr[nex]==1:
        k +=1
        if k == len(CusArr)-1:
            return 0
            break
        nex = (dis2+dis).index(SorDis[k])    
    return nex
'''
DstTrueArr = np.array(DistArr)

#gefunden über google, optimal für uns, nutzt numpy funktionen 
def NN(A, start):
    """Nearest neighbor algorithm.
    A is an NxN array indicating distance between N locations
    start is the index of the starting location
    Returns the path and cost of the found solution
    """
    path = [start]
    cost = 0
    N = A.shape[0]
    mask = np.ones(N, dtype=bool)  # boolean values indicating which 
                                   # locations have not been visited
    mask[start] = False

    for i in range(N-1):
        last = path[-1]
        next_ind = np.argmin(A[last][mask]) # find minimum of remaining locations
        next_loc = np.arange(N)[mask][next_ind] # convert to original location
        path.append(next_loc)
        mask[next_loc] = False
        cost += A[last, next_loc]

    return path, cost

# Array of all Customers, 0 if unvisited 1 if visited        
CusArr = [0 for x in range(NoCus+1)]

#auskommentiert weils nicht funktioniert, war mein erster draft an Lösung
'''
# Building Starting Solution
def BuildSol(DistArr):
    VisitedCus = 0
    sol = [0]
    i = 0
    VeLoad = 0
    while VisitedCus < NoCus:
        VecNo = 1
        
        nex = NearestCus(i)
        VeLoad += CoordArr[nex][2]
        sol.append(nex)
        CusArr[nex]=1
        VisitedCus += 1
    return sol, VeLoad
    

'''
'''

        while VeLoad <= VeCap:
            nex = NearestCus(i)
            VeLoad += CoordArr[nex][2]
            sol.append(nex)
            CusArr[nex]=1
            VisitedCus += 1
        VecNo +=1}
            
 '''
sol, VeLoad = NN(DstTrueArr, 0)
sol.append(0)
#solution plotting

x = [ CoordArr[i][0] for i in range(1, NoCus+1)]
y = [ CoordArr[i][1] for i in range(1, NoCus+1)]

xsol = [ CoordArr[i][0] for i in sol]
ysol = [ CoordArr[i][1] for i in sol]

plt.plot(30,40, "rs")
plt.scatter(x,y)
plt.plot(xsol, ysol, "r-")
plt.show()

