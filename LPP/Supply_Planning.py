import pandas as pd
from pulp import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df_inbound = pd.read_csv('df_inprice.csv', index_col = 0)
print(df_inbound)

print("\n")
df_outbound = pd.read_csv('df_outprice.csv', index_col = 0)
print(df_outbound)

print("\n")
df_melt = pd.melt(df_outbound.set_index('from').T.reset_index(), id_vars=['index'], value_vars=['D1', 'D2'])
print("{:,} records".format(len(df_melt)))
print("\n")
print(df_melt.head())


#Plot Outbound Transportation Costs BoxPlot


plt.figure(figsize=(12, 6))
ax = sns.boxplot(x='from', y='value', data=df_melt, color='#99c2a2')
# ax = sns.swarmplot(x="from", y="value", data=df_melt, color='#7d0013')
plt.xlabel('Distribution Center')
plt.ylabel('Outbound Transportation Unit Costs (₹/Carton)')
plt.show()



# Production capacity
df_prod = pd.DataFrame({
    'plant': ['P1','P2'],
    'max': [200, 300],
})[['plant', 'max']]
print(df_prod)


# Cross-Docking Capacity
df_t = pd.DataFrame({
    'DC': ['D1','D2'],
    'CAPACITY': [450, 300]
})[['DC', 'CAPACITY']]
print(df_t)


# Demand
df_demand = pd.read_csv('df_demand.csv', index_col = 0)
print("{:,} total demand".format(df_demand.DEMAND.sum()))
print(df_demand.head())


#Build the Optimization Model
# 1. Initiliaze Class


model = LpProblem("Transhipment_Problem", LpMinimize)

# 2. Define Decision Variables
# Inbound Flows

I = LpVariable.dicts("I", [(i+1,j+1) for i in range(2) for j in range(2)],
                     lowBound=0, upBound=None, cat='Integer') # I(i,j) from plant i for platform j
# Outbound Flows
O = LpVariable.dicts("O", [(i+1,j+1) for i in range(2) for j in range(200)],
                     lowBound=0, upBound=None, cat='Integer') # O(i,j) from platform i for customer j

# 3. Define Objective Function
# Total Transportation Cost

model += lpSum([df_inbound.iloc[i,j+1] * I[i+1,j+1] for i in range(2) for j in range(2)]) + lpSum([df_outbound.iloc[i,j+1] * O[i+1,j+1] for i in range(2) for j in range(200)]) 


# 4. Define Constraints
# Max capacity for plants
# for i in range(5):
#     model += lpSum([I[i+1, j+1] for j in range(2)]) <= df_prod.loc[i,'max']
# Shipment from DCs higher than demand per store

for j in range(200):
    model += lpSum([O[i+1, j+1] for i in range(2)]) >= df_demand.loc[j,'DEMAND']
    
# Conservation of the flow in the local DC (X-Docking Platform)
for p in range(2):
    model += lpSum([I[i+1, p+1] for i in range(2)]) == lpSum([O[p+1, j+1] for j in range(200)])
# Maximum Inbound Capacity in Platform i
# for p in range(2):
#     model += lpSum([I[i+1, p+1] for i in range(5)]) <= df_t.loc[p,'capacity']

# Solve Model
status = model.solve()
#print(LpStatus[status])
#print("Objective: z* = {}".format(value(model.objective)))

# Matrix result
inbound, outbound = np.zeros([2,2]), np.zeros([2,200])
for i in range(2):
    for j in range(2):
#         print(I[i+1, j+1].varValue, I[i+1, j+1].name)
        inbound[i, j] = I[i+1, j+1].varValue
for i in range(2):
    for j in range(200):
#         print(O[i+1, j+1].varValue, O[i+1, j+1].name)
        outbound[i, j] = O[i+1, j+1].varValue




#Results
# Inbound flow
df_resin = pd.DataFrame(data = inbound, index =['P' + str(i+1) for i in range(2)], 
                        columns = ['D' + str(i+1) for i in range(2)]).astype(int)
df_resin.to_csv('df_inbound_flow.csv')
print(df_resin)



# Outbound flow
df_resout = pd.DataFrame(data = outbound, index =['D' + str(i+1) for i in range(2)], columns = ['S' + str(i+1) for i in range(200)])
print(df_resout.T)