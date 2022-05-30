import pandas as pd
import matplotlib.pyplot as plt

# Import
df = pd.read_excel('1-2017.xlsx')
print("{:,} order lines for {:,} orders".format(len(df), df.ORDER_NUMBER.nunique()))
print(df.head())

print('\n\n')

# BOX/SKU
df_par = pd.DataFrame(df.groupby(['SKU'])['BOX'].sum())
df_par.columns = ['BOX']

# Sort Values
df_par.sort_values(['BOX'], ascending = False, inplace = True)
df_par.reset_index(inplace = True)

# Cumulative Sum 
df_par['CumSum'] = df_par['BOX'].cumsum()

# % CumSum
df_par['%CumSum'] = (100 * df_par['CumSum']/df_par['BOX'].sum())

# % SKU
df_par['%SKU'] = (100 * (df_par.index + 1).astype(float)/(df_par.index.max() + 1))

# > 80% Volume
df_par80 = df_par[df_par['%CumSum'] > 80].copy()
perc_sku80 = df_par80['%SKU'].min()
perc_sum80 = df_par80['%CumSum'].min()

# 20% SKU
df_sku20 = df_par[df_par['%SKU'] > 20].copy()
perc_sku20 = df_sku20['%SKU'].min()
perc_sum20 = df_sku20['%CumSum'].min()

# 5% SKU
df_sku5 = df_par[df_par['%SKU'] > 5].copy()
perc_sku5 = df_sku5['%SKU'].min()
perc_sum5 = df_sku5['%CumSum'].min()



print("Pareto Analysis for {:,} unique SKU".format(len(df_par)))
print(df_par.head())
print()
print(df_par.tail(4400))

ax = df_par.plot(x='%SKU', y='%CumSum', figsize = (20,7.5))
plt.xlabel('Percentage of SKU (%)',fontsize=15)
plt.ylabel('Percentage of Boxes Ordered (%)',fontsize=15)
plt.title('Pareto Analysis using Cumulative Sum of Boxes Prepared (%) = f(%SKU)', fontsize = 15)

# 5% SKU
ax.axhline(perc_sum5 , color="black", linestyle="--", linewidth = 1.0)
ax.axvline(perc_sku5, color="black", linestyle="--", linewidth = 1.0)

# 80% Volume
ax.axhline(perc_sum80 , color="red", linestyle="--", linewidth = 1.0)
ax.axvline(perc_sku80, color="red", linestyle="--", linewidth = 1.0)

# 20% SKU
ax.axhline(perc_sum20 , color="blue", linestyle="--", linewidth = 1.0)
ax.axvline(perc_sku20, color="blue", linestyle="--", linewidth = 1.0)

plt.show()