import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pysal.lib as ps
import mapclassify
from matplotlib.colors import LinearSegmentedColormap
from pysal.lib import weights
from pysal.explore import esda
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Create outputs directory
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)


dsny     = pd.read_csv("data/dsny.csv")
pop      = pd.read_csv("data/pop.csv")
rest     = pd.read_csv("data/rest.csv")
obesity  = pd.read_csv("data/obesity.csv")
boro_gdf = gpd.read_file("data/nybb_25a/nybb.shp")
cd_gdf   = gpd.read_file("data/nycd_25a/nycd.shp")


# Standardize column names
dsny.columns = dsny.columns.str.lower()
pop.columns  = pop.columns.str.lower()

# Filter BRFSS obesity data
obesity = obesity[
    (obesity['StateAbbr'] == 'NY') &
    (obesity['Short_Question_Text'] == 'Obesity') &
    (obesity['Data_Value_Type'] == 'Age-adjusted prevalence')
]

# Select & rename
obesity = obesity[['LocationName','Data_Value']].rename(
    columns={'LocationName':'county','Data_Value':'obesity_rate'}
)

# Map to NYC boroughs
county_to_boro = {
    'Bronx':    'Bronx',
    'Kings':    'Brooklyn',
    'Queens':   'Queens',
    'Richmond': 'Staten Island',
    'New York': 'Manhattan'
}
obesity['borough'] = obesity['county'].map(county_to_boro)

# Clean up
obesity['obesity_rate'] = pd.to_numeric(obesity['obesity_rate'], errors='coerce')
obesity = obesity.dropna(subset=['borough']).reset_index(drop=True)

# Ensure 2-digit codes
dsny['communitydistrict'] = dsny['communitydistrict'].astype(str).str.zfill(2)

# Fill NaNs and total waste
for col in ['refusetonscollected','papertonscollected','mgptonscollected','resorganicstons']:
    dsny[col] = dsny[col].fillna(0)
dsny['total_waste'] = dsny[['refusetonscollected','papertonscollected','mgptonscollected']].sum(axis=1)

# Aggregate by CD
dsny_cd = dsny.groupby('communitydistrict').agg({
    'refusetonscollected':'mean',
    'papertonscollected':'mean',
    'mgptonscollected':'mean',
    'resorganicstons':'mean',
    'total_waste':'mean'
}).reset_index()

# Map borough codes
borough_map = {1:'Manhattan',2:'Bronx',3:'Brooklyn',4:'Queens',5:'Staten Island'}

cd_gdf['borough'] = (cd_gdf['BoroCD'] // 100).map(borough_map)
cd_gdf['communitydistrict'] = (cd_gdf['BoroCD'] % 100).astype(str).str.zfill(2)

cd_to_boro = cd_gdf[['communitydistrict','borough']].drop_duplicates()

# Assume pop has columns 'cd number' and '2010 population'
pop = pop[['cd number','2010 population']].rename(
    columns={'2010 population':'population'}
)
pop['communitydistrict'] = (pop['cd number'] % 100).astype(int).astype(str).str.zfill(2)
pop = pop[['communitydistrict','population']]

rest = rest.dropna(subset=['Community Board'])
rest['communitydistrict'] = (
    rest['Community Board'].astype(int).mod(100)
    .astype(str).str.zfill(2)
)
rest = rest.groupby('communitydistrict') \
           .size().reset_index(name='restaurant_count')

merged = (
    dsny_cd
      .merge(cd_to_boro, on='communitydistrict', how='left')
      .merge(pop,        on='communitydistrict', how='left')
      .merge(rest,       on='communitydistrict', how='left')
      .merge(obesity,    on='borough',           how='left')
)

# Per-capita waste and restaurant density
for c in ['refusetonscollected','papertonscollected','mgptonscollected','resorganicstons','total_waste']:
    merged[f'{c}_pc'] = merged[c] / merged['population']

merged['rest_density'] = merged['restaurant_count'] / merged['population']

# 1. Create a custom diverging colormap: light blue → white → red
cmap = LinearSegmentedColormap.from_list(
    'LightBlueWhiteGreen',
    ['lightBlue','white', 'green']
)

waste_group = merged.groupby('borough').agg({
    'papertonscollected_pc':'mean',
    'resorganicstons_pc':'mean',
    'rest_density':'mean',
    'obesity_rate':'mean'
})
corr = waste_group.corr()

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(
    corr.values,
    cmap=cmap,
    vmin=-1, vmax=1,
    aspect='equal'
)

for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        ax.text(
            j, i,
            f"{corr.iloc[i, j]:.2f}",
            ha='center',
            va='center',
            fontsize=10,
            color='black'
        )

ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Correlation coefficient', rotation=270, labelpad=15)

ax.set_title('Borough‐level Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'borough_correlation_improved.png'))
plt.show()

# 10a. Organic Waste vs. Obesity
x1 = merged['resorganicstons_pc']
y  = merged['obesity_rate']

fig, ax = plt.subplots()
ax.scatter(x1, y)
m1, b1 = np.polyfit(x1.dropna(), y.dropna(), 1)
ax.plot(x1, m1*x1 + b1, label=f'y={m1:.2f}x+{b1:.2f}')
ax.set_xlabel('Organic Waste (tons/person)')
ax.set_ylabel('Obesity Rate (%)')
ax.set_title('Organic Waste vs. Obesity (CD Level)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'organic_vs_obesity.png'))
plt.show()

# 10b. Restaurant Density vs. Obesity
x2 = merged['rest_density']

fig, ax = plt.subplots()
ax.scatter(x2, y)
m2, b2 = np.polyfit(x2.dropna(), y.dropna(), 1)
ax.plot(x2, m2*x2 + b2, label=f'y={m2:.2f}x+{b2:.2f}')
ax.set_xlabel('Restaurant Density (per person)')
ax.set_ylabel('Obesity Rate (%)')
ax.set_title('Restaurant Density vs. Obesity (CD Level)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'rest_vs_obesity.png'))
plt.show()

X = merged[['resorganicstons_pc','rest_density']].fillna(0)
X = sm.add_constant(X)
model = sm.OLS(merged['obesity_rate'].fillna(merged['obesity_rate'].mean()), X).fit()
print(model.summary())

# 12. Bivariate Choropleth: Obesity vs. Organic Waste
# Merge the obesity and organic‐waste metrics back onto the spatial GeoDataFrame
cd_gdf = cd_gdf.merge(
    merged[['communitydistrict','obesity_rate','resorganicstons_pc']],
    on='communitydistrict',
    how='left'
)


# Quantile‐class both variables into 5 bins
cd_gdf['ob_class']      = mapclassify.Quantiles(cd_gdf['obesity_rate'].fillna(0), k=5).yb
cd_gdf['organic_class'] = mapclassify.Quantiles(cd_gdf['resorganicstons_pc'].fillna(0), k=5).yb

# Combine into 25 joint classes
cd_gdf['bivar_class']   = cd_gdf['ob_class'] * 5 + cd_gdf['organic_class']

# 2) Define a 25‐color colormap (here we pick a red→white→blue diverging map, but you can choose any)
base = plt.cm.RdYlBu_r(np.linspace(0,1,25))
cmap = ListedColormap(base)

# 3) Plot the map using that colormap
fig, ax = plt.subplots(figsize=(8,8))
cd_gdf.plot(
    column='bivar_class',
    cmap=cmap,
    linewidth=0.1,
    edgecolor='grey',
    categorical=True,
    legend=False,        # we will build our own legend
    ax=ax
)
ax.axis('off')
ax.set_title('Bivariate Map: Obesity (vert) vs. Organic Waste (horiz)')
plt.tight_layout()

# 4) Build the legend patches
patches = []
labels  = []
for class_id in range(25):
    ob_qr    = class_id // 5  + 1
    org_qr   = class_id % 5   + 1
    color    = cmap(class_id)
    label    = f"Obesity Q{ob_qr}, OrgWaste Q{org_qr}"
    patches.append(mpatches.Patch(color=color, label=label))
    labels.append(label)

# 5) Add the legend outside the map
ax.legend(
    handles=patches, 
    title="Quintiles", 
    bbox_to_anchor=(1.02, 1), 
    loc='upper left',
    ncol=1, 
    fontsize='small',
    title_fontsize='medium'
)
# Plot
# 6) Save & show
plt.savefig(os.path.join(output_dir,'bivariate_obesity_organic_legend.png'),
            bbox_inches='tight')
plt.show()

# 1) Define human‐readable names for each cluster
cluster_names = {
    0: 'Cluster 0: Low Org / Low Obesity',
    1: 'Cluster 1: Low Org / High Obesity',
    2: 'Cluster 2: High Org / Low Obesity',
    3: 'Cluster 3: High Org / High Obesity'
}

# 2) Pick a categorical colormap and extract 4 distinct colors
cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(cluster_names))]

# 3) Merge clusters into the GeoDataFrame and map to labels
plot_gdf = cd_gdf.merge(merged[['communitydistrict','cluster']], on='communitydistrict')
plot_gdf['cluster_label'] = plot_gdf['cluster'].map(cluster_names)

# 4) Plot without the built‐in legend, specifying our colors
fig, ax = plt.subplots(figsize=(8,8))
plot_gdf.plot(
    column='cluster_label',
    categorical=True,
    legend=False,
    ax=ax,
    color=[colors[i] for i in plot_gdf['cluster']]
)
ax.axis('off')
ax.set_title('Neighborhood Clusters by Waste & Obesity')

# 5) Build and add a custom legend
handles = [
    mpatches.Patch(color=colors[i], label=cluster_names[i])
    for i in sorted(cluster_names)
]
ax.legend(
    handles=handles,
    title='Cluster Profiles',
    loc='upper left',
    fontsize='small',
    title_fontsize='medium',
    frameon=True,
    framealpha=0.9
)

plt.tight_layout()
plt.savefig(os.path.join(output_dir,'neighborhood_clusters_legend.png'))
plt.show()


