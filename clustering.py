import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


funnel_data = pd.read_excel('W27307.xlsx')
tnb_policy_data = pd.read_excel('W27308.xlsx')
regional_data = pd.read_excel('W27309.xlsx')

funnel_data['index'] = range(1, len(funnel_data) + 1)

funnel_data_policy = funnel_data.sort_values(['updated_on','index']).drop_duplicates('policy_number',keep='last')
funnel_data_non_policy = funnel_data[funnel_data['policy_number'].isnull()].sort_values(['updated_on','index']).drop_duplicates(['offer_number','affinity_name'],keep='last')
funnel_policy_regional = funnel_data_policy.merge(regional_data, how="inner", on=['zipcode_link'])
funnel_non_policy_regional = funnel_data_non_policy.merge(regional_data, how="inner", on=['zipcode_link'])
test_data_policy_regional = pd.concat([funnel_policy_regional,funnel_non_policy_regional])


test_data_policy_regional.fillna(0,inplace = True)

columns = ['PROVINCES','URB','INCOME','SOCCL_A','SOCCL_B1','SOCCL_B2','SOCCL_C','SOCCL_D','EDU_HIGH','EDU_MID','EDU_LOW','DINK','OWN_HOUSE','AVG_HOUSE','RENT_PRICE','STAGE_OF_LIFE','SINGLE','FAM','FAM_WCHILD','SINGLES_YOUNG','SINGLES_MID','SINGLES_OLD','FAM_CHILD_Y','FAM_CHILD_O','FAM_WCHILD_Y','FAM_WCHILD_MED','FAM_WCHILD_OLD','CIT_HOUSEHOLD','LOAN','SAVINGS','SHOP_ONLINE','CAR']

le = preprocessing.LabelEncoder()

test_data_policy_regional['PROVINCES'] = le.fit_transform(test_data_policy_regional['PROVINCE'])

X_std = StandardScaler().fit_transform(test_data_policy_regional[columns])
# Create a PCA instance: pca
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(X_std)
# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)

ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(PCA_components.iloc[:, :3])

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
plt.savefig('elbow.png')

model = KMeans(n_clusters=5).fit(PCA_components.iloc[:, :3])
# label = model.labels_
# centres = model.cluster_centers_
y_kmeans = model.predict(PCA_components.iloc[:, :3])

plt.scatter(PCA_components.iloc[:, :3].values[:, 0], PCA_components.iloc[:, :3].values[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
plt.savefig('kmeans_cluster.png')

test_data_policy_regional['Cluster'] = y_kmeans
test_data_policy_regional.to_csv("clustered_data.csv")







