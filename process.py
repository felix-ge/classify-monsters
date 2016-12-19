import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

def color_code(df):
	monster_color = df["color"].unique()		
	for i_color, m_color in enumerate(monster_color):
		df.loc[df["color"] == m_color, "color"] = i_color
	df["color"] = pd.to_numeric(df["color"])	

def type_code(df):
	monster_type = df["type"].unique()
	for i_type, m_type in enumerate(monster_type):
		df.loc[df["type"] == m_type, "type"] = i_type

	df["type"] = pd.to_numeric(df["type"])
	#df["type"].apply(pd.to_numeric)	
	
monster = pd.read_csv("train.csv")

'''
print monster.describe()
print monster.info()
print monster.head()
'''

color_code(monster)
type_code(monster)

predictors = ["bone_length", "rotting_flesh", "hair_length", "has_soul", "color"]

selector = SelectKBest(f_classif, k='all')
selector.fit(monster[predictors], monster["type"])
scores = -np.log10(selector.pvalues_)
print scores

plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()

predictors = ["bone_length", "rotting_flesh", "hair_length", "has_soul"]
sample_leaf_options = list(range(1, 25, 1))
n_estimators_options = list(range(5, 55, 5))
groud_truth = monster["type"][300:]
results = []

for leaf_size in sample_leaf_options:
	for n_estimators_size in n_estimators_options:
		clf = RandomForestClassifier(n_estimators=n_estimators_size, min_samples_leaf=leaf_size, random_state=50)
		clf.fit(monster[predictors][:300], monster["type"][:300])
		predictions = clf.predict(monster[predictors][300:])
		kf = cross_validation.KFold(n=monster.shape[0], n_folds=3, random_state=0)
		scores = cross_validation.cross_val_score(clf, monster[predictors], monster["type"], cv=kf)
		# print(leaf_size, n_estimators_size, (groud_truth == predictions).mean())
		print(leaf_size, n_estimators_size, scores.mean())
		# results.append((leaf_size, n_estimators_size, (groud_truth == predictions).mean()))		
		results.append((leaf_size, n_estimators_size, scores.mean()))		

print(max(results, key=lambda x:x[2]))

monster_test = pd.read_csv("test.csv")
color_code(monster_test)

clf.fit(monster[predictors], monster["type"])
predictions = clf.predict(monster_test[predictors])

submission = pd.DataFrame({
	"id": monster_test["id"],
	"type": predictions
})

def type_decode(df):
	df.loc[df["type"] == 0, "type"] = "Ghoul"
	df.loc[df["type"] == 1, "type"] = "Goblin"
	df.loc[df["type"] == 2, "type"] = "Ghost"

type_decode(submission)
submission.to_csv("result.csv", index=False)

f = open("result.csv")
# print(f.read())
