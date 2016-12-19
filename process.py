import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

clf = RandomForestClassifier()
print cross_validation.cross_val_score(clf, monster[predictors], monster["type"], cv=5).mean()
#kf = cross_validation.KFold(n=monster.shape[0], n_folds=3, random_state=0)
#scores = cross_validation.cross_val_score(clf, monster[predictors], monster["type"], cv=kf)

xgbc = XGBClassifier()
print cross_validation.cross_val_score(xgbc, monster[predictors], monster["type"], cv=5).mean()

monster_test = pd.read_csv("test.csv")
color_code(monster_test)

xgbc.fit(monster[predictors], monster["type"])
predictions = xgbc.predict(monster_test[predictors])

submission = pd.DataFrame({
	"id": monster_test["id"],
	"type": predictions
})

def type_decode(df):
	df.loc[df["type"] == 0, "type"] = "Ghoul"
	df.loc[df["type"] == 1, "type"] = "Goblin"
	df.loc[df["type"] == 2, "type"] = "Ghost"

type_decode(submission)
submission.to_csv("xgbc_result.csv", index=False)

f = open("result.csv")
