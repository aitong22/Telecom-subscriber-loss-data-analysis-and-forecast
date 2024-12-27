import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib._cm_listed import cmaps
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score, \
    roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.utils import resample
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取数据
df = pd.read_csv("C:\\Users\\Aitong\\Desktop\\临时文件\\数据挖掘\\Telco Customer Churn 电信客户流失\\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.info()
# 数据预处理
# 转换TotalCharges为数值类型\
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# print("缺失值统计：")
# print(df.isnull().sum())
missing_rows = df[df.isnull().any(axis=1)]
nan_rows = df[pd.isna(df['TotalCharges'])]
# print(nan_rows[['tenure', 'MonthlyCharges', 'TotalCharges']])
#将总消费额填充为月消费额
nan_mask = pd.isna(df['TotalCharges'])
df.loc[nan_mask, 'TotalCharges'] = df.loc[nan_mask, 'MonthlyCharges']
df.loc[nan_mask, 'tenure']=1
# print(df[nan_mask][['tenure', 'MonthlyCharges', 'TotalCharges']])
df = df.drop(columns=['customerID'])
df['MonthlyCharges'].describe()
df['MonthlyCharges']=pd.qcut(df['MonthlyCharges'],4,labels=['1','2','3','4'])
df['MonthlyCharges'].head()
df['TotalCharges']=pd.qcut(df['TotalCharges'],4,labels=['1','2','3','4'])
df['TotalCharges'].head()




df.replace(to_replace='No internet service',value='No',inplace=True)
df.replace(to_replace='No phone service',value='No',inplace=True)
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
categorical_features = df.select_dtypes(include=['object']).columns
df.info()
print(numeric_features)
print(categorical_features)

def labelencode(x):
    df[x] = LabelEncoder().fit_transform(df[x])
for i in categorical_features:
    labelencode(i)
for i in df.columns:
    print(i,":",df[i].unique())
X = df.drop(columns=['Churn'])
y = df['Churn']
X_minority = X[y == 1]
X_majority = X[y == 0]
y_minority = y[y == 1]
y_majority = y[y == 0]

X_minority_upsampled = resample(X_minority, replace=True, n_samples=len(X_majority), random_state=42)
y_minority_upsampled = resample(y_minority, replace=True, n_samples=len(y_majority), random_state=42)

X_balanced = pd.concat([X_majority, X_minority_upsampled])
y_balanced = pd.concat([y_majority, y_minority_upsampled])





selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
selector.fit(X_balanced, y_balanced)

selected_features = selector.get_support(indices=True)
print(selected_features)

X_selected = X_balanced.iloc[:, selected_features]
print(X_selected.columns)

xgb_model = XGBClassifier(eval_metric='logloss', alpha=0.1, reg_lambda=1.0)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# XGBoost参数网格
xgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.6, 0.8, 1.0],
    'alpha':[0.05,0.1,0.2],
    'reg_lambda':[1.0,5.0,10.0],
    'min_child_weight':[3,5,7],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

# 随机森林参数网格
rf_param_grid1 = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_param_grid = {
    'n_estimators': [200],
    'max_depth': [30],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=rf_param_grid, n_iter=1, cv=skf, scoring='roc_auc', n_jobs=1)
X_train, X_val, y_train, y_val = train_test_split(X_selected, y_balanced, test_size=0.2, random_state=42)


random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)

best_model = random_search.best_estimator_

y_val_pred = best_model.predict(X_val)
y_val_proba = best_model.predict_proba(X_val)[:, 1]
## 模型训练
# model_pipeline.fit (X_train, y_train)

## 验证集预测
# y_val_pred = model_pipeline.predict(X_val)
# y_val_proba = model_pipeline.predict_proba(X_val)[:, 1]

# 输出验证集性能
print(classification_report(y_val, y_val_pred))
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)


print("准确率:", accuracy)
print("精确率:", precision)
print("召回率:", recall)
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
roc_auc_value = auc(fpr, tpr)  # 使用不同的变量名
print("ROC AUC Score:", roc_auc_value)
# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC曲线 (AUC = {:.2f})'.format(roc_auc_value))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()




feature_importances = best_model.feature_importances_
sorted_indices = np.argsort(feature_importances)
sorted_importances = feature_importances[sorted_indices]
sorted_features = X_selected.columns[sorted_indices]

for feature, importance in zip(sorted_features, sorted_importances):
    print(f"{feature}: {importance}")

# 使用颜色映射来表示特征重要性的排序
colors = cmaps['viridis'](np.linspace(0, 1, len(sorted_features)))

plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances, color=colors)
plt.xticks(rotation=90)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importances")
plt.tight_layout()
plt.show()


# 把No internet service归为no里面对0的识别率增长0.01
# 本来使用中位数填充'TotalCharges'
# 本来使用标准差来平衡钢梁，现在离散数据使用四分位数
# 本来使用把上采样放在数据预处理之前
