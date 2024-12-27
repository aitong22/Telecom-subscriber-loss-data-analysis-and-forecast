
import pandas as pd
import seaborn as sns
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
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})




df.replace(to_replace='No internet service',value='No',inplace=True)
df.replace(to_replace='No phone service',value='No',inplace=True)

# 分离特征和目标变量
X = df.drop(columns=['Churn'])
y = df['Churn']

# 使用上采样对少数类进行平衡
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_minority = X_train[y_train == 1]
X_majority = X_train[y_train == 0]
y_minority = y_train[y_train == 1]
y_majority = y_train[y_train == 0]

X_minority_upsampled = resample(X_minority, replace=True, n_samples=len(X_majority), random_state=42)
y_minority_upsampled = resample(y_minority, replace=True, n_samples=len(y_majority), random_state=42)

X_balanced = pd.concat([X_majority, X_minority_upsampled])
y_balanced = pd.concat([y_majority, y_minority_upsampled])









# 分商数值特征和分类特征
numeric_features = X_balanced.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X_balanced.select_dtypes(include=['object']).columns

print(numeric_features)
print(categorical_features)

# 数值特征处理管道
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# 分类特征处理管道
categorical_transformer = Pipeline(steps=[
    # ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ('targetEncoder', TargetEncoder())
])

# 特征预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 预处理训练集
X_train_preprocessed = preprocessor.fit_transform(X_balanced, y_balanced)

# 使用随机森林选择特征
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
selector.fit(X_train_preprocessed, y_balanced)

# 获取选择的特征索引
selected_features = selector.get_support(indices=True)


# 获取选择的特征名称
selected_numeric_features = [numeric_features[i] for i in selected_features if i < len(numeric_features)]
selected_categorical_features = [categorical_features[i - len(numeric_features)] for i in selected_features if
                                 i >= len(numeric_features)]

# 结合选择的特征名称
selected_feature_names = list(selected_numeric_features) + list(selected_categorical_features)
print(selected_feature_names)

# 根据选择的特征重新划分训练集
X_selected = X_balanced[selected_feature_names]

# 重新分离数值特征和分类特征
numeric_features_selected = X_selected.select_dtypes(include=['float64', 'int64']).columns
categorical_features_selected = X_selected.select_dtypes(include=['object']).columns

# 特征预处理管道，由于参数变化，需要重置
preprocessor_selected = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features_selected),
        ('cat', categorical_transformer, categorical_features_selected)
    ])




# 定义模型

xgb_model = XGBClassifier(eval_metric='logloss', scale_pos_weight=len(y[y == 0]) / len(y[y == 1]), alpha=0.1, reg_lambda=1.0)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
logreg_model = LogisticRegression(random_state=42, max_iter=1000)
svc_model = SVC(probability=True, random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)

# 结合特征处理与 XGBoost 模型
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor_selected), ('classifier', rf_model)])
# 超参数搜索空间
# XGBoost参数网格
xgb_param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 4, 5],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__alpha':[0.05,0.1,0.2],
    'classifier__reg_lambda':[1.0,5.0,10.0],
    'classifier__min_child_weight':[3,5,7],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
    'classifier__scale_pos_weight': [None, len(y[y == 0]) / len(y[y == 1])]
}

# 随机森林参数网格
rf_param_grid1 = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

rf_param_grid = {
    'classifier__n_estimators': [200],
    'classifier__max_depth': [30],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1]
}
rf_param_grid2 = {
    'classifier__n_estimators': [50],
    'classifier__max_depth': [30],
    'classifier__min_samples_split': [10],
    'classifier__min_samples_leaf': [2]
}

# 逻辑回归参数网格
logreg_param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg'],  # 注意：'sag'和'saga'仅支持'l2'惩罚
    'classifier__penalty': ['l1', 'l2'],
    'classifier__max_iter': [100, 200, 500]
}

# 支持向量机参数网格
svc_param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [1, 0.1, 0.01, 0.001],
    'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

# 梯度提升参数网格
gb_param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 4, 5],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# AdaBoost参数网格
ada_param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__learning_rate': [0.5, 1.0, 1.5],
    'classifier__base_estimator': [None,  DecisionTreeClassifier(max_depth=1),
                                    DecisionTreeClassifier(max_depth=3)]  # 使用元组指定自定义基估计器
}



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(model_pipeline, param_distributions=rf_param_grid2, n_iter=1, cv=skf, scoring='roc_auc',
                                   n_jobs=1)

# 训练集和验证集划分



random_search.fit(X_balanced, y_balanced)
print("Best parameters:", random_search.best_params_)

# 使用最佳貘型对验证集进行评佶
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
# auc = roc_auc_score(y_val, y_val_proba)

print("准确率:", accuracy)
print("精确率:", precision)
print("召回率:", recall)
# print("ROC AUC Score:", auc)



# 计算 AUC 值
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)

# 计算 AUC 值
roc_auc_value = auc(fpr, tpr)  # 使用不同的变量名
print("ROC AUC Score:", roc_auc_value)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC曲线 (AUC = {:.2f})'.format(roc_auc_value))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # 对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()
final_score = 0.4 * roc_auc_value + 0.4 * recall + 0.2 * precision
print("最终的得分是：", final_score)