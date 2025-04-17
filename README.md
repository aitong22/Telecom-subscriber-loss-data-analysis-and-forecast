# Telecom-subscriber-loss-data-analysis-and-forecast
电信用户流失数据分析及预测
## 技术路线
 ![image](https://github.com/user-attachments/assets/67bce440-ab56-4282-9bdd-b5ee9f90495e)

数据预测流程图
## 数据集描述
本数据集为电信客户流失情况，包括三个部分用户账户信息、用户个人信息、用户注册的服务
 ![image](https://github.com/user-attachments/assets/62d8c63a-7431-4f1c-b1a5-696e226acb18)

用户账户信息
tenure ：任期 
MonthlyCharges：月费用
TotalCharges：总费用
Contract：签订合同方式 （按月，一年，两年）
PaperlessBilling：是否开通电子账单（Yes or No）
PaymentMethod：付款方式（bank transfer，credit card，electronic check，mailed check）

用户注册的服务
PhoneService ：是否开通电话服务业务 （Yes or No）
MultipleLines：是否开通了多线业务（Yes 、No or No phoneservice）
InternetService：是否开通互联网服务 （No, DSL数字网络，fiber optic光纤网络）
OnlineSecurity：是否开通网络安全服务（Yes，No，No internetserive）
OnlineBackup：是否开通在线备份业务（Yes，No，No internetserive）
DeviceProtection：是否开通了设备保护业务（Yes，No，No internetserive）
TechSupport：是否开通了技术支持服务（Yes，No，No internetserive）
StreamingTV：是否开通网络电视（Yes，No，No internetserive）
StreamingMovies：是否开通网络电影（Yes，No，No internetserive）

用户个人信息
Churn：该用户是否流失（Yes or No） 
customerID ：用户ID。
gender：性别。（Female & Male）
SeniorCitizen ：老年人 （1表示是，0表示不是）
Partner ：是否有配偶 （Yes or No）
Dependents ：是否经济独立 （Yes or No）
