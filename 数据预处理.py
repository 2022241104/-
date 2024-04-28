import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
carddata = pd.read_csv(
    'D:/文档/数据挖掘/operation_question_anwer/第6章/data/credit_card.csv', encoding='GBK')

# 筛选逾期但是不是瑕疵户的数据
exp1 = (carddata['逾期'] == 1) & (carddata['瑕疵户'] == 2)
# 筛选呆账但是不是瑕疵户的数据
exp2 = (carddata['呆账'] == 1) & (carddata['瑕疵户'] == 2)
# 筛选有强制停卡记录但是不是瑕疵户的数据
exp3 = (carddata['强制停卡记录'] == 1) & (carddata['瑕疵户'] == 2)
# 筛选退票但是不是瑕疵户的数据
exp4 = (carddata['退票'] == 1) & (carddata['瑕疵户'] == 2)
# 筛选有拒收记录但是不是瑕疵户的数据
exp5 = (carddata['拒往记录'] == 1) & (carddata['瑕疵户'] == 2)
# 筛选有呆账但是没有拒收记录的数据
exp6 = (carddata['呆账'] == 1) & (carddata['拒往记录'] == 2)
# 筛选有强制停卡记录但是没有拒收记录的数据
exp7 = (carddata['强制停卡记录'] == 1) & (carddata['拒往记录'] == 2)
# 筛选退票但是没有拒收记录的数据
exp8 = (carddata['退票'] == 1) & (carddata['拒往记录'] == 2)
# 筛选频率为5但是月刷卡额大于1的数据
exp9 = (carddata['频率'] == 5) & (carddata['月刷卡额'] > 1)
# 筛选异常数据
Final = carddata.loc[(exp1 | exp2 | exp3 | exp4 | exp5 |
                      exp6 | exp7 | exp8 | exp9).apply(lambda x: not (x)), :]
Final.reset_index(inplace=True)


# 个人月收入（万元）
PersonalMonthIncome = [0, 1, 2, 3, 4, 5, 6, 7, 8]
for i in range(8):
    Final.loc[Final['个人月收入'] == i + 1, '个人月收入'] = PersonalMonthIncome[i]
# 根据5 、6的情况计算个人月收入和家庭月收入的比值，确定家庭月收入为未知的情况
FamilyMonthIncome = [2, 4, 6, 8, 10, 12]
m = (Final.loc[:, '家庭月收入'] == 5)
Final.loc[m, '家庭月收入'] = FamilyMonthIncome[4]
ratio5 = Final.loc[m, '个人月收入'] / Final.loc[m, '家庭月收入']
m1 = Final.loc[:, '家庭月收入'] == 6
Final.loc[m1, '家庭月收入'] = FamilyMonthIncome[5]
ratio6 = Final.loc[m1, '个人月收入'] / Final.loc[m1, '家庭月收入']

# 家庭月收入（万元）
FamilyMonthIncome = [2, 4, 6, 8, 10, 15]
Final.loc[Final['家庭月收入'] == 0, '家庭月收入'] = 6
for i in range(6):
    m2 = Final.loc[:, '家庭月收入'] == i + 1
    Final.loc[m2, '家庭月收入'] = FamilyMonthIncome[i]

# 月刷卡额（万元）
MonthCardPay = [2, 4, 6, 8, 10, 15, 20, 25]
for i in range(8):
    m = Final.loc[:, '月刷卡额'] == i + 1
    Final.loc[m, '月刷卡额'] = MonthCardPay[i]

# 个人月开销（万元）
PersonalMonthOutcome = [1, 2, 3, 4, 6]
for i in range(5):
    m = Final['个人月开销'] == i + 1
    Final.loc[m, '个人月开销'] = PersonalMonthOutcome[i]


# 属性值为1（是）的记为1分，属性值为2（否）的记为0分
def GetScore(x):
    if x == 2:
        a = 0
    else:
        a = 1
    return (a)


BuguserSocre = Final['瑕疵户'].apply(GetScore)
OverdueScore = Final['逾期'].apply(GetScore)
BaddebtScore = Final['呆账'].apply(GetScore)
CardstopedScore = Final['强制停卡记录'].apply(GetScore)
BounceScore = Final['退票'].apply(GetScore)
RefuseScore = Final['拒往记录'].apply(GetScore)
Final['历史信用风险'] = (BuguserSocre + OverdueScore * 2 + BaddebtScore * 3
                   + CardstopedScore * 3 + BounceScore * 3 + RefuseScore * 3)


# 月刷卡额/个人月收入
CardpayPersonal = Final['月刷卡额'] / Final['个人月收入']
# 月刷卡额/家庭月收入
CardpayFamily = Final['月刷卡额'] / Final['家庭月收入']
EconomicScore = []
for i in range(Final.shape[0]):
    if CardpayPersonal[i] <= 1:
        if Final.loc[i, '借款余额'] == 1:
            EconomicScore.append(1)
        else:
            EconomicScore.append(0)

    if CardpayPersonal[i] > 1:
        if CardpayFamily[i] <= 1:
            if Final.loc[i, '借款余额'] == 1:
                EconomicScore.append(2)
            else:
                EconomicScore.append(1)

    if CardpayFamily[i] > 1:
        if Final.loc[i, '借款余额'] == 1:
            EconomicScore.append(4)
        else:
            EconomicScore.append(2)

# 个人月开销/月刷卡额
OutcomeCardpay = Final['个人月开销'] / Final['月刷卡额']
OutcomeCardpayScore = []
for i in range(Final.shape[0]):
    if (OutcomeCardpay[i] <= 1):
        OutcomeCardpayScore.append(1)
    else:
        OutcomeCardpayScore.append(0)

Final['经济风险情况'] = np.array(EconomicScore) + np.array(OutcomeCardpayScore)


# 判断用户是否具有稳定的收入
HouseScore = []
for i in range(Final.shape[0]):
    if 3 <= Final.loc[i, '住家'] <= 5:
        HouseScore.append(0)
    else:
        HouseScore.append(1)

JobScore = []
for i in range(Final.shape[0]):
    if (Final.loc[i, '职业'] <= 7 | Final.loc[i, '职业'] == 19 |
            Final.loc[i, '职业'] == 21):
        JobScore.append(2)
    if (Final.loc[i, '职业'] >= 8 & Final.loc[i, '职业'] <= 11):
        JobScore.append(1)
    if (Final.loc[i, '职业'] <= 18 & Final.loc[i, '职业'] >= 12 |
            Final.loc[i, '职业'] == 20 | Final.loc[i, '职业'] == 22):
        JobScore.append(0)

AgeScore = []
for i in range(Final.shape[0]):
    if Final.loc[i, '年龄'] <= 2:
        AgeScore.append(1)
    else:
        AgeScore.append(0)

Final['收入风险情况'] = np.array(HouseScore) + \
    np.array(JobScore) + np.array(AgeScore)


StdScaler = StandardScaler().fit(Final[['历史信用风险', '经济风险情况', '收入风险情况']])
ScoreModel = StdScaler.transform(Final[['历史信用风险', '经济风险情况', '收入风险情况']])
