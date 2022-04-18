import os
import pandas as pd

path = r'E:\ec\工业源打印报表导出20200108'
files = os.listdir(path)
data = []
a = ['企业名称', '企业地址', '法人代表', '联系电话', '产品名称', '产品代码', '生产工艺名称', '生产工艺代码', '计量单位', '生产能力', '实际产量',
     '一般工业废物', '一般工业固废名称', '一般工业固废代码', '一般工业产生量', '一般工业固废综合利用量', '13', '14', '15', '16', '17', '18', '19',
     '危险废物', '危险废物名称', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']


def basicInfo():
    i = 0
    for f in files:
        i += 1
        if i == 1:
            continue
        print(f)
        file = path + "\\" + f
        df = pd.read_excel(file)
        df2 = pd.read_excel(file, sheet_name='G101_2_打印')
        df3 = pd.read_excel(file, sheet_name='G101_1_续表打印')
        row = []
        row.append(df.iloc[7, 3])
        row.append(df.iloc[13, 4].replace('\xa0乡（镇）', '') + df.iloc[14, 1].replace('\xa0街（村）、门牌号', ''))
        row.append(df.iloc[19, 1])
        row.append(df.iloc[21, 6])
        print(df2.iloc[7, :])
        row.append(df2.iloc[7, 0])
        for j in range(6):
            row.append(df2.iloc[7, j + 2])

        if df3.iloc[7, 1] == '是':
            row.append('是')
            df4 = pd.read_excel(file, sheet_name='G104_1')

            for k in range(11):
                row.append(df4.iloc[7 + k, 4])
        else:
            row.append('否')
            for k in range(11):
                row.append('')

        if df3.iloc[8, 1] == '是':
            row.append('是')
            df5 = pd.read_excel(file, sheet_name='G104_2')
            for k in range(11):
                row.append(df5.iloc[7 + k, 4])
        else:
            row.append('否')
            for k in range(11):
                row.append('')

        print(row)
        data.append(row)

    df = pd.DataFrame(data, columns=a)
    df.to_excel('结果s.xlsx')


basicInfo()
