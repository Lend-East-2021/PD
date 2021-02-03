import numpy as np
import pandas as pd

def read_data(path):
    with open(path, 'r') as f:
        data = json.loads(f.read())
    data = pd.json_normalize(path, record_path = ['historical'])
    return data

def make_df(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.index = df['date'].dt.strftime('%b-%y')
    df.index.name = None
    idx = df.loc[:,['finCostperAUM']].notna().idxmax()[0]#.bfill().ffill().rolling(12).mean()
    df.loc[idx:, ['finCostperAUM', 'SMCostperAUM', 'OpCostperAUM', 'CLperAUM']] = (df.loc[idx:, ['finCostperAUM', 'SMCostperAUM', 'OpCostperAUM', 'CLperAUM']]/3).bfill()
    df.loc[:, ['finCostperAUM', 'SMCostperAUM', 'OpCostperAUM', 'CLperAUM']] = df.loc[:, ['finCostperAUM', 'SMCostperAUM', 'OpCostperAUM', 'CLperAUM']].rolling(12).mean().ffill()
    df['pfTotal'] = df.loc[:, ['aumCurrent', 'aum1to30DPD', 'aum31to60DPD', 'aum61to90DPD', 'aum91to120DPD', 'aum121to150DPD', 'aum151to180DPD']].sum(axis=1)
    df['M0+'] = df.loc[:, ['aum1to30DPD', 'aum31to60DPD', 'aum61to90DPD', 'aum91to120DPD', 'aum121to150DPD', 'aum151to180DPD']].sum(axis=1) / df['pfTotal']
    df[['NonIntIncome_Growth', 'Total_Growth']] = df.loc[:, ['NonIntIncome', 'Total']].pct_change()
    #replace >100% growth with mean
    mean = df['Total_Growth'].mean()
    df['Total_Growth'] = df.apply(lambda c: mean if c['Total_Growth'] > 1 else c['Total_Growth'], axis=1)
    return df

def forecast(df):
    np.random.seed(seed=1)
    months = 12 #forecast months
    
    #growth for Total aum, 12 months
    df_growth = pd.DataFrame(np.random.normal(loc=df['Total_Growth'].mean(), 
                                              scale=df['Total_Growth'].std(), 
                                              size=(2000, 12)))
    
    #growth for NI Income, 12 months
    df_NI_Income_growth = pd.DataFrame(np.random.normal(loc=df['NonIntIncome_Growth'].mean(), 
                                                        scale=df['NonIntIncome_Growth'].std(), 
                                                        size=(2000, 12)))
    #m0+ proportion, 12 months
    df_M0 = pd.DataFrame(np.random.normal(loc=df['M0+'].mean(), 
                                              scale=df['M0+'].std(), 
                                              size=(2000, 12)))
    
    APR_MA = list(df['averageAPR'].iloc[-6:]) #APR for last 6 months
    
    for i in range(months):
        APR_MA.append(np.mean(APR_MA[i:i+6])) #forecast APR for next 12 months

    #forecast aum for next 12 months from last month
    df_aum = (df_growth + 1).cumprod(axis=1) * df.iloc[-1, :].loc['Total'] #last Total Amt in dataset
    #forecast non-interest income for next 12 months from last month
    df_NI_Income = (df_NI_Income_growth + 1).cumprod(axis=1) * df.iloc[-1, :].loc['NonIntIncome']
    APR_forecast = APR_MA[6:] #APR for next 12 months
    df_revenue = df_aum.multiply(1 - df_M0) #revenue df
    df_revenue *= [apr/12 for apr in APR_forecast] #interest revenue
    
    return df_revenue, df_NI_Income, df_aum

def calc_default_prob(df, interest_income, non_int_income, aum, equity=0, RWA=1, pct_RWA=0.05):
    num_iterations = 2000
    finCost = pd.DataFrame(df['finCostperAUM'].iloc[-1], index=range(2000), columns=range(12)).mul(aum)
    SMCost = pd.DataFrame(df['SMCostperAUM'].iloc[-1], index=range(2000), columns=range(12)).mul(aum)
    OpCost = pd.DataFrame(df['OpCostperAUM'].iloc[-1], index=range(2000), columns=range(12)).mul(aum)
    CLoss = pd.DataFrame(df['CLperAUM'].iloc[-1], index=range(2000), columns=range(12)).mul(aum)
    
    total_rev = interest_income.add(non_int_income) #Total Revenue
    net_income = total_rev.subtract(finCost).subtract(SMCost).subtract(OpCost).subtract(CLoss).add(equity)
    
    #calc default prob
    final_mth_forecast = net_income.iloc[:, [-1]].copy()
    final_mth_forecast['Default'] = aum.iloc[:, [-1]] * RWA * pct_RWA
    final_mth_forecast.columns = ['Forecast', 'Default_level']
    num_of_default = ((final_mth_forecast['Forecast'] - final_mth_forecast['Default_level'])
                        <= 0).sum()
    prob_of_default = num_of_default / num_iterations
    
    return dict({'PD':prob_of_default})

def main(data):
    df = read_data(data)
    df = make_df(df)
    int_rev, NI_rev, aum = forecast(df)
    pd = calc_default_prob(df, int_rev, NI_rev, aum)
    return pd