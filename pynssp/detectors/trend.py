import pandas as pd
import numpy as np
import statsmodels.api as sm

def classify_trend(df, t='date', data_count='dataCount', all_count='allCount', B=12):
    t = df[t]
    data_count = df[data_count]
    all_count = df[all_count]
    
    trend_analysis = pd.DataFrame({'t': t, 'data_count': data_count, 'all_count': all_count})\
        .rolling(window=B)\
        .apply(lambda x: np.nan \
               if np.sum(x['data_count']) <= 10 \
                else sm.GLM(np.column_stack([x['data_count'], x['all_count'] - x['data_count']]), 
                            sm.add_constant(x['t']), 
                            family=sm.families.Binomial()).fit().params[1],
                            raw=True
        )
    
    p_value = pd.DataFrame({'t': t, 'data_count': data_count, 'all_count': all_count})\
        .rolling(window=B)\
        .apply(lambda x: np.nan \
               if np.sum(x['data_count']) <= 10 \
                else sm.GLM(np.column_stack([x['data_count'], x['all_count'] - x['data_count']]), 
                            sm.add_constant(x['t']), 
                            family=sm.families.Binomial()).fit().pvalues[1],
                            raw=True
        )
    
    trend_classification = np.select(
        condlist=[p_value < 0.01, p_value < 0.01, np.isnan(p_value)], 
        choicelist=['Significant Increase', 'Significant Decrease', 'Insufficient Data'], 
        default='Stable'
    )
    
    trend_classification = pd.Categorical(
        trend_classification, 
        categories=['Significant Increase', 'Significant Decrease', 'Stable', 'Insufficient Data']
    )
    
    return pd.DataFrame({
        'trend_analysis': trend_analysis, 
        'p.value': p_value, 
        'trend_classification': trend_classification
    })
