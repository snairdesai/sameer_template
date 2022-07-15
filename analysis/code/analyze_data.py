import pandas as pd
import numpy as np
from linearmodels import PanelOLS

### DEFINE
def main():
    df = import_data()
    fit = run_regression(df)
    fit_rev = run_regression(df = df[df['year']>=1960])
    formatted = format_model(fit)
    formatted_rev = format_model(fit_rev)
    
    with open('output/regression.csv', 'w') as f:
        f.write('<tab:regression>' + '\n')
        formatted.to_csv(f, sep = '\t', index = False, header = False)

    with open('output/regression_rev.csv', 'w') as f:
        f.write('<tab:regression_rev>' + '\n')
        formatted_rev.to_csv(f, sep = '\t', index = False, header = False)
    
def import_data():
    df = pd.read_csv('input/data_cleaned.csv')
    df['post_tv'] = df['year'] > df['year_tv_introduced']
    
    return(df)

def run_regression(df):
    df = df.set_index(['county_id', 'year'])
    model = PanelOLS.from_formula('chips_sold ~ 1 + post_tv + EntityEffects + TimeEffects', data = df)
    fit = model.fit()
    
    return(fit)
    
def format_model(fit):
    formatted = pd.DataFrame({'coef'     : fit.params, 
                              'std_error': fit.std_errors, 
                              'p_value'  : fit.pvalues})
    formatted = formatted.loc[['post_tv[T.True]']]
    
    return(formatted)
    
### EXECUTE
main()