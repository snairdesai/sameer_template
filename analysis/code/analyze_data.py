import pandas as pd
import numpy as np
from linearmodels import PanelOLS

### DEFINE
def main():
    df = import_data()
    fit = run_regression(df)
    formatted = format_model(fit)
    
    with open('output/regression.csv', 'w') as f:
        f.write('<tab:regression>' + '\n')
        formatted.to_csv(f, sep = '\t', index = False, header = False)
    
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

def main_rev():
    df_rev = import_data_rev()
    fit_rev = run_regression_rev(df_rev)
    formatted_rev = format_model_rev(fit_rev)
    
    with open('output/regression_rev.csv', 'w') as f:
        f.write('<tab:regression_rev>' + '\n')
        formatted_rev.to_csv(f, sep = '\t', index = False, header = False)
    
def import_data_rev():
    df_rev = pd.read_csv('input/data_cleaned.csv')
    df_rev['post_tv'] = df_rev['year'] > df_rev['year_tv_introduced']
    df_rev = df_rev[df_rev['year'] >= 1960]
    return(df_rev)

def run_regression_rev(df_rev):
    df_rev = df_rev.set_index(['county_id', 'year'])
    model_rev = PanelOLS.from_formula('chips_sold ~ 1 + post_tv + EntityEffects + TimeEffects', data = df_rev)
    fit_rev = model_rev.fit()
    
    return(fit_rev)
    
def format_model_rev(fit_rev):
    formatted_rev = pd.DataFrame({'coef'     : fit_rev.params, 
                              'std_error': fit_rev.std_errors, 
                              'p_value'  : fit_rev.pvalues})
    formatted_rev = formatted_rev.loc[['post_tv[T.True]']]
    
    return(formatted_rev)
    
### EXECUTE
main()
main_rev()