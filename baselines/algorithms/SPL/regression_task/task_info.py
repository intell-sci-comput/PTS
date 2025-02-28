task_true_eqs = {
    'nguyen-1': 'x**3+x**2+x',  
    'nguyen-2': 'x**4+x**3+x**2+x',  
    'nguyen-3': 'x**5+x**4+x**3+x**2+x', 
    'nguyen-4': 'x**6+x**5+x**4+x**3+x**2+x',
    'nguyen-5': 'sin(x**2)*cos(x)-1',
    'nguyen-6': 'sin(x)+sin(x+x**2)',
    'nguyen-7': 'log(x+1)+log(x**2+1)',
    'nguyen-8': 'sqrt(x)',
    'nguyen-9': 'sin(x)+sin(y**2)',
    'nguyen-10': '2*sin(x)*cos(y)',
    'nguyen-11': 'x**y',
    'nguyen-12': 'x**4-x**3+0.5*y**2-y', 
    'nguyen-1c': '3.39*x**3+2.12*x**2+1.78*x', 
    'nguyen-2c': '0.48*x**4+3.39*x**3+2.12*x**2+1.78*x', 
    'nguyen-5c': 'sin(x**2)*cos(x)-0.75',
    'nguyen-7c': 'log(x+1.4)+log(x**2+1.3)',
    'nguyen-8c': 'sqrt(1.23*x)', 
    'nguyen-9c': 'sin(1.5*x)+sin(0.5*y**2)'
}

## number of independent variables in each task
num_vars = {
    'nguyen-1': 1,  
    'nguyen-2': 1,  
    'nguyen-3': 1, 
    'nguyen-4': 1,
    'nguyen-5': 1,
    'nguyen-6': 1,
    'nguyen-7': 1,
    'nguyen-8': 1,
    'nguyen-9': 2,
    'nguyen-10': 2,
    'nguyen-11': 2,
    'nguyen-12': 2, 
    'nguyen-1c': 1, 
    'nguyen-2c': 1, 
    'nguyen-5c': 1, 
    'nguyen-7c': 1, 
    'nguyen-8c': 1, 
    'nguyen-9c': 2, 
}

## bounding value for independent variable samplings: 
## [training lower bound, trianing upper bound, 
##  testing lower bound, testing upper bound]
bounds = {
    'nguyen-1': [-10, 10, 0, 100],  
    'nguyen-2': [-10, 10, 0, 100],    
    'nguyen-3': [-10, 10, 0, 100],   
    'nguyen-4': [-10, 10, 0, 100],  
    'nguyen-5': [-10, 10, 0, 100],  
    'nguyen-6': [-10, 10, 0, 100],  
    'nguyen-7': [0, 100, 0, 100],  
    'nguyen-8': [0, 100, 0, 100],  
    'nguyen-9': [-10, 10, 0, 100],  
    'nguyen-10': [-10, 10, 0, 100],  
    'nguyen-11': [0, 10, 1, 11],  
    'nguyen-12': [-10, 10, 0, 100], 
    'nguyen-1c': [-10, 10, 0, 100], 
    'nguyen-2c': [-10, 10, 0, 100], 
    'nguyen-5c': [-10, 10, 0, 100], 
    'nguyen-7c': [0, 100, 0, 10],  
    'nguyen-8c': [0, 100, 0, 100], 
    'nguyen-9c': [-10, 10, 0, 100]
}
