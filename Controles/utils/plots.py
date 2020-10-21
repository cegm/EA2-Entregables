from statsmodels.api import ProbPlot, qqline

colors = {
    'dots':  '#2b7bba',
    'dots2': '##539ecd'
    'model': '#e13342'
}

def palette_hex(palette_name):
    palette = sns.color_palette(palette_name)
    sns.palplot(palette)
    return palette.as_hex()

def fit_data(df):
    import statsmodels.formula.api as sm
    formula_str = df.columns[-1]+' ~ '+'+'.join(df.columns[:-1])
    model=sm.ols(formula=formula_str, data=df)
    fitted = model.fit()

    b0 = fitted.summary2().tables[1]['Coef.']['Intercept']
    b1 = fitted.summary2().tables[1]['Coef.']['electricidad']

    residuals = fitted.resid
    y_approx = fitted.fittedvalues
    normal_resid = fitted.resid_pearson

    return fitted, b0, b1, y_approx, residuals, normal_resid

def format_plot(ax, x_label, y_label, caption = None):
    AXIS_SIZE = 12
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if caption:
        x_label = x_label + '\n\n' + caption

    ax.set_xlabel(x_label,fontsize=AXIS_SIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_SIZE)
    
def model_plot(ax, y, X, b0, b1):
    y_approx = b0 + b1*X
    label = 'y = {:.2f}x + {:.2f}'.format(b1, b0)
    ax.plot(X, y_approx, label = label, color = colors['model'])
    ax.scatter(X,y, alpha = 0.6, color = colors['dots'])

def scatter_plot(ax, X, y):
    ax.scatter(X,y, color = colors['dots'])
    
def residual_plot(ax, residuals, y_approx):
    xmin=min(y_approx)
    xmax = max(y_approx)
    ax.hlines(y=0, xmin=xmin*0.9, xmax=xmax*1.1, color='black',linestyle='--',lw=1, alpha = 0.6)
    ax.scatter(y_approx, residuals, color = colors['dots'])
    
def qq_plot(ax, residuals):
    pp = ProbPlot(residuals, fit=True)
    qq = pp.qqplot(alpha=0.7, markeredgecolor = colors['dots'], markerfacecolor = colors['dots'], ax = ax)
    qqline(qq.axes[0], line='45', color='black',linestyle='--',lw=1, alpha = 0.6)
    
def histogram_plot(ax, y_approx):
    ax.hist(y_approx, bins=20, color = colors['dots'], edgecolor = 'k')

def plot(params, num = 1, x_labels = ['x'], y_labels = ['y'], captions = [''], types = ['scatter']):
    plots = {
        'scatter': scatter_plot,
        'residual': residual_plot,
        'modelo': model_plot,
        'qqplot': qq_plot,
        'histogram': histogram_plot
    }
    
    fig_num = num
    n_plots = len(types)
    
    if n_plots == 1: axes = [axes]
        
    for ax, type_, param, x_label, y_label, caption in zip(axes, types, params, x_labels, y_labels, captions):
        plots[type_](ax = ax, fig_num = fig_num, **param)
        caption = 'Figura {}. {}'.format(fig_num, caption)
        format_plot(ax,
            x_label = x_label,
            y_label = y_label,
            caption = caption
        )
        fig_num += 1
        
    if 'modelo' in types:
        plt.legend()

    plt.show()