import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

def diagnostic_plots(model, cooksd_prop=False):
    """
    Generate diagnostic plots for a regression model to assess the validity of model assumptions.
    
    This function creates a 2x2 grid of plots including Residuals vs Fitted, Q-Q plot, 
    Scale-Location plot, and Cook's Distance plot. These plots help to diagnose various 
    aspects of a regression model, such as linearity, homoscedasticity, and influential observations.
    
    Parameters:
    - model: A fitted regression model object from statsmodels.
    - cooksd_prop: A boolean flag. If True, the Cook's Distance plot will use a dynamic threshold
                   of 4/n (where n is the number of observations). Otherwise, a fixed threshold
                   of 1 is used. Default is False.

    Returns:
    - None: The function creates and displays the plot grid but does not return any values.
    
    Example usage:
    >>> model = sm.OLS(y, X).fit()
    >>> diagnostic_plots(model, cooksd_prop=True)
    """
    # Create a 2 by 2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # =============================================================================
    # RESIDUALS VS FITTED
    # =============================================================================
    sns.residplot(
        x=model.fittedvalues,
        y=model.resid,
        lowess=True,
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red', 'lw': 3, 'alpha': 0.8},
        ax=axs[0, 0]
    )
    axs[0, 0].set_title('Residuals vs Fitted')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')
    
    # =============================================================================
    # QQ PLOT
    # =============================================================================
    # Q-Q plot with standardized residuals
    QQ = sm.ProbPlot(model.get_influence().resid_studentized_internal)
    QQ.qqplot(line='45', alpha=0.5, lw=0.5, ax=axs[0, 1])
    axs[0, 1].set_title('Q-Q Residuals')
    axs[0, 1].set_xlabel('Theoretical Quantiles')
    axs[0, 1].set_ylabel('Standardized Residuals')
    
    # =============================================================================
    # SCALE-LOCATION PLOT
    # =============================================================================
    standardized_resid = model.get_influence().resid_studentized_internal
    axs[1, 0].scatter(model.fittedvalues, np.sqrt(np.abs(standardized_resid)), alpha=0.5)
    axs[1, 0].set_title('Scale-Location')
    axs[1, 0].set_ylabel('√|Standardized residuals|')
    axs[1, 0].set_xlabel('Fitted values')

    # =============================================================================
    # COOKS DISTANCE PLOT
    # =============================================================================
    influence = model.get_influence()
    (c, p) = influence.cooks_distance
    axs[1, 1].stem(np.arange(len(c)), c, markerfmt=",", use_line_collection=True)
    axs[1, 1].set_title("Cook's distance")
    axs[1, 1].set_xlabel('Obs. number')
    axs[1, 1].set_ylabel("Cook's distance")

    # Draw Cook's distance threshold line
    if cooksd_prop:
        cooks_d_threshold = 4 / len(model.fittedvalues)  # Calculate the threshold
        label = f"Cook's d = {cooks_d_threshold:.2g}"
        threshold = cooks_d_threshold
    else:
        label = "Cook's d = 1"
        threshold = 1
    
    axs[1, 1].axhline(y=threshold, linestyle='--', color='orange', linewidth=1)
    axs[1, 1].text(x=np.max(np.arange(len(c)))*0.85, y=threshold*1.1, s=label, color='orange', va='bottom', ha='center')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()
