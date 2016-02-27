import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib

def norm_plot(mu, sigma, nstds=4, npts=500, **kw):
    """Draw the probability density fxn for N(mu,sigma).
    
    nstds = how many standard deviation to left/right to draw pdf
    npts = number of pts over which to evaluate the pdf
    **kw = any additional keywords are passed along to pyplot.plot 
    """
    distribution = stats.norm(loc=mu, scale=sigma)
    xmin, xmax = mu - nstds*sigma, mu + nstds*sigma
    x = np.linspace(xmin,xmax,npts)
    y = distribution.pdf(x)
    plt.plot(x, y, **kw)  # notice how we pass any additional keywords to the plot fxn
    
    
    # make it look nice
    ax = plt.gca() 
    ax.set_xlim(xmin*0.95, xmax*1.05)
    ax.set_ylim(0, max(y)*1.1)
    ax.set_yticks([])
    ax.xaxis.set_ticks_position('bottom')
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Density")
    
    # return the "frozen" distribution and the axes representing our plot  
    return distribution, ax
    
    
def area_under_distn(distribution, xmin, xmax, npts=500, **kw):
    """Draw the area under the pdf of the given distribution from xmin, xmax.
    
    distribution = a frozen distribution from scipy.stats
    """
    x = np.linspace(xmin,xmax,npts)
    y = distribution.pdf(x)
    plt.fill_between(x, np.zeros_like(y), y, **kw)