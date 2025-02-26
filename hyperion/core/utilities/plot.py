import shutil
import matplotlib.pyplot as plt


def latexify(plot_func):
    """A decorator to apply LaTeX styling to matplotlib plots.
    """
    def wrapper_plot(*args, **kwargs):
        #use latex if available on the machine
        if shutil.which("latex"): 
            plt.rcParams.update({"text.usetex": True, 
                                 "font.family": "serif"})
        return plot_func(*args, **kwargs)
    return wrapper_plot
