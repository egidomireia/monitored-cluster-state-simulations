"""Paper-sized plots from individual Pauli measurement realisations."""
from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
from .pauli_measurements import load_statistics


def plot_pauli_results(raw_file, output_dir, *, target_probabilities=(0,.2,.4,.6,.8),
                       target_sizes=None, figsize=(6,4.6)):
    """Save S(L) at selected p and S/L(p) at selected L as separate PDFs.

    Bands show mean +/- sample SD/sqrt(N). Counts are read per point; missing
    and single-sample points are omitted (zero entropy is not missing data).
    There is no assumption of a deletion threshold for X/Y measurements.
    """
    data=load_statistics(raw_file)
    sizes,ps,mean,sem,counts=[data[k] for k in ('L','p','mean','sem','counts')]
    basis=data['config']['basis'];boundary='PBC' if data['config']['periodic'] else 'OBC'
    if not np.any(counts>=2):
        raise ValueError('Need at least two saved realisations at some points to plot SEM.')
    if np.any(counts<2):
        warnings.warn('Incomplete results: points with fewer than two samples are omitted.',stacklevel=2)
    mean=np.where(counts>=2,mean,np.nan)
    sem=np.where(counts>=2,sem,np.nan)
    selected_p=[]
    for p in target_probabilities:
        if not ps[0]<=p<=ps[-1]: raise ValueError('Requested probability outside stored range.')
        j=int(np.argmin(abs(ps-p)))
        if j not in selected_p: selected_p.append(j)
    target_sizes=sizes if target_sizes is None else target_sizes
    selected_L=[]
    for L in target_sizes:
        if L not in sizes: raise ValueError(f'L={L} absent from this dataset.')
        selected_L.append(int(np.where(sizes==L)[0][0]))
    styles=dict(fontsize=16)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.size':16,'axes.labelsize':22,'axes.titlesize':20,
                         'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize':13,
                         'pdf.fonttype':42}):
        fig_L,axL=plt.subplots(figsize=figsize,layout='constrained')
        fig_p,axp=plt.subplots(figsize=figsize,layout='constrained')
        cmap=plt.get_cmap('viridis')
        markers=['o','s','^','D','v','P','X']
        for k,j in enumerate(selected_p):
            color=cmap(.08+.82*k/max(1,len(selected_p)-1))
            axL.fill_between(sizes,mean[:,j]-sem[:,j],mean[:,j]+sem[:,j],color=color,alpha=.25,lw=0)
            axL.plot(sizes,mean[:,j],marker=markers[k%len(markers)],color=color,lw=1.4,
                     ms=5,label=rf'$p={ps[j]:.3f}$')
        for k,i in enumerate(selected_L):
            color=cmap(.08+.82*k/max(1,len(selected_L)-1))
            y,e=mean[i]/sizes[i],sem[i]/sizes[i]
            axp.fill_between(ps,y-e,y+e,color=color,alpha=.25,lw=0)
            axp.plot(ps,y,marker=markers[k%len(markers)],markevery=max(1,int(np.ceil(len(ps)/12))),
                     color=color,lw=1.4,ms=5,label=rf'$L={sizes[i]:g}$')
        axL.set(xlabel=r'$L$',ylabel=r'$\langle S\rangle$ (bits)')
        axp.set(xlabel=r'$p$',ylabel=r'$\langle S\rangle/L$')
        figures={'S_vs_L':fig_L,'S_over_L_vs_p':fig_p};paths={}
        for name,fig in figures.items():
            ax=fig.axes[0];ax.set_title(f'Pauli-{basis}, {boundary}')
            ax.legend(frameon=False);ax.grid(alpha=.16,lw=.5)
            ax.spines[['top','right']].set_visible(False)
            paths[name]=output_dir/f'pauli_{basis.lower()}_{boundary.lower()}_{name}.pdf'
            fig.savefig(paths[name],bbox_inches='tight',pad_inches=.04)
    return dict(figures=figures,paths=paths,statistics=data,actual_probabilities=ps[selected_p])
