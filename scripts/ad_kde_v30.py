#!/usr/bin/env python3
"""Enhanced KDE via Algebraic Diversity (v5).

Adds (i) an algebraic-residue (AD+AR) floor-stripping option to the AD bandwidth
selector and the AD-Wiener estimator, selectable with --strip residue (the simple
1/n strip remains the default), and (ii) comparisons to Chiu's ECF selector,
Botev's diffusion estimator (improved Sheather-Jones), and Abramson's adaptive
variable-bandwidth estimator. Includes a heaped-data robustness experiment.

The residue strip estimates the noise floor as the maximum-entropy (white) level
of the squared-ECF spectrum and recovers the structured part with the soft Wiener
(LMMSE) taper, after the published AD+AR method (Thornton, framework paper, and the
algebraic-residue working paper). The best-matched group is known (cyclic) for the
binned measure, so no group library or CLEAN/PEEL engine is needed or included.

Reproduces fig_kde_examples.pdf, fig_kde_hard.pdf, fig_kde_heaped.pdf and the tables.
Seed fixed (see DATA.md)."""
import os, argparse, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter1d
try:
    from KDEpy.bw_selection import improved_sheather_jones; HAVE_ISJ=True
except Exception: HAVE_ISJ=False
LN2=np.log(2.0); FIG=os.path.join(os.path.dirname(__file__),"..","figures")

MW={
 "Gaussian":([1.0],[0.0],[1.0]),
 "Bimodal":([.5,.5],[-1,1],[2/3,2/3]),
 "Kurtotic":([2/3,1/3],[0,0],[1,0.1]),
 "Claw":([.5]+[.1]*5,[0,-1,-.5,0,.5,1],[1,.1,.1,.1,.1,.1]),
 "Asym. claw":([.5]+[2.0**(1-l)/31 for l in range(-2,3)],[0]+[l+0.5 for l in range(-2,3)],[1]+[2.0**(-l)/10 for l in range(-2,3)]),
 "Smooth comb":([2.0**(5-l)/63 for l in range(6)],[(65-96*2.0**(-l))/21 for l in range(6)],[(32/63)*2.0**(-l) for l in range(6)]),
 "Discrete comb":([2/7]*3+[1/21]*3,[(12*l-15)/7 for l in (0,1,2)]+[2*l/7 for l in (8,9,10)],[2/7]*3+[1/21]*3),
 "Strongly skewed":([1/8]*8,[3*((2/3)**l-1) for l in range(8)],[(2/3)**l for l in range(8)]),
}
RANGE={"Gaussian":(-5,5),"Bimodal":(-4,4),"Kurtotic":(-4,4),"Claw":(-3.5,3.5),
 "Asym. claw":(-3,4),"Smooth comb":(-3,4),"Discrete comb":(-3.5,3.5),"Strongly skewed":(-3,3)}

def pdf(name,x):
    w,mu,sd=MW[name]; return sum(wi/(si*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mi)/si)**2) for wi,mi,si in zip(w,mu,sd))
def sample(name,n,rng):
    w,mu,sd=MW[name]; w=np.array(w)/np.sum(w); c=rng.choice(len(w),size=n,p=w); return rng.normal(np.array(mu)[c],np.array(sd)[c])
def _binned(d,xg):
    dx=xg[1]-xg[0]; lo=xg[0]-dx/2; idx=np.clip(((d-lo)/dx).astype(int),0,len(xg)-1)
    p=np.bincount(idx,minlength=len(xg)).astype(float); return p/p.sum(),dx
def fft_kde(d,h,xg):
    p,dx=_binned(d,xg); p=p/dx
    g=np.arange(-len(xg)+1,len(xg))*dx; kern=np.exp(-0.5*(g/h)**2)/(h*np.sqrt(2*np.pi))
    return np.clip(fftconvolve(p,kern,mode='same')*dx,0,None)
def ise(name,f,xg): return float(np.trapezoid((f-pdf(name,xg))**2,xg))
def h_silverman(d):
    n=len(d); s=np.std(d,ddof=1); iqr=np.subtract(*np.percentile(d,[75,25])); return 0.9*min(s,iqr/1.349)*max(n,2)**(-1/5)
def h_isj(d):                                   # Botev improved Sheather-Jones bandwidth
    if not HAVE_ISJ: return h_silverman(d)
    try: return float(improved_sheather_jones(d.reshape(-1,1)))
    except Exception: return h_silverman(d)

# ---------------- floor estimates ----------------
def _floor_simple(ecf2,n): return 1.0/n                       # theoretical ECF floor (default)
def _floor_residue(ecf2):  return np.median(ecf2[1:])/LN2     # AD+AR max-entropy floor (robust)

# ---------------- AD bandwidth selector ----------------
def h_ad(d,xg,strip="simple"):
    n=len(d); p,dx=_binned(d,xg); ecf2=np.abs(np.fft.rfft(p))**2
    tk=2*np.pi*np.arange(len(ecf2))/(len(xg)*dx); sm=uniform_filter1d(ecf2,max(3,len(ecf2)//128))
    nu=_floor_residue(ecf2) if strip=="residue" else _floor_simple(ecf2,n)
    below=np.where(sm[1:]<nu)[0]; kc=(below[0]+1) if len(below) else len(ecf2)-1
    if strip=="residue":
        w=np.clip(sm-nu,0,None)/np.maximum(sm,1e-15); S=w*ecf2; S[kc:]=0.0     # soft taper + coherent cutoff
    else:
        S=np.clip(ecf2-nu,0,None); S[kc:]=0.0                                  # hard strip + cutoff
    hs=h_silverman(d); hgrid=np.geomspace(0.08*hs,4*hs,90); best,bh=np.inf,hs
    for h in hgrid:
        psi=np.exp(-0.5*(h*tk)**2); G=np.sum(ecf2[1:]*psi[1:]**2-2*S[1:]*psi[1:])
        if G<best: best,bh=G,h
    return bh
def ad_bw(d,xg,strip="simple"): return fft_kde(d,h_ad(d,xg,strip),xg)

# ---------------- optional uniformity gate (flatten to a perfect uniform) ----------------
def uniform_gate(d, xg, alpha=0.01):
    """Spectral-flatness test against the uniform null. Each non-DC squared ECF coefficient has
    n*ecf2[k] ~ Exp(1) under uniform, so its maximum over the K candidate modes concentrates near
    ln(K); the stream is declared uniform when that maximum stays below ln(K/alpha), an extreme-value
    threshold that scales the detectable ripple as 1/n. Returns (is_uniform, statistic, threshold)."""
    n = len(d); p, dx = _binned(d, xg); ecf2 = np.abs(np.fft.rfft(p)) ** 2
    K = len(ecf2) - 1
    if K < 1 or n < 1: return False, 0.0, np.inf
    stat = float(ecf2[1:].max() * n); tau = float(np.log(K / alpha))
    return stat < tau, stat, tau

def _flat_density(xg):
    f = np.ones_like(xg, dtype=float); return f / np.trapezoid(f, xg)

# ---------------- AD-Wiener adaptive estimator ----------------
def ad_wiener(d,xg,strip="simple",flatten=False,flatten_alpha=0.01):
    if flatten and uniform_gate(d, xg, flatten_alpha)[0]:
        return _flat_density(xg)                                                # snap to a perfect uniform
    n=len(d); p,dx=_binned(d,xg); Phat=np.fft.rfft(p); ecf2=np.abs(Phat)**2
    Ssm=uniform_filter1d(ecf2,max(3,len(ecf2)//128))
    nu=_floor_residue(ecf2) if strip=="residue" else _floor_simple(ecf2,n)
    below=np.where(Ssm[1:]<nu)[0]; kc=(below[0]+1) if len(below) else len(ecf2)-1
    if strip=="residue":
        Wf=np.clip(Ssm-nu,0,None)/np.maximum(Ssm,1e-15); Wf[kc:]=0.0           # soft Wiener taper + cutoff
    else:
        Shat=np.clip(Ssm-nu,0,None); Shat[kc:]=0.0; Wf=Shat/(Shat+nu)
    f=np.clip(np.fft.irfft(Wf*Phat,n=len(xg))/dx,0,None); s=np.trapezoid(f,xg); return f/s if s>0 else f

# ---------------- baselines ----------------
def chiu(d,xg):                                  # Chiu (1991) ECF plug-in selector
    n=len(d); p,dx=_binned(d,xg); ecf2=np.abs(np.fft.rfft(p))**2
    tk=2*np.pi*np.arange(len(ecf2))/(len(xg)*dx); nu=1.0/n
    sm=uniform_filter1d(ecf2,max(3,len(ecf2)//128)); below=np.where(sm[1:]<2*nu)[0]
    Lam=(below[0]+1) if len(below) else len(ecf2)-1
    dt=tk[1]-tk[0] if len(tk)>1 else 1.0
    psi4=(1.0/np.pi)*np.sum((tk[1:Lam]**4)*np.clip(ecf2[1:Lam]-nu,0,None))*dt   # est of int (f'')^2
    psi4=max(psi4,1e-8); RK=1.0/(2*np.sqrt(np.pi))
    h=(RK/(psi4*n))**0.2
    return fft_kde(d,h,xg)
def botev(d,xg): return fft_kde(d,h_isj(d),xg)    # Botev diffusion estimator (ISJ optimal diffusion time)
def abramson(d,xg):                               # Abramson (1982) adaptive sqrt-law variable bandwidth
    n=len(d); h0=h_silverman(d); fp=fft_kde(d,h0,xg)
    fpx=np.clip(np.interp(d,xg,fp),1e-12,None); g=np.exp(np.mean(np.log(fpx)))
    hi=np.clip(h0*np.sqrt(g/fpx),0.2*h0,5*h0)
    F=np.exp(-0.5*((xg[None,:]-d[:,None])/hi[:,None])**2)/(hi[:,None]*np.sqrt(2*np.pi))
    return F.mean(axis=0)

# ---------------- AD diagnostic + symmetrizing-warp enhancement ----------------
def coherent_effdim(d,xg):
    """Effective dimension D_2 (participation ratio) of the stripped (coherent) squared-ECF
    spectrum; elevated for spike-plus-tail (kurtotic, strongly-skewed) densities."""
    p,dx=_binned(d,xg); ecf2=np.abs(np.fft.rfft(p))**2
    nu=_floor_residue(ecf2); S=np.clip(ecf2-nu,0,None); pk=S/max(S.sum(),1e-15)
    return float(1.0/np.sum(pk**2))
def _yj_T(x,lam):
    x=np.asarray(x,float); o=np.empty_like(x); pos=x>=0
    with np.errstate(all="ignore"):
        o[pos]=(np.power(x[pos]+1,lam)-1)/lam if lam!=0 else np.log(x[pos]+1)
        o[~pos]=-(np.power(1-x[~pos],2-lam)-1)/(2-lam) if lam!=2 else -np.log(1-x[~pos])
    return o
def _yj_Tp(x,lam):
    with np.errstate(all="ignore"):
        return np.where(np.asarray(x)>=0,np.power(np.asarray(x,float)+1,lam-1),np.power(1-np.asarray(x,float),1-lam))
def ad_warp(d,xg):
    """AD-warp KDE for asymmetric/heavy-tailed targets: a monotone symmetrizing transform
    restores the translation-stationarity the cyclic group assumes, AD-Wiener estimates in
    the warped domain, and the Jacobian maps back. Gate on a skewness/kurtosis probe in use."""
    from scipy.stats import yeojohnson
    z,lam=yeojohnson(d); pad=0.25*(z.max()-z.min()); zg=np.linspace(z.min()-pad,z.max()+pad,1600)
    fz=ad_wiener(z,zg,"simple")
    fx=np.interp(_yj_T(xg,lam),zg,fz,left=0,right=0)*_yj_Tp(xg,lam)
    s=np.trapezoid(fx,xg); return fx/s if s>0 else fx

# ---------------- dilation-group (wavelet) construction ----------------
try:
    import pywt; HAVE_PYWT=True
except Exception: HAVE_PYWT=False
def _wthr(f0,wave,lam):
    co=pywt.wavedec(f0,wave,mode="periodization")
    out=[co[0]]+[pywt.threshold(c,lam,mode="soft") for c in co[1:]]
    return pywt.waverec(out,wave,mode="periodization")[:len(f0)]
def ad_wavelet(d,xg,wave="sym6",cfac=0.8,nshift=12):
    """Dilation-group AD KDE. The translation group's eigenbasis is Fourier; the dilation
    (scaling) group's is a wavelet multiresolution. The coherent/residue split of the
    framework becomes wavelet-coefficient soft-thresholding of the empirical density: the
    coarse scaling approximation is the linear (smooth) part, detail coefficients above the
    white floor are the structured part, and the rest is the residue. Noise scale is the std
    of the finest (noise-dominated) detail levels (the empirical-density coefficients are
    sparse and non-Gaussian, so a robust MAD scale underestimates the floor). Cycle-spinning
    over nshift circular shifts makes the estimate translation-invariant."""
    if not HAVE_PYWT: return ad_wiener(d,xg,"simple")
    n=len(d); p,dx=_binned(d,xg); M=len(xg); f0=p/dx
    co=pywt.wavedec(f0,wave,mode="periodization")
    sig=np.median([np.std(c) for c in co[-2:]]); lam=cfac*sig*np.sqrt(2*np.log(M))
    acc=np.zeros(M)
    for s in np.linspace(0,M,nshift,endpoint=False).astype(int):
        acc+=np.roll(_wthr(np.roll(f0,s),wave,lam),-s)
    f=np.clip(acc/nshift,0,None); z=np.trapezoid(f,xg); return f/z if z>0 else f

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--strip",choices=["simple","residue"],default="simple",
                    help="Floor-stripping for AD methods (default simple = 1/n; residue = AD+AR).")
    ap.add_argument("--flatten-uniform",action="store_true",
                    help="Snap an AD-Wiener estimate to a perfect uniform when its non-DC ripple is "
                         "below the n-scaled uniformity threshold (default off).")
    ap.add_argument("--flatten-alpha",type=float,default=0.01,
                    help="Significance for the uniformity gate (default 0.01).")
    args=ap.parse_args()
    if args.flatten_uniform:
        xgu=np.linspace(0.0,1.0,1024); du=np.random.default_rng(0).uniform(0,1,8000)
        ok,stat,tau=uniform_gate(du,xgu,args.flatten_alpha)
        print("uniformity gate on a uniform sample (n=8000): fired=%s (stat %.2f vs threshold %.2f)\n" % (ok,stat,tau))
    names=list(MW.keys()); Ns=[200,2000]; R={200:60,2000:40}; rng=np.random.default_rng(2024)

    # ---- clean benchmark: AD methods (both strips) vs baselines ----
    res={}
    for name in names:
        xg=np.linspace(*RANGE[name],1600); hg=np.geomspace(0.01,3,120)
        for n in Ns:
            acc={k:[] for k in("bestfix","silver","botev","chiu","abram","bw_s","bw_r","wi_s","wi_r")}
            for _ in range(R[n]):
                d=sample(name,n,rng)
                acc["bestfix"].append(min(ise(name,fft_kde(d,h,xg),xg) for h in hg))
                acc["silver"].append(ise(name,fft_kde(d,h_silverman(d),xg),xg))
                acc["botev"].append(ise(name,botev(d,xg),xg)); acc["chiu"].append(ise(name,chiu(d,xg),xg))
                acc["abram"].append(ise(name,abramson(d,xg),xg))
                acc["bw_s"].append(ise(name,ad_bw(d,xg,"simple"),xg)); acc["bw_r"].append(ise(name,ad_bw(d,xg,"residue"),xg))
                acc["wi_s"].append(ise(name,ad_wiener(d,xg,"simple"),xg)); acc["wi_r"].append(ise(name,ad_wiener(d,xg,"residue"),xg))
            res[(name,n)]={k:float(np.mean(v)) for k,v in acc.items()}
    print("=== CLEAN DATA: mean ISE x1e3 ===")
    hdr=("density","n","bestfix","Silver","Botev","Chiu","Abram","ADbw","ADbwR","ADWi","ADWiR")
    print("%-15s %5s |"%(hdr[0],hdr[1])+" ".join("%7s"%h for h in hdr[2:]))
    for name in names:
        for n in Ns:
            r=res[(name,n)]; cand={k:r[k] for k in("silver","botev","chiu","abram","bw_s","bw_r","wi_s","wi_r")}
            b=min(cand,key=cand.get)
            cells=[]
            for k in("bestfix","silver","botev","chiu","abram","bw_s","bw_r","wi_s","wi_r"):
                s="%7.2f"%(r[k]*1e3); cells.append(("["+s.strip()+"]").rjust(7) if k==b else s)
            print("%-15s %5d | %s"%(name,n," ".join(cells)))

    # ---- heaped-data robustness: simple vs residue strip ----
    print("\n=== HEAPED DATA (rounded to 0.1), n=2000: simple vs residue strip, mean ISE x1e3 ===")
    print("%-15s | %15s %15s"%("density","AD-bw  s / r","AD-Wiener s / r"))
    heap={}
    for name in names:
        xg=np.linspace(*RANGE[name],1600); a={k:[] for k in("bs","br","ws","wr")}
        for _ in range(40):
            d=np.round(sample(name,2000,rng),1)
            a["bs"].append(ise(name,ad_bw(d,xg,"simple"),xg)); a["br"].append(ise(name,ad_bw(d,xg,"residue"),xg))
            a["ws"].append(ise(name,ad_wiener(d,xg,"simple"),xg)); a["wr"].append(ise(name,ad_wiener(d,xg,"residue"),xg))
        heap[name]={k:float(np.mean(v)) for k,v in a.items()}
        print("%-15s | %6.2f / %6.2f  %7.2f / %6.2f"%(name,1e3*heap[name]["bs"],1e3*heap[name]["br"],1e3*heap[name]["ws"],1e3*heap[name]["wr"]))

    # ---- diagnostic + symmetrizing-warp on asymmetric/heavy-tailed targets ----
    print("\n=== AD diagnostic (coherent effective dim D2) and symmetrizing warp, n=2000 ===")
    print("%-16s %8s | %12s %12s %8s"%("density","D2_coh","AD-Wiener","AD-warp","change"))
    warp={}
    for name in names:
        xg=np.linspace(*RANGE[name],1600); dd=[]; aw=[]; ww=[]
        for _ in range(40):
            d=sample(name,2000,rng); dd.append(coherent_effdim(d,xg))
            aw.append(ise(name,ad_wiener(d,xg,"simple"),xg)); ww.append(ise(name,ad_warp(d,xg),xg))
        warp[name]=(float(np.mean(dd)),float(np.mean(aw)),float(np.mean(ww)))
        ch=100*(np.mean(ww)-np.mean(aw))/np.mean(aw)
        print("%-16s %8.1f | %12.2f %12.2f %+7.0f%%"%(name,warp[name][0],1e3*warp[name][1],1e3*warp[name][2],ch))

    # ---- dilation-group wavelet construction vs spectral AD-Wiener ----
    print("\n=== Dilation-group wavelet vs spectral AD-Wiener, n=2000 (mean ISE x1e3) ===")
    print("%-16s %10s %10s %10s"%("density","best-fix","AD-Wiener","AD-wavelet"))
    def _epdf(x): return np.where(x>=0,np.exp(-x),0.0)
    for name in ["Gaussian","Kurtotic","Claw","Smooth comb","Discrete comb","Strongly skewed"]:
        xg=np.linspace(*RANGE[name],2048); hg=np.geomspace(0.01,3,70); bf=[];aw=[];wv=[]
        for _ in range(40):
            d=sample(name,2000,rng)
            bf.append(min(ise(name,fft_kde(d,h,xg),xg) for h in hg))
            aw.append(ise(name,ad_wiener(d,xg,"simple"),xg)); wv.append(ise(name,ad_wavelet(d,xg),xg))
        print("%-16s %10.2f %10.2f %10.2f"%(name,1e3*np.mean(bf),1e3*np.mean(aw),1e3*np.mean(wv)))
    xg=np.linspace(-1,9,2048); hg=np.geomspace(0.01,3,70); bf=[];aw=[];wv=[]
    for _ in range(40):
        d=rng.exponential(1,2000); ie=lambda f:float(np.trapezoid((f-_epdf(xg))**2,xg))
        bf.append(min(ie(fft_kde(d,h,xg)) for h in hg)); aw.append(ie(ad_wiener(d,xg,"simple"))); wv.append(ie(ad_wavelet(d,xg)))
    print("%-16s %10.2f %10.2f %10.2f  (jump at 0)"%("Exponential",1e3*np.mean(bf),1e3*np.mean(aw),1e3*np.mean(wv)))

    def fits(axes,cases,seed,strip="simple"):
        rg=np.random.default_rng(seed)
        for ax,name in zip(axes,cases):
            xg=np.linspace(*RANGE[name],1600); d=sample(name,2000,rg)
            ax.plot(xg,pdf(name,xg),color='0.0',lw=1.3,label="true")
            ax.plot(xg,fft_kde(d,h_silverman(d),xg),'--',color='0.6',lw=1.0,label="Silverman")
            ax.plot(xg,botev(d,xg),':',color='0.4',lw=1.0,label="Botev")
            ax.plot(xg,ad_wiener(d,xg,strip),'-',color='0.0',lw=0.9,alpha=0.6,label="AD-Wiener")
            ax.set_title(name,fontsize=8.5); ax.set_xlabel("x"); ax.set_yticks([])
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.8)); fits(ax,["Claw","Strongly skewed"],7,args.strip)
    ax[0].set_ylabel("density"); ax[0].legend(fontsize=6.5,frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_kde_examples.pdf")); plt.close()
    fig,ax=plt.subplots(1,3,figsize=(7.4,2.5)); fits(ax,["Kurtotic","Asym. claw","Discrete comb"],11,args.strip)
    ax[0].set_ylabel("density"); ax[1].legend(fontsize=6.3,frameon=False,loc="upper right")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_kde_hard.pdf")); plt.close()
    # heaped figure: simple vs residue on a density where 1/n fails
    rg=np.random.default_rng(5); fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
    for axi,name in zip(ax,["Kurtotic","Strongly skewed"]):
        xg=np.linspace(*RANGE[name],1600); d=np.round(sample(name,2000,rg),1)
        axi.plot(xg,pdf(name,xg),color='0.0',lw=1.3,label="true")
        axi.plot(xg,ad_wiener(d,xg,"simple"),'--',color='0.55',lw=1.1,label="AD-Wiener (simple 1/n)")
        axi.plot(xg,ad_wiener(d,xg,"residue"),'-',color='0.0',lw=1.0,alpha=0.7,label="AD-Wiener (residue)")
        axi.set_title(name+"  (heaped data)",fontsize=8.5); axi.set_xlabel("x"); axi.set_yticks([])
    ax[0].set_ylabel("density"); ax[0].legend(fontsize=6.3,frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_kde_heaped.pdf")); plt.close()
    print("\nfigures written: fig_kde_examples.pdf, fig_kde_hard.pdf, fig_kde_heaped.pdf")
