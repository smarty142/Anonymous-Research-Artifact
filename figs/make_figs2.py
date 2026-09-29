import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.size":8,"font.family":"serif"})
# Fig 3a: adaptive AUC with CIs
names=["text","provenance","pattern","text+prov.","text+pattern","prov.+pattern","all three","LR head (tool-call)","LR head (InjecAgent)"]
auc=[0.778,0.627,0.651,0.782,0.791,0.699,0.795,0.795,0.782]
lo=[0.748,0.608,0.630,0.753,0.762,0.676,0.766,0.766,0.752]
hi=[0.808,0.649,0.673,0.812,0.822,0.723,0.825,0.825,0.812]
fig,ax=plt.subplots(figsize=(3.3,2.6))
y=np.arange(len(names))[::-1]
cols=["#4c72b0"]*6+["#2a9d8f","#999999","#999999"]
ax.barh(y,auc,color=cols,height=0.6)
ax.errorbar(auc,y,xerr=[np.array(auc)-np.array(lo),np.array(hi)-np.array(auc)],fmt="none",ecolor="black",capsize=2,lw=0.8)
ax.set_yticks(y,names); ax.set_xlim(0.5,0.85); ax.set_xlabel("AUC (adaptive bench, 95% CI)")
ax.axvline(0.778,ls="--",lw=0.7,color="#4c72b0")
ax.set_title("(a) Fused margin under adaptation",fontsize=8)
fig.tight_layout(); fig.savefig("fig_adaptive_auc.pdf")
# Fig 3b: per-shape detection
shapes=["tool-argument","emitted-text","data-flow"]
fused=[47.9,79.0,20.0]; nopat=[39.6,66.0,20.0]; evade=[26.5,63.2,0.0]
fig,ax=plt.subplots(figsize=(3.3,2.6))
x=np.arange(3); w=0.26
ax.bar(x-w,fused,w,label="fused, all 440 draws",color="#2a9d8f")
ax.bar(x,nopat,w,label="without pattern channel",color="#4c72b0")
ax.bar(x+w,evade,w,label="fused, 307 signature-evading draws",color="#e76f51")
for xi,v in zip(list(x-w)+list(x)+list(x+w),fused+nopat+evade):
    ax.text(xi,v+1.5,f"{v:.0f}",ha="center",fontsize=6.5)
ax.set_xticks(x,shapes); ax.set_ylabel("detected at 7% FPR (%)"); ax.set_ylim(0,118)
ax.legend(fontsize=6,loc="upper left",frameon=False,ncol=1)
ax.set_title("(b) Detection by attack shape",fontsize=8)
fig.tight_layout(); fig.savefig("fig_adaptive_shape.pdf")
# Fig (appendix): contamination
fig,ax=plt.subplots(figsize=(3.4,2.2))
m=["benign FPR @0.5","TPR @7% FPR","AUC x100"]; pre=[81.4,1.1,51.8]; post=[0.0,83.4,91.6]
x=np.arange(3); ax.bar(x-0.18,pre,0.36,label="pre-fix",color="#e76f51"); ax.bar(x+0.18,post,0.36,label="post-fix",color="#2a9d8f")
for xi,v in zip(list(x-0.18)+list(x+0.18),pre+post): ax.text(xi,v+1.5,f"{v:.1f}",ha="center",fontsize=6.5)
ax.set_xticks(x,m); ax.set_ylim(0,100); ax.legend(fontsize=7,frameon=False)
fig.tight_layout(); fig.savefig("fig_contamination.pdf"); print("ok")
