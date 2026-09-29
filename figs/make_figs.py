import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.size":8,"font.family":"serif"})
rows=["GLM-4.6","GLM-5.2","GPT-4o-mini","Claude Haiku 4.5","Llama Guard 4","full CaMeL (provenance)","signature layer (pattern)"]
cols=["InjecAgent","tool-call","AgentDojo"]
M=np.array([[79.1,23.8,21],[95.2,34.6,44],[97.9,27.1,84],[89.3,34.6,34],[46.0,10.8,70],[4.3,42.5,np.nan],[0,70.0,10]])
fig,ax=plt.subplots(figsize=(4.8,2.9))
im=ax.imshow(np.nan_to_num(M,nan=0),cmap="Blues",vmin=0,vmax=100,aspect="auto")
for i in range(M.shape[0]):
  for j in range(M.shape[1]):
    v=M[i,j]; t="n/a" if np.isnan(v) else f"{v:.0f}"
    ax.text(j,i,t,ha="center",va="center",color="white" if (not np.isnan(v) and v>55) else "black",fontsize=8)
ax.set_xticks(range(3),cols); ax.set_yticks(range(len(rows)),rows)
cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02); cb.set_label("TPR (%)")
fig.tight_layout(); fig.savefig("fig_regime_heatmap.pdf")
subs=["text","provenance","pattern","text + provenance","text + pattern","provenance + pattern","all three"]
A=np.array([[0.972,0.832,0.923],[0.503,0.713,np.nan],[0.500,0.656,0.152],[0.944,0.845,np.nan],[0.972,0.861,0.237],[0.503,0.754,np.nan],[0.944,0.874,np.nan]])
cols2=["InjecAgent","tool-call","AgentDojo"]
fig,ax=plt.subplots(figsize=(4.8,2.9))
im=ax.imshow(np.where(np.isnan(A),0.5,A),cmap="RdYlGn",vmin=0.5,vmax=1.0,aspect="auto")
for i in range(A.shape[0]):
  for j in range(A.shape[1]):
    v=A[i,j]; ax.text(j,i,"n/a" if np.isnan(v) else f"{v:.3f}",ha="center",va="center",fontsize=8)
ax.set_xticks(range(3),cols2); ax.set_yticks(range(len(subs)),subs)
cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02); cb.set_label("AUC (max-fusion)")
fig.tight_layout(); fig.savefig("fig_ablation_heatmap.pdf")
print("done")
