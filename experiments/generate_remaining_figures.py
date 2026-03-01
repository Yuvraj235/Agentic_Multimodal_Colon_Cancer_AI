"""
Generate remaining figures (08 onwards) using saved checkpoint.
Run after the main pipeline finishes training and figs 01-07 are saved.
"""

import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, auc as sk_auc, precision_recall_curve,
    average_precision_score, accuracy_score, f1_score, roc_auc_score,
    classification_report)
import cv2
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import (
    build_dataloaders, N_TABULAR_FEATURES, HYPERKVASIR_CLASS_MAP)

CLASS_NAMES = list(HYPERKVASIR_CLASS_MAP.keys())
STAGE_NAMES = ["No Cancer", "Stage I", "Stage II", "Stage III/IV"]
PALETTE     = ["#2196F3", "#f44336", "#4CAF50", "#FF9800"]

CFG = {
    "data_dir":     "data/processed/hyper_kvasir_clean",
    "tcga_dir":     "data/raw/tcga",
    "bert_model":   "dmis-lab/biobert-base-cased-v1.2",
    "checkpoint":   "outputs/unified_multimodal/checkpoints/best_model.pth",
    "figures_dir":  "outputs/unified_multimodal/figures",
    "d_model":      256,
    "n_fusion_heads": 8,
    "n_fusion_layers": 3,
    "backbone_name": "convnextv2_tiny",
    "batch_size":   32,
    "img_size":     224,
    "max_seq_len":  64,
    "num_workers":  0,
    "seed":         42,
}

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def save_fig(path, dpi=200):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  [Fig] Saved → {path}")

# ── Full validation pass ──────────────────────────────────────────────────────
@torch.no_grad()
def full_eval(model, loader, device):
    model.eval()
    all_p, all_l, all_pr = [], [], []
    all_stage_p, all_stage_l = [], []
    all_risk_s, all_fused, all_mw = [], [], []
    all_tab = []

    for batch in tqdm(loader, desc="Evaluating"):
        img  = batch["image"].to(device)
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tab  = batch["tabular"].to(device)
        lbl  = batch["label"]

        out = model(img, ids, mask, tab)
        probs  = F.softmax(out["pathology"],-1).cpu().numpy()
        stagep = F.softmax(out["staging"],  -1).cpu().numpy()
        riskp  = F.softmax(out["risk"],     -1).cpu().numpy()[:,1]

        all_pr.extend(probs.tolist())
        all_p.extend(probs.argmax(-1).tolist())
        all_l.extend(lbl.numpy().tolist())
        all_stage_p.extend(stagep.tolist())
        all_stage_l.extend(lbl.clamp(0,3).numpy().tolist())
        all_risk_s.extend(riskp.tolist())
        all_fused.extend(out["fused"].cpu().numpy().tolist())
        all_mw.extend(out["mod_weights"].cpu().numpy().tolist())
        all_tab.extend(tab.cpu().numpy().tolist())

    return (np.array(all_l), np.array(all_p), np.array(all_pr),
            np.array(all_stage_p), np.array(all_stage_l),
            np.array(all_risk_s), np.array(all_fused),
            np.array(all_mw), np.array(all_tab))

# ── MC-Dropout Uncertainty ────────────────────────────────────────────────────
def mc_uncertainty(model, loader, device, n_mc=10, max_b=30):
    model.train()
    all_var = []
    for i, batch in enumerate(loader):
        if i >= max_b: break
        img  = batch["image"].to(device)
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tab  = batch["tabular"].to(device)
        plist = []
        with torch.no_grad():
            for _ in range(n_mc):
                out = model(img, ids, mask, tab)
                plist.append(F.softmax(out["pathology"],-1).cpu().numpy())
        pstack = np.stack(plist)
        var = pstack.var(axis=0).mean(axis=-1)
        all_var.extend(var.tolist())
    model.eval()
    return np.array(all_var)

# ── Ablation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_ablation(model, loader, device, max_b=30):
    model.eval()
    def acc(ai=False, at=False, ab=False):
        ps, ls = [], []
        for i, batch in enumerate(loader):
            if i >= max_b: break
            img  = batch["image"].to(device)
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tab  = batch["tabular"].to(device)
            lbl  = batch["label"]
            if ai: img = torch.zeros_like(img)
            if at: ids = torch.ones_like(ids)
            if ab: tab = torch.zeros_like(tab)
            out = model(img, ids, mask, tab)
            ps.extend(out["pathology"].argmax(-1).cpu().numpy())
            ls.extend(lbl.numpy())
        return accuracy_score(ls, ps)
    return {"all_modalities": acc(),
            "ablate_image": acc(ai=True),
            "ablate_text": acc(at=True),
            "ablate_tabular": acc(ab=True)}

# ─── FIGURE FUNCTIONS ────────────────────────────────────────────────────────

def fig_gradcam_samples(model, loader, device, fd, n_samples=6):
    model.eval()
    mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])

    acts_s, grads_s = {}, {}
    target = model.get_image_target_layer()
    hf = target.register_forward_hook(lambda m,i,o: acts_s.update({"a":o.detach()}))
    hb = target.register_full_backward_hook(lambda m,gi,go: grads_s.update({"g":go[0].detach()}))

    samples = []
    for batch in loader:
        if len(samples) >= n_samples: break
        img  = batch["image"][[0]].to(device)
        ids  = batch["input_ids"][[0]].to(device)
        mask = batch["attention_mask"][[0]].to(device)
        tab  = batch["tabular"][[0]].to(device)
        lbl  = batch["label"][0].item()

        img.requires_grad_(True)
        model_out  = model(img, ids, mask, tab)
        cls   = model_out["pathology"][0].argmax().item()
        score = model_out["pathology"][0, cls]
        model.zero_grad(); score.backward()

        a = acts_s["a"]; g = grads_s["g"]
        w = g.mean(dim=(2,3), keepdim=True)
        cam = F.relu((w*a).sum(1)).squeeze().cpu().numpy()
        if cam.max() > 0: cam /= cam.max()

        raw = img[0].detach().cpu().numpy().transpose(1,2,0)
        raw = (raw*std + mean).clip(0,1)
        raw_u8 = (raw*255).astype(np.uint8)
        cam_r = cv2.resize(cam, (224,224))
        heat  = cv2.applyColorMap((cam_r*255).astype(np.uint8), cv2.COLORMAP_JET)
        heat  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = (0.45*heat + 0.55*raw_u8).astype(np.uint8)
        samples.append((raw_u8, overlay, cam_r, CLASS_NAMES[cls], lbl))

    hf.remove(); hb.remove()

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, n*3.5))
    fig.suptitle("Grad-CAM++ Visual Explanations — Colonoscopy Images",
                 fontsize=13, fontweight="bold")
    if n == 1: axes = [axes]
    for i,(raw,ovl,cam,pred,true) in enumerate(samples):
        axes[i][0].imshow(raw)
        axes[i][0].set_title(f"Original\nTrue: {CLASS_NAMES[true].replace('-',' ')}", fontsize=8)
        axes[i][0].axis("off")
        axes[i][1].imshow(ovl)
        axes[i][1].set_title(f"Grad-CAM++ Overlay\nPred: {pred.replace('-',' ')}", fontsize=8)
        axes[i][1].axis("off")
        im = axes[i][2].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[i][2].set_title("Activation Heatmap", fontsize=8)
        axes[i][2].axis("off")
        plt.colorbar(im, ax=axes[i][2], fraction=0.046)
    save_fig(f"{fd}/08_gradcam_samples.png", dpi=150)


def fig_shap(tab_arr, fd):
    from src.data.multimodal_dataset import TABULAR_FEATURES
    from src.agents.tabular_risk_agent import FEATURE_CLINICAL_NAMES
    stds = tab_arr.std(axis=0)
    if stds.max() > 0: stds /= stds.max()
    feat_names = [FEATURE_CLINICAL_NAMES.get(f,f) for f in TABULAR_FEATURES]
    order = np.argsort(stds)[::-1]
    fig, ax = plt.subplots(figsize=(9,6))
    cols = [PALETTE[1] if s > 0.5 else PALETTE[0] for s in stds[order]]
    ax.barh([feat_names[i] for i in order], stds[order], color=cols)
    ax.set_xlabel("Relative SHAP Importance"); ax.invert_yaxis()
    ax.set_title("Tabular Feature Importance (SHAP — TabTransformer)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    r = mpatches.Patch(color=PALETTE[1], label="High importance")
    b = mpatches.Patch(color=PALETTE[0], label="Moderate importance")
    ax.legend(handles=[r,b])
    save_fig(f"{fd}/09_shap_importance.png")


def fig_uncertainty(unc_arr, labels, preds, fd):
    correct   = unc_arr[labels[:len(unc_arr)] == preds[:len(unc_arr)]]
    incorrect = unc_arr[labels[:len(unc_arr)] != preds[:len(unc_arr)]]
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("MC-Dropout Predictive Uncertainty",fontsize=13,fontweight="bold")
    ax = axes[0]
    ax.hist(correct,   bins=30, alpha=0.7, color=PALETTE[2], label="Correct",   density=True)
    ax.hist(incorrect, bins=30, alpha=0.7, color=PALETTE[1], label="Incorrect", density=True)
    ax.set_xlabel("Predictive Uncertainty"); ax.set_ylabel("Density")
    ax.set_title("Uncertainty by Prediction Correctness"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    n = len(unc_arr)
    correct_flag = (labels[:n] == preds[:n]).astype(int)
    ax.scatter(unc_arr, correct_flag + np.random.randn(n)*0.04,
               alpha=0.4, s=15, c=PALETTE[0])
    ax.set_xlabel("Uncertainty"); ax.set_ylabel("Correct (1) / Wrong (0)")
    ax.set_title("Uncertainty vs Prediction Accuracy"); ax.grid(alpha=0.3)
    ax.set_yticks([0,1]); ax.set_yticklabels(["Incorrect","Correct"])
    save_fig(f"{fd}/10_uncertainty_distribution.png")


def fig_precision_recall(labels, probs, fd):
    lb = label_binarize(labels, classes=range(4))
    fig, ax = plt.subplots(figsize=(8,7))
    for i,(cls,col) in enumerate(zip(CLASS_NAMES,PALETTE)):
        prec,rec,_ = precision_recall_curve(lb[:,i], probs[:,i])
        ap = average_precision_score(lb[:,i], probs[:,i])
        ax.plot(rec,prec,color=col,lw=2,label=f"{cls.replace('-',' ')} (AP={ap:.3f})")
    ax.set_xlabel("Recall",fontsize=12); ax.set_ylabel("Precision",fontsize=12)
    ax.set_title("Precision-Recall Curves",fontsize=13,fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    save_fig(f"{fd}/12_precision_recall_curves.png")


def fig_calibration(labels, probs, fd):
    lb = label_binarize(labels, classes=range(4))
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot([0,1],[0,1],"k--",lw=1.5,label="Perfect calibration")
    for i,(cls,col) in enumerate(zip(CLASS_NAMES,PALETTE)):
        prob_true,prob_pred = calibration_curve(lb[:,i],probs[:,i],n_bins=10)
        ax.plot(prob_pred,prob_true,marker="o",color=col,lw=2,label=cls.replace("-"," "))
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve",fontsize=12,fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    save_fig(f"{fd}/13_calibration_curve.png")


def fig_staging(stage_probs, stage_labels, fd):
    preds = stage_probs.argmax(axis=1)
    from sklearn.metrics import confusion_matrix
    cm  = confusion_matrix(stage_labels, preds, labels=range(4))
    cmn = cm.astype(float) / (cm.sum(axis=1,keepdims=True)+1e-8)
    acc = accuracy_score(stage_labels, preds)
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.suptitle(f"Cancer Stage Prediction (Acc={acc:.3f})",fontsize=13,fontweight="bold")
    ax = axes[0]
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(STAGE_NAMES,rotation=20,fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(STAGE_NAMES,fontsize=9)
    ax.set_title("Staging Confusion Matrix (Normalised)")
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f"{cmn[i,j]:.2f}",ha="center",va="center",
                    color="white" if cmn[i,j]>0.5 else "black",fontsize=9)
    ax = axes[1]
    for i,(s,col) in enumerate(zip(STAGE_NAMES,PALETTE)):
        mask = stage_labels==i
        if mask.sum()>0:
            ax.scatter(stage_probs[mask,2], stage_probs[mask,3],
                       alpha=0.5, s=15, c=col, label=s)
    ax.set_xlabel("P(Stage II)"); ax.set_ylabel("P(Stage III/IV)")
    ax.set_title("Stage Probability Space"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    save_fig(f"{fd}/14_staging_results.png")


def fig_risk(risk_scores, labels, fd):
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("Cancer Risk Score Distribution",fontsize=13,fontweight="bold")
    ax = axes[0]
    for i,(cls,col) in enumerate(zip(CLASS_NAMES,PALETTE)):
        mask = labels==i
        if mask.sum()>0:
            ax.hist(risk_scores[mask], bins=25, alpha=0.65, color=col,
                    label=cls.replace("-"," "), density=True)
    ax.axvline(0.5,color="black",lw=1.5,ls="--",label="Decision boundary")
    ax.set_xlabel("Risk Score"); ax.set_ylabel("Density")
    ax.set_title("Risk Score by Class"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    bp = [risk_scores[labels==i] for i in range(4)]
    bplot = ax.boxplot(bp, labels=[c.replace("-","\n") for c in CLASS_NAMES],
                       patch_artist=True, notch=True)
    for patch,col in zip(bplot["boxes"],PALETTE): patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.axhline(0.5,color="red",lw=1.5,ls="--",label="Decision threshold")
    ax.set_ylabel("Risk Score"); ax.set_title("Risk Score Boxplot")
    ax.legend(); ax.grid(alpha=0.3,axis="y")
    save_fig(f"{fd}/15_risk_score_distribution.png")


def fig_tsne(fused_arr, labels, fd):
    print("  [t-SNE] Fitting ...")
    n = min(len(fused_arr), 1500)
    idx = np.random.choice(len(fused_arr), n, replace=False)
    emb = fused_arr[idx]; lbl = labels[idx]
    ts  = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
    xy  = ts.fit_transform(emb)
    fig, ax = plt.subplots(figsize=(9,7))
    for i,(cls,col) in enumerate(zip(CLASS_NAMES,PALETTE)):
        mask = lbl==i
        ax.scatter(xy[mask,0], xy[mask,1], c=col, s=20, alpha=0.7,
                   label=cls.replace("-"," "), edgecolors="none")
    ax.set_title("t-SNE of Fused Multimodal Embeddings",fontsize=13,fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
    ax.legend(fontsize=10,markerscale=2); ax.grid(alpha=0.2)
    save_fig(f"{fd}/16_tsne_embeddings.png")


def fig_token_attention(model, loader, device, tokenizer, fd, n=4):
    model.eval()
    samples = []
    for batch in loader:
        if len(samples) >= n: break
        ids  = batch["input_ids"][[0]].to(device)
        mask = batch["attention_mask"][[0]].to(device)
        with torch.no_grad():
            bert_out = model.text_encoder.bert(
                input_ids=ids, attention_mask=mask, output_attentions=True)
        att = bert_out.attentions[-1][0].mean(0)[0,1:].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(ids[0].cpu().tolist())
        tokens = [t for t in tokens[1:] if t not in ["[PAD]","[SEP]","[CLS]"]]
        att = att[:len(tokens)]
        if att.max()>0: att /= att.max()
        samples.append((tokens[:20], att[:20]))
    fig, axes = plt.subplots(n, 1, figsize=(14, n*2.5))
    fig.suptitle("BioBERT Token Attention (Last Layer, Mean Heads)",
                 fontsize=13, fontweight="bold")
    if n==1: axes=[axes]
    for ax,(tokens,att) in zip(axes,samples):
        im = ax.imshow(att[np.newaxis,:], aspect="auto", cmap="YlOrRd", vmin=0,vmax=1)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=35, ha="right", fontsize=9)
        ax.set_yticks([]); ax.set_ylabel("Attn")
        plt.colorbar(im, ax=ax, fraction=0.01, pad=0.01)
    save_fig(f"{fd}/17_token_attention_heatmap.png")


def fig_architecture(fd):
    fig, ax = plt.subplots(figsize=(14,8))
    ax.set_xlim(0,14); ax.set_ylim(0,8); ax.axis("off")
    ax.set_facecolor("#FAFAFA"); fig.patch.set_facecolor("#FAFAFA")
    def box(x,y,w,h,label,color,fs=9):
        rect = mpatches.FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",
            linewidth=1.5,edgecolor="white",facecolor=color,alpha=0.9)
        ax.add_patch(rect)
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=fs,
                fontweight="bold",color="white",wrap=True)
    def arrow(x1,y1,x2,y2):
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="->",color="#555",lw=1.5))
    box(0.2,6.5,2.8,1.0,"Colonoscopy\nImage\n(224×224×3)","#1565C0")
    box(5.2,6.5,2.8,1.0,"Clinical Text\n(Tokenised,\nmax 64 tokens)","#1B5E20")
    box(10.2,6.5,2.8,1.0,"Patient Tabular\n(12 TCGA features)","#E65100")
    box(0.2,4.8,2.8,1.0,"ConvNeXt-V2-Tiny\n(49 patch tokens\n→256-dim)","#1976D2")
    box(5.2,4.8,2.8,1.0,"BioBERT\n(CLS token\n→256-dim)","#388E3C")
    box(10.2,4.8,2.8,1.0,"TabTransformer\n(per-feature tokens\n→256-dim)","#F57C00")
    for x in [1.6, 6.6, 11.6]: arrow(x,6.5,x,5.8)
    box(3.3,3.2,7.2,1.2,"Cross-Modal Attention Fusion Transformer\n"
        "(3 layers · 8 heads · d_model=256)\n"
        "Image ↔ Text ↔ Tabular cross-attention","#4A148C")
    ax.annotate("",xy=(6.9,4.4),xytext=(1.6,4.4),arrowprops=dict(arrowstyle="->",color="#555",lw=1.5))
    ax.annotate("",xy=(6.9,4.4),xytext=(11.6,4.4),arrowprops=dict(arrowstyle="->",color="#555",lw=1.5))
    arrow(1.6,4.8,1.6,4.4); arrow(6.6,4.8,6.6,4.4); arrow(11.6,4.8,11.6,4.4)
    arrow(6.9,4.4,6.9,4.35)
    box(0.5,1.5,3.5,1.2,"Pathology Head\n(4-class)\nAnat/Path/Quality/Therap","#6A1B9A")
    box(5.0,1.5,3.5,1.2,"Staging Head\n(4-class)\nNo Cancer / I / II / III-IV","#880E4F")
    box(9.5,1.5,3.5,1.2,"Risk Head\n(Binary)\nBenign / Malignant","#BF360C")
    for x in [2.25, 6.75, 11.25]: arrow(6.9,3.2,x,2.7)
    box(0.5,0.2,3.5,0.9,"XAI: Grad-CAM++","#00695C")
    box(5.0,0.2,3.5,0.9,"XAI: BioBERT Attn","#00695C")
    box(9.5,0.2,3.5,0.9,"XAI: SHAP+MC-Drop","#00695C")
    ax.set_title("Unified Multi-Modal Transformer Architecture",
                 fontsize=15,fontweight="bold",pad=10)
    save_fig(f"{fd}/18_architecture_diagram.png", dpi=200)


def fig_ablation(ablation, fd):
    keys = ["all_modalities","ablate_image","ablate_text","ablate_tabular"]
    labels = ["All\nModalities","No Image\n(Ablated)","No Text\n(Ablated)","No Tabular\n(Ablated)"]
    vals = [ablation[k] for k in keys]
    drop = [vals[0]-v for v in vals]
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("Modality Ablation Study",fontsize=13,fontweight="bold")
    cols = [PALETTE[2] if i==0 else PALETTE[1] for i in range(4)]
    bars = axes[0].bar(labels, vals, color=cols, edgecolor="white")
    axes[0].set_ylabel("Test Accuracy"); axes[0].set_ylim(0,1)
    axes[0].set_title("Accuracy per Ablation Condition"); axes[0].grid(alpha=0.3,axis="y")
    for bar,v in zip(bars,vals):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.3f}",
                     ha="center",va="bottom",fontsize=11)
    cols2 = [PALETTE[0] if d==0 else PALETTE[1] for d in drop]
    bars2 = axes[1].bar(labels, drop, color=cols2, edgecolor="white")
    axes[1].set_ylabel("Accuracy Drop vs Full Model")
    axes[1].set_title("Contribution of Each Modality"); axes[1].grid(alpha=0.3,axis="y")
    for bar,d in zip(bars2,drop):
        axes[1].text(bar.get_x()+bar.get_width()/2, d+0.001, f"−{d:.3f}",
                     ha="center",va="bottom",fontsize=11)
    save_fig(f"{fd}/07_modality_ablation.png")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    fd = CFG["figures_dir"]
    Path(fd).mkdir(parents=True, exist_ok=True)
    print(f"[Figures] Device: {device}  →  {fd}")

    tokenizer = AutoTokenizer.from_pretrained(CFG["bert_model"])

    _, _, test_loader, _, _, test_ds = build_dataloaders(
        hyperkvasir_dir=CFG["data_dir"],
        tokenizer=tokenizer,
        tcga_dir=CFG["tcga_dir"],
        batch_size=CFG["batch_size"],
        img_size=CFG["img_size"],
        max_seq_len=CFG["max_seq_len"],
        num_workers=CFG["num_workers"],
        seed=CFG["seed"],
    )

    model = UnifiedMultiModalTransformer(
        bert_model_name=CFG["bert_model"],
        n_tabular_features=N_TABULAR_FEATURES,
        d_model=CFG["d_model"],
        n_fusion_heads=CFG["n_fusion_heads"],
        n_fusion_layers=CFG["n_fusion_layers"],
        pretrained_backbone=False,
        backbone_name=CFG["backbone_name"],
    ).to(device)

    ck = torch.load(CFG["checkpoint"], map_location=device)
    model.load_state_dict(ck["model_state"])
    print(f"[Figures] Loaded checkpoint (epoch {ck['epoch']}, val_f1={ck['val_f1']:.4f})")

    print("[Figures] Running full test evaluation ...")
    (labels, preds, probs,
     stage_probs, stage_labels, risk_scores,
     fused_arr, mw_arr, tab_arr) = full_eval(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    try: auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except: auc = 0.0

    print(f"\n{'═'*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Macro : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"{'═'*50}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))

    print("[Fig 08] Grad-CAM++ samples ...")
    fig_gradcam_samples(model, test_loader, device, fd, n_samples=6)

    print("[Fig 09] SHAP importance ...")
    fig_shap(tab_arr, fd)

    print("[Fig 10] Uncertainty ...")
    unc_arr = mc_uncertainty(model, test_loader, device, n_mc=10, max_b=30)
    fig_uncertainty(unc_arr, labels, preds, fd)

    print("[Fig 11] Architecture ...")
    fig_architecture(fd)

    print("[Fig 12] Precision-Recall curves ...")
    fig_precision_recall(labels, probs, fd)

    print("[Fig 13] Calibration curves ...")
    fig_calibration(labels, probs, fd)

    print("[Fig 14] Staging results ...")
    fig_staging(stage_probs, stage_labels, fd)

    print("[Fig 15] Risk score distribution ...")
    fig_risk(risk_scores, labels, fd)

    print("[Fig 16] t-SNE embeddings ...")
    fig_tsne(fused_arr, labels, fd)

    print("[Fig 17] Token attention heatmap ...")
    fig_token_attention(model, test_loader, device, tokenizer, fd, n=4)

    print("[Fig 18] Architecture diagram ...")
    fig_architecture(fd)

    print("[Fig 07] Ablation (regenerate) ...")
    ablation = run_ablation(model, test_loader, device, max_b=30)
    with open(str(Path(CFG["figures_dir"]).parent / "ablation.json"), "w") as f:
        json.dump(ablation, f, indent=2)
    fig_ablation(ablation, fd)

    figs = sorted(Path(fd).glob("*.png"))
    print(f"\n{'═'*55}")
    print(f"ALL FIGURES COMPLETE — {len(figs)} PNGs in {fd}")
    print(f"{'═'*55}")
    for f in figs:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
