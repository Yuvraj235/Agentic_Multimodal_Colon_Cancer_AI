# How to start ColonAI

## The 1-click way (recommended)

**Double-click `run_app.command`** in this folder.

That's it. A Terminal window will open and the browser will launch the app at `http://localhost:8501`.
Keep the Terminal window open while you present. Press `Ctrl-C` in it when you're done.

---

## The 1-line command (Terminal users)

```bash
cd "/Users/yuvrajpratapsingh/Desktop/Agentic_Multimodal_Colon_Cancer_AI copy"
./run_app.command
```

Or, totally manual:

```bash
python3 -m streamlit run app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

---

## Sharing your screen

If you're presenting via Zoom / Meet / Teams, just share the **browser tab** (not the whole screen).

To let someone on your local network reach the app, use the **Network URL** that Streamlit prints (it looks like `http://192.168.x.x:8501`).
Don't share the **External URL** unless you're certain you want the app on the public internet.

---

## What if it doesn't start?

| Problem | Fix |
|---|---|
| `Permission denied` when double-clicking | Right-click → Open → Open. macOS will remember it. |
| `port 8501 is already in use` | The launcher tries 8502/8503 automatically. Or run `lsof -ti:8501 \| xargs kill -9` |
| `streamlit: command not found` | `pip3 install -r requirements.txt` (the launcher does this for you on first run) |
| Sidebar shows red "Model load failed" | Confirm `outputs/unified_multimodal/checkpoints/best_model.pth` exists. Without it the app falls into demo mode. |
| First analysis is slow | Normal — model loads ~20 s on first run, then caches. The Quick-demo cards trigger this loading early. |

---

## Once the app is up

1. Use the **Quick-demo** panel on Step 1 (loads a full case in one click).
2. Or read the in-app **Site Guide → How to Present** tab for the full demo script.
3. The full clinical PDF script lives at `outputs/ColonAI_Presentation_Script.pdf`.
