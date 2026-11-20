# Deploying TabulaML on Streamlit Community Cloud

The repository is ready to deploy as-is: `app.py` sits at the root next to the `tabulaml/` package, so no `pip install -e .` is needed.

## Files Streamlit Cloud uses
| File | Purpose |
|---|---|
| `requirements.txt` | Runtime packages with exact, tested versions (scikit-learn stays at 1.5.2) |
| `packages.txt` | `libgomp1`, the OpenMP runtime LightGBM needs on Linux |
| `.streamlit/config.toml` | Upload limit (200 MB), no usage stats, and the accent colour defined separately for `[theme.light]` and `[theme.dark]`. A single `[theme]` section would hide Streamlit's System / Light / Dark picker. |
| `.streamlit/secrets.toml.example` | Shows the optional `TABULAML_CLOUD = "1"` switch |

## Steps
1. Push `main` to GitHub (already at `github.com/Dinusha-Ekanayake/TabulaML`).
2. Sign in at **share.streamlit.io** with GitHub and click **Create app**, then **Deploy a public app from GitHub**.
3. Choose repository `Dinusha-Ekanayake/TabulaML`, branch `main`, main file `app.py`.
4. Open **Advanced settings**:
   - **Python version:** 3.12.
   - **Secrets:** paste `TABULAML_CLOUD = "1"`. This enables *cloud mode*, which lowers the upload cell limit to 400k and the SVM row limit to 5k to stay inside the ~1 GB memory limit.
5. Click **Deploy**. The first build takes about 5-10 minutes, because CatBoost, XGBoost and LightGBM wheels are large. Later pushes to `main` redeploy automatically.
6. Put the app URL in `README.md` as the live demo link.

## Before deploying
- Run the smoke test locally: `python scripts/smoke_deploy.py --cloud`. It should print `OK: health=ok, page=200`.
- Check in a fresh venv that the runtime set alone is enough: `pip install -r requirements.txt`, then `streamlit run app.py`.

## Limits and troubleshooting
| Symptom | Fix |
|---|---|
| `OSError: libgomp.so.1` (LightGBM) | Make sure `packages.txt` contains `libgomp1` |
| App restarts or "Oh no" during training | Memory limit hit: enable cloud mode, select fewer models, or use a smaller upload |
| Upload rejected | Cloud mode allows up to 400k cells; the Streamlit upload limit is 200 MB |
| `TypeError: fmin_l_bfgs_b() got an unexpected keyword argument 'disp'` (RF + ARIMA) | statsmodels older than 0.15 with scipy 1.18: keep `statsmodels==0.15.0` from `requirements.txt` |
| Training seems stuck | Training runs in the background; watch the progress card in step 4, or press **Cancel** to keep the finished models |
| Wrong scikit-learn version | Keep `scikit-learn==1.5.2`; newer versions break XGBoost/CatBoost inside stacking |
| Telco dataset missing | It needs a Kaggle login, so it is not bundled; fetch it locally with `scripts/fetch_public_datasets.py telco` and commit the CSV only if the licence allows (CC0: it does) |

## Privacy
Uploaded files are processed in the server session only and are never written to the repository. Model bundles are generated in memory for download.
