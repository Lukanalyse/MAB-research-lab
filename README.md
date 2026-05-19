# MAB Research Lab

The purpose of this project is to provide a clear, dynamic, and interactive introduction to
Multi-Armed Bandits, starting from the fundamentals and progressively moving toward more
advanced algorithms and applications. The project is based on research papers listed in
the `papers/` folder.

## Project layout
1. `app.py`: main Streamlit entry point.
2. `src/`: application code (algorithms, pages, plots).
3. `papers/`: PDF library displayed on the Home page.
4. `outputs/pdf/`: optional research-note PDFs; commit them if you want them visible on Streamlit Cloud.

## Run locally
1. `python -m venv .venv`
2. `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

## Deploy on Streamlit Cloud
1. Push this repository to GitHub.
2. Create a new app on Streamlit Cloud and select the GitHub repo.
3. Set the main file to `streamlit_app.py` (auto-detected) or `app.py`.
4. Make sure any PDFs you want available in the app are committed under `papers/` and `outputs/pdf/`.
