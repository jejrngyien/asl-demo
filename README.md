# 🤟 ASL-Fingeralphabet-Erkennung — Live-Demo

Interaktive **Streamlit**-Demo eines **C3D**-Modells, das das **Fingeralphabet der
amerikanischen Gebärdensprache** (A–Z + del/space/nothing) aus einem einzelnen Bild
erkennt. Zeig deiner Webcam eine Handform oder lade ein Foto hoch — die App schneidet
mit **MediaPipe** deine Hand aus und sagt den Buchstaben voraus.

Bereitstellbares Frontend zum Projekt
**[ASL Recognition with 3D CNNs](https://github.com/jejrngyien/asl)**
(Trainings-Pipeline, R(2+1)D-Vergleich und WLASL-Experimente auf Wortebene liegen dort).

## Wie das Modell geladen wird

Die Gewichte (~330 MB) sind **nicht** eingecheckt. Sie werden beim ersten Start einmalig
von einem **GitHub-Release** heruntergeladen. Die URL wird über die Konstante `MODEL_URL`
in [`streamlit_app.py`](streamlit_app.py) oder als Streamlit-Secret gesetzt, z. B.:

```
https://github.com/USERNAME/asl-demo/releases/download/v1.0/model.pt
```

Eine lokale `model.pt` neben der App wird verwendet, falls vorhanden (praktisch für lokale Läufe).

## Lokal ausführen

```bash
pip install -r requirements.txt
# den Checkpoint als model.pt neben die App legen (oder MODEL_URL setzen)
streamlit run streamlit_app.py
```

## Auf Streamlit Community Cloud deployen (kostenlos)

1. Diesen Ordner in ein **GitHub**-Repo pushen.
2. Ein **Release** auf dem Repo erstellen und `model.pt` als Asset anhängen.
3. Die Release-Download-URL in `MODEL_URL` eintragen (in `streamlit_app.py` oder als Secret).
4. Auf **share.streamlit.io** → **New app** → Repo und `streamlit_app.py` wählen → **Deploy**.

## Hinweise

- **Modell:** statisches C3D, ~84 % Top-1 auf dem Kaggle-ASL-Alphabet-Testset.
- **Eingabe:** Das Modell sieht einen 112×112-Handausschnitt (in der App zur Transparenz angezeigt).
- **Tipp:** Die Buchstaben **Y, C, L, B, V** sind am einfachsten; nutze eine klare, gut beleuchtete Handform.
