# multimodal-project

Quick start (valós adat, kontrasztív tanítás)
- Tedd a Flickr30k adatot ide: `data/flickr30k/images/` és `data/flickr30k/annotations/annotations.csv`.
- Telepítsd a csomagokat: `pip install -r requirements.txt`.
- Sanity check (betöltés, tokenizálás): `python main.py`.
- Multimodális CLIP-szerű tanítás (kép-caption párokra): `python train.py`. Ment: `results/models/flickr30k_clip_best.pth` (model_state + vocab).
- Kép-encoder mentése CLIP tanítás után: `python train_image_only.py` (ment: `results/models/image_encoder_from_clip.pth`).

Mit csinál a kód?
- Képeket EfficientNet backbone-nal kódolja, szöveget egyszerű vocab tokenizáló + masked mean pool encoderrel.
- Kontrasztív (InfoNCE) loss-szal tanulja, hogy a batch-ben a helyes kép-caption párok legyenek egymás legközelebbi embeddingjei.
- Értékelés: batch-szintű top-1/top-5 retrieval pontosság a validációs spliten.
