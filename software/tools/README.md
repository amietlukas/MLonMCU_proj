# Tools

Helper scripts and notebooks used to prepare datasets, kept out of the model
`src/` folders to keep those clean.

## Contents

- `RGBtoGRAY.py` — convert an image dataset from RGB to grayscale (uint8 0..255),
  preserving the folder structure. Used to build the grayscale input for the
  U5 gesture classifier.
- `dataset_stuff.ipynb` — dataset inspection / preparation notebook.
- `datset_keypoint.ipynb` — keypoint / annotation exploration notebook.

## Note

These were moved here from the model `src/` folders, so paths inside them may
still point at the old locations — adjust the input/output paths before running.
