# TIPSTER

This is the official repository for **Disentanglement of Audio Representations: A Case Study on Timbre Transfer**. Audio demos are available on the [TIPSTER project page](https://fernandotnco.github.io/TIPSTER/), where you can compare the source performance, timbre reference, TIPSTER output, and AFTER baseline.

## How TIPSTER works

<img width="1139" height="354" alt="image" src="https://github.com/user-attachments/assets/954d57b0-1aea-4893-b708-d8fa78a47019" />


TIPSTER transfers the instrument sound of a reference recording to a source performance while aiming to preserve the source’s notes and timing. A shared audio autoencoder works on complex spectrograms. ConDe splits each latent representation into three additive components.

- **Structure** captures what is played, including pitch and timing.
- **Timbre** captures the sound of the instrument.
- **Residual** retains variation outside those two factors.

At transfer time, TIPSTER combines the source structure and residual with the reference timbre, then decodes the sum in one pass. During training, contrastive examples share either the same passage or the same instrument. These pairs come from ASAP MIDI performances rendered as piano, harp, vibraphone, and xylophone.

<img width="1160" height="457" alt="image" src="https://github.com/user-attachments/assets/3f391993-440c-4bc2-a038-130cfc1e1023" />


## Evaluation

Timbre transfer was evaluated on a composer-disjoint ASAP test set. Target instrument accuracy measures whether an instrument classifier recognizes the intended timbre. Chroma DTW and onset correlation measure preservation of pitch-class content and timing, respectively.

| Model | Target instrument accuracy ↑ | Mean transfer time ↓ | Chroma DTW ↓ | Onset correlation ↑ |
| --- | ---: | ---: | ---: | ---: |
| **TIPSTER** | **97.5%** | **10 ms** | **0.077** | **0.787** |
| AFTER | 96.5% | 701 ms | 0.290 | 0.643 |
| SS-VQ-VAE | 24.6% | 20 ms | 0.292 | 0.063 |

The paper also evaluates the latent components with Synesis. Probes predict chords and notes well from the structure component, which still contains some timbre information.

## Code and training

| Path | Purpose |
| --- | --- |
| [`src/timbre_train.py`](src/timbre_train.py) | Loads the configuration and starts training |
| [`src/config/timbre_asap_paired.py`](src/config/timbre_asap_paired.py) | Example configuration for training with paired ASAP audio |
| [`src/data/asap_triplet.py`](src/data/asap_triplet.py) | Loads rendered ASAP audio and samples training triplets |
| [`src/models/music_autoencoder.py`](src/models/music_autoencoder.py) and [`src/models/conde.py`](src/models/conde.py) | Implement the audio autoencoder and latent decomposition |
| [`src/conde_trainer.py`](src/conde_trainer.py) | Applies the training objectives and runs training steps |

The supplied configuration resumes training from a checkpoint. Point the dataset loader to your rendered ASAP HDF5 and split JSON, then set the checkpoint and output paths in the configuration. Run the training script from `src/` with the paired ASAP configuration.

```sh
cd src
python timbre_train.py --config_module config.timbre_asap_paired
```

