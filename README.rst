eeg-brain-age
=============

This repository contains resources that were used for our study entitled

"**Brain Age Revisited: Investigating the State vs. Trait Hypotheses of EEG-derived Brain-Age Dynamics with Deep Learning"**

which is currenty in submission.

Requirements
============
Requirements are specified in *environment.yml*. *versions.txt* contains additional versions of packages installed via pip (e.g. PyTorch) used to run the code.

Data
====
Our study is based on the Temple University Hospital EEG Corpus (v1.2.0), the Temple University Hospital Abnormal EEG Corpus (v2.0.0), and several novel datasets derived thereof.
The corpora are available for download at: https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml. Detail on how to create the derivatives can be found in the code database.

Citing
======

If you use this code or if you want to refer to our scientific publication, please cite us as:

.. code-block:: bibtex

  @article{gemein2023age,
    title={Brain Age Revisited: Investigating the State vs. Trait Hypotheses of EEG-derived Brain-Age Dynamics with Deep Learning},
    author={Gemein, Lukas AW and Schirrmeister, Robin T and Boedecker, Joschka and Ball, Tonio},
    journal={arXiv preprint arXiv:},
    year={2023},
  }
