MIT xPro Data Science Case Study 1.4 - Article Extraction and Spectral Clustering Analysis
==================================================================================================

Intro
-----

Article Extraction and Spectral Clustering Analysis Case Study based on 
Python version of Goose written by Xavier Grangier hosted at https://github.com/grangier/python-goose
and MITIE (MIT-NLP) hosted at https://github.com/mit-nlp/MITIE. 
See those pages for the full documentation and source code.

Licensing
---------

See the LICENSE file for more details.

Setup
-----

::

Please download https://github.com/mit-nlp/MITIE/releases/download/v0.4/mitie-v0.2-python-2.7-windows-or-linux64.zip
and add MITIE-models folder to the project directory!

    pip install -r requirements.txt
    python setup.py install

Take goose for a spin
---------------------

::

    >>> from goose import Goose
    >>> url = 'http://edition.cnn.com/2012/02/22/world/europe/uk-occupy-london/index.html?hpt=ieu_c2'
    >>> g = Goose()
    >>> article = g.extract(url=url)
    >>> article.title
    u'Occupy London loses eviction fight'
    >>> article.meta_description
    "Occupy London protesters who have been camped outside the landmark St. Paul's Cathedral for the past four months lost their court bid to avoid eviction Wednesday in a decision made by London's Court of Appeal."
    >>> article.cleaned_text[:150]
    (CNN) -- Occupy London protesters who have been camped outside the landmark St. Paul's Cathedral for the past four months lost their court bid to avoi
    >>> article.top_image.src
    http://i2.cdn.turner.com/cnn/dam/assets/111017024308-occupy-london-st-paul-s-cathedral-story-top.jpg

Run
---

::

    python extract_n_analyze.py

