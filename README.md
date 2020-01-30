# A collection of resources on machine learning, data processing and related areas

* [Analysis of different types of data](#analysis-of-different-types-of-data)
* [Data and knowledge engineering](#data-and-knowledge-engineering)
* [Libraries, utilies tools](#libraries-utilies-tools)
* [Other resources](#other-resources)


## Analysis of different types of data

### Time series and forecasting

Books and articles:
* ARIMA Model – Complete Guide to Time Series Forecasting in Python: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python

Implementations:
* (10k) https://github.com/facebook/prophet Prophet https://facebook.github.io/prophet/ 
* Forecast (R)

Stock price forecasting:
* (1.4k) https://github.com/huseinzol05/Stock-Prediction-Models
* (1.9k) https://github.com/borisbanushev/stockpredictionai

### Spatial data

* (1.9k) https://github.com/geopandas/geopandas Python tools for geographic data

### Audio and sound

General audio processing libraries:
* (4.8k) https://github.com/facebookresearch/wav2letter Automatic Speech Recognition Toolkit
* (3.3k) https://github.com/librosa/librosa Librosa
* (1.4k) https://github.com/MTG/essentia Essentia http://essentia.upf.edu/
* (2.8k) https://github.com/tyiannak/pyAudioAnalysis Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications
* (500) https://github.com/keunwoochoi/kapre keras audio preprocessor that let you calculate STFT/melspectrograms directly in your keras model on the fly without saving them in your storage
* (100) https://github.com/bmcfee/resampy Efficient sample rate conversion in python

Pitch trackers:
* (300) https://github.com/marl/crepe REPE is a monophonic pitch tracker based on a deep convolutional neural network operating directly on the time-domain waveform input. CREPE is state-of-the-art (as of 2018), outperforming popular pitch trackers such as pYIN and SWIPE https://arxiv.org/abs/1802.06182
* pYIN: https://www.eecs.qmul.ac.uk/~simond/pub/2014/MauchDixon-PYIN-ICASSP2014.pdf
* SWIPE: https://pdfs.semanticscholar.org/0fd2/6e267cfa9b6d519967ea00db4ffeac272777.pdf

Other
* (385) https://github.com/AddictedCS/soundfingerprinting audio acoustic fingerprinting fingerprinting in C# (advertising etc. - not speech)

Books:
* Think DSP: Digital Signal Processing in Python: http://greenteapress.com/thinkdsp/thinkdsp.pdf
  * Include a Python lib and examples in Python: https://github.com/AllenDowney/ThinkDSP 
* Digital Signal Processing: https://dsp-nbsphinx.readthedocs.io/en/nbsphinx-experiment/index.html
* Sound Processing with Short Time Fourier Transform: https://www.numerical-tours.com/matlab/audio_1_processing/

### Text NLP

* (20k) https://github.com/facebookresearch/fastText - Library for fast text representation and classification
* (15k) https://github.com/explosion/spaCy (in Cython) -  Industrial-strength Natural Language Processing (NLP) with Python and Cython https://spacy.io
* (10k) https://github.com/RaRe-Technologies/gensim Topic Modelling for Humans, robust semantic analysis, topic modeling and vector-space modeling
* (21k) https://github.com/huggingface/pytorch-transformers A library of state-of-the-art pretrained models for Natural Language Processing (NLP)
* (8k) https://github.com/nltk/nltk - NLTK
* (8k) https://github.com/zalandoresearch/flair A very simple framework for state-of-the-art Natural Language Processing (NLP)
* (4k) https://github.com/wireservice/csvkit
* (3.9k) DeepPavlov: https://github.com/deepmipt/DeepPavlov - building end-to-end dialog systems and training chatbots

Lists:
* https://github.com/keon/awesome-nlp
* https://github.com/astorfi/Deep-Learning-NLP
* https://github.com/brianspiering/awesome-dl4nlp

### Video

* (6k) https://github.com/Zulko/moviepy - Video editing with Python
* (12k) https://github.com/iperov/DeepFaceLab DeepFaceLab is a tool that utilizes machine learning to replace faces in videos.

### Images

* (3.5k) scikit-image: for image io and transforms
* (7k) https://github.com/python-pillow/Pillow PIL is the Python Imaging Library
* OpenCV: for images
* YOLO: https://pjreddie.com/darknet/yolo/
  * How to train YOLO to detect your own objects: https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/
* (1k) sod: https://github.com/symisc/sod/ An Embedded, Modern Computer Vision & Machine Learning Library
* DeepDetect https://github.com/beniz/deepdetect/tree/master/demo/objsearch http://www.deepdetect.com/
  * Object similarity search: https://github.com/beniz/deepdetect/tree/master/demo/objsearch
    * for suggesting labels (bounding box)

### Graphs, RDFs etc.

Graph stores:
* (1.5k) https://github.com/eBay/beam A distributed knowledge graph store 
* (1.6k) https://github.com/gchq/Gaffer A large-scale entity and relation database supporting aggregation of properties 

Databases:
* (10k) Dgraph Fast, Distributed Graph DB https://dgraph.io https://dgraph.io/
* Neo4j
* (2.5k) Janus Graph https://janusgraph.org/ https://github.com/janusgraph/janusgraph
  JanusGraph [2] has support for a lot of different backends, built by the old team behind TitanDB
* (8.2k) https://github.com/arangodb/arangodb ArangoDB is a native multi-model database with flexible data models for documents, graphs, and key-values. Best open source graph database is ArrangoDB they have master to master cluster

Visualizations and dashboards:
* graphviz


## Data and knowledge engineering

### Feature engineering

Lists:
* https://github.com/MaxHalford/xam - Personal data science and machine learning toolbox
* https://github.com/xiaoganghan/awesome-feature-engineering

Time series:
* (4.1k) tsfresh: https://github.com/blue-yonder/tsfresh 
* https://github.com/chlubba/catch22 catch-22: CAnonical Time-series CHaracteristics
* (235) https://github.com/benfulcher/hctsa Highly comparative time-series analysis code repository
* (600) https://github.com/bukosabino/ta Technical Analysis Library using Pandas (Python)

Feature extraction:
* (206) https://github.com/tyarkoni/pliers - Automated feature extraction in Python (audio/video)
* (4.1k) https://github.com/Featuretools/featuretools
* (1.4k) https://github.com/mapbox/robosat - feature extraction from aerial and satellite imagery. Semantic segmentation on aerial and satellite imagery. Extracts features such as: buildings, parking lots, roads, water, clouds

Feature selection:
* (1.2k) https://github.com/WillKoehrsen/feature-selector Feature selector is a tool for dimensionality reduction of machine learning datasets.
  Find unnecessary (redundant) features using simple methods: missing values, high correlation etc.
  See article: https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0
* (600) https://github.com/scikit-learn-contrib/boruta_py - Python implementations of the Boruta all-relevant feature selection method
* (200) https://github.com/EpistasisLab/scikit-rebate - A scikit-learn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms for Machine Learning
* (560) https://github.com/abhayspawar/featexp eature exploration for supervised learning

Hyper-parameter optimization:
* (362) https://github.com/machinalis/featureforge - creating and testing machine learning features, with a scikit-learn compatible API
* (1.5k) https://github.com/instacart/lore Lore makes machine learning approachable for Software Engineers and maintainable for Machine Learning Researchers
* (6.1k) https://github.com/EpistasisLab/tpot Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming
* (1.3k) https://github.com/ClimbsRocks/auto_ml [UNMAINTAINED] Automated machine learning for analytics & production
* (1.1k) https://github.com/pfnet/optuna - A hyperparameter optimization framework https://optuna.org

### AutoML

AutoML libs:
* AutoWeka
* MLBox
* auto-sklearn
* TPOT
* HpBandSter
* AutoKeras

Systems and companies:
* DataRobot
* DarwinAI
* H2O.ai Dreverless AI
* OneClick.ai

### AI, data mining, machine learning algorithms

Resources:
* https://github.com/astorfi/TensorFlow-World-Resources
* https://github.com/astorfi/Deep-Learning-World

Algorithms:
* XGBoost, CatBoost, LightGBM
* https://github.com/spotify/annoy 
  * Approximate Nearest Neighbors 
  * USP: ability to use static files as indexes, share index across process, that is, in-memory and efficient and multi-process
  * e.g, music recommendation in Spotify, simlar images (for labeling etc.)  

### Data integration, ETL, data integration

* (13k) https://github.com/apache/incubator-airflow https://airflow.apache.org programmatically author, schedule, and monitor workflows as directed acyclic graphs (DAGs) of tasks
* (12k) https://github.com/spotify/luigi build complex pipelines of (long-running) batch jobs like Hadoop jobs, Spark jobs, dumping data to/from databases, running machine learning algorithms, Python snippet etc. The dependency graph is specified within Python (not XML or JSON).
* (2.7k) https://github.com/azkaban/azkaban Azkaban workflow manager
* Oozie 
* (750) https://github.com/d6t/d6tflow - Python library for building highly effective data science workflows (on top of luigi)

ETL:
* (1k) https://github.com/python-bonobo/bonobo https://www.bonobo-project.org/ Transform Load Extract for Python 3.5+ 
* (1.2k) https://github.com/mara/data-integration A lightweight opinionated ETL framework, halfway between plain scripts and Apache Airflow

Stream processing:
* (1.5k) https://github.com/nerevu/riko A Python stream processing engine modeled after Yahoo! Pipes

Web scrapping
* (34k) https://github.com/scrapy/scrapy - create spiders bots that scan website pages and collect structured data

### Labeling (with suggestions)

* https://github.com/Cartucho/yolo-boundingbox-labeler-GUI, YOLO format
* Labelbox: https://github.com/Labelbox/Labelbox https://www.labelbox.io/ non-YOLO format
* https://github.com/beniz/deepdetect/tree/master/demo/objsearch
* labelmg (free)
* rectlabel (for Pascal VOC format)

### Visualization and VA

* Matplotlib http://matplotlib.org/
* Bokeh http://bokeh.pydata.org/en/latest/ The Bokeh library creates interactive and scalable visualizations in a browser using JavaScript widgets.
* Seaborn http://stanford.edu/~mwaskom/software/seaborn/# https://seaborn.pydata.org/ - a higher-level API based on the matplotlib library. It contains more suitable default settings for processing charts. Also, there is a rich gallery of visualizations. 
* Lightning http://lightning-viz.org/
* Plotly https://plot.ly/
* Pandas built-in plotting http://pandas.pydata.org/pandas-docs/stable/visualization.html
* HoloViews http://holoviews.org/
* VisPy http://vispy.org/
* pygg http://www.github.com/sirrice/pygg
* Altair https://github.com/altair-viz/altair
* hypertools: https://github.com/ContextLab/hypertools

Dashboards:
* Grafana (time series) https://grafana.com/  https://github.com/grafana/grafana 
  * Integration with InfluxDB
* Superset (apache) https://github.com/apache/incubator-superset
  * Integration with druid.io database https://github.com/apache/incubator-druid/
* Metabase https://metabase.com/ https://github.com/metabase/metabase
* Redash https://redash.io/  https://github.com/getredash/redash
* Zeppelin Apache https://zeppelin.apache.org/ https://github.com/apache/zeppelin 
* Mixpanel https://mixpanel.com/
* Blazer https://github.com/ankane/blazer
* Bdash https://github.com/bdash-app/bdash
* Datamill, ?
* wagonhq, ?

Publishing notebooks (from github etc.):
* https://mybinder.org/
* http://colab.research.google.com/ a kind of Jupyter notebooks stored in Google Drive


## Libraries, utilies tools

### Python

* Top 10 Python libraries of 2017: https://tryolabs.com/blog/2017/12/19/top-10-python-libraries-of-2017/
* https://github.com/vinta/awesome-python - A curated list of awesome Python frameworks, libraries, software and resources
* Checklist for Python libraries APIs: http://python.apichecklist.com/

### Tools

* Data structures:
  * https://github.com/mahmoud/boltons boltons should be builtins
  * https://github.com/pytoolz/toolz List processing tools and functional utilities (replaces itertools and functools)
  * Zict: Composable Mutable Mappings
  * HeapDict: a heap with decrease-key and increase-key operations
  * sortedcontainers: Python Sorted Container Types: SortedList, SortedDict, and SortedSet

* Networking:
  * certifi: A carefully curated collection of Root Certificates for validating the trustworthiness of SSL certificates while verifying the identity of TLS hosts
  * urllib3: HTTP library with thread-safe connection pooling, file post support, sanity friendly, and more

Other:
* click: creating beautiful command line interfaces
* chardet

### Formats, persistence and serialization

* https://github.com/wesm/feather  fast, interoperable binary data frame storage for Python, R, and more powered by Apache Arrow
* https://github.com/apache/arrow standardized language-independent columnar memory format for flat and hierarchical data
* https://github.com/Blosc/bcolz A columnar data container that can be compressed
* cloudpickle: serialize Python constructs not supported by the default pickle module from the Python standard library (lambdas etc.)
* pytables (905): https://github.com/PyTables/PyTables - A Python package to manage extremely large amounts of data http://www.pytables.org
  * based on hdf5 https://www.hdfgroup.org/ HDF supports n-dimensional datasets and each element in the dataset may itself be a complex object.
* h5py (1.3k): https://github.com/h5py/h5py HDF5 for Python -- The h5py package is a Pythonic interface to the HDF5 binary data format

### Authentication, Authorization, Security

* https://github.com/dwyl/learn-json-web-tokens/ -  Learn how to use JSON Web Token (JWT) to secure your next Web App! (Tutorial/Example with Tests!!)

### Linux and OS

* https://0xax.gitbooks.io/linux-insides/content/ - Linux inside
* https://john-millikin.com/unix-syscalls - UNIX Syscalls

### Platform and servers

Load balancing:
* ngnix
* haproxy - http://www.haproxy.org/
* GLB Director - https://github.com/github/glb-director - Github, Layer 4 load balancer
* https://code.fb.com/open-source/open-sourcing-katran-a-scalable-network-load-balancer/ - Facebook Katran
* https://cloudplatform.googleblog.com/2016/03/Google-shares-software-network-load-balancer-design-powering-GCP-networking.html - Google

Dockerized automated https reverse proxy:
* https://github.com/jwilder/nginx-proxy Automated nginx proxy for Docker containers using docker-gen
  * https://github.com/JrCs/docker-letsencrypt-nginx-proxy-companion LetsEncrypt companion container for nginx-proxy 
* https://traefik.io/ https://github.com/containous/traefik/ - cloud native edge router. a modern HTTP reverse proxy and load balancer that makes deploying microservices easy
* https://github.com/SteveLTN/https-portal A fully automated HTTPS server powered by Nginx, Let's Encrypt and Docker. 
* https://github.com/sozu-proxy/sozu HTTP reverse proxy, configurable at runtime, fast and safe, built in Rust.
* https://caddyserver.com/docs/automatic-https https://github.com/mholt/caddy (free if you build it yourself) Caddy automatically enables HTTPS for all your sites, given that some reasonable criteria. Fast, cross-platform HTTP/2 web server with automatic HTTPS 
* https://github.com/Valian/docker-nginx-auto-ssl Docker image for automatic generation of SSL certs using Let's encrypt and Open Resty 

Discussions:
* https://news.ycombinator.com/item?id=17984970

Service registry and orchestrator:
* etcd
* consul

Logging and tracing
* fast logging library for Go Zap: https://github.com/uber-go/zap
* http://opentracing.io/ - OpenTracing standard
* CNCF Jaeger, a Distributed Tracing System  https://github.com/jaegertracing/jaeger https://uber.github.io/jaeger/ https://jaegertracing.io/
* Lightstep https://github.com/lightstep/lightstep-tracer-go
* Application tracing system for Go, based on Google's Dapper. (OpenTracing) https://sourcegraph.com https://github.com/sourcegraph/appdash

### Async, event buses, actor model, job queues etc.

* (13k) https://github.com/celery/celery Distributed Task Queue http://celeryproject.org/
* (4.7k) http://www.gevent.org/ https://github.com/gevent/gevent Coroutine-based concurrency library
* (800) http://eventlet.net/ https://github.com/eventlet/eventlet/ Concurrent networking library

Web frameworks (concurrent, async):
* (18k) http://www.tornadoweb.org https://github.com/tornadoweb/tornado web framework and asynchronous networking library
* (3k) https://github.com/Pylons/pyramid web framework https://trypyramid.com/

Actor models:
* (1.8k) https://github.com/quantmind/pulsar/ Event driven concurrent framework for Python https://quantmind.github.io/pulsar/index.html 
Pulsar implements two layers of components on top of python asyncio module: the actor layer and the application framework.
* (800) https://github.com/jodal/pykka Pykka is a Python implementation of the actor model, which makes it easier to build concurrent applications
* (80) https://github.com/kquick/Thespian Python Actor concurrency library
* https://gitlab.com/python-actorio/actorio

### Computing

* https://github.com/dask/dask (6.2k): Parallel computing with task scheduling
* https://github.com/dask/distributed (939): distributed dask
* https://github.com/ray-project/ray A system for parallel and distributed Python that unifies the ML ecosystem (similar to Dask)
* https://github.com/databricks/spark-sklearn Scikit-learn integration package for Spark
* https://github.com/arrayfire/arrayfire (2.9k): a general purpose GPU library

Books:
* https://chryswoods.com/parallel_python/index.html - Parallel Programming with Python


## Other resources

### Data sources

* Binance:
  * https://github.com/binance-exchange/binance-official-api-docs
  * https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md
  * https://github.com/binance-exchange/python-binance

### Books

* Free data science books:
  * https://web.stanford.edu/~hastie/ElemStatLearn/ - The Elements of Statistical Learning: Data Mining, Inference, and Prediction
  * http://www.statlearning.com/ - An Introduction to Statistical Learning
  * https://d2l.ai/ - Dive into Deep Learning
  * https://www.inferentialthinking.com/ - Computational and Inferential Thinking. The Foundations of Data Science
  * http://www.cs.cornell.edu/jeh/book.pdf - Foundations of Data Science
  * https://otexts.org/fpp2/ - Forecasting: Principles and Practice
  * https://github.com/christophM/interpretable-ml-book - Interpretable machine learning
