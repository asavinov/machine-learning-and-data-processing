# A collection of resources on machine learning, data processing and related areas

* [Analysis of different types of data](#analysis-of-different-types-of-data)
* [Data and knowledge engineering](#data-and-knowledge-engineering)
* [Asynchronous data processing](#asynchronous-data-processing)
* [Libraries, utilities tools](#libraries-utilities-tools)
* [Other resources](#other-resources)


## Analysis of different types of data

### Time series and forecasting

Books and articles:
* ARIMA Model – Complete Guide to Time Series Forecasting in Python: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python

Implementations:
* (10.6k) https://github.com/facebook/prophet Prophet https://facebook.github.io/prophet/ 
* Forecast (R)

Stock price forecasting:
* (1.8k) https://github.com/huseinzol05/Stock-Prediction-Models
* (2.1k) https://github.com/borisbanushev/stockpredictionai

### Spatial data

* (2.0k) https://github.com/geopandas/geopandas Python tools for geographic data

### Audio and sound

General audio processing libraries:
* (5.0k) https://github.com/facebookresearch/wav2letter Automatic Speech Recognition Toolkit
* (3.5k) https://github.com/librosa/librosa Librosa
* (1.4k) https://github.com/MTG/essentia Essentia http://essentia.upf.edu/
* (3.0k) https://github.com/tyiannak/pyAudioAnalysis Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications
* (537) https://github.com/keunwoochoi/kapre keras audio preprocessor that let you calculate STFT/melspectrograms directly in your keras model on the fly without saving them in your storage
* (118) https://github.com/bmcfee/resampy Efficient sample rate conversion in python

Pitch trackers:
* (329) https://github.com/marl/crepe REPE is a monophonic pitch tracker based on a deep convolutional neural network operating directly on the time-domain waveform input. CREPE is state-of-the-art (as of 2018), outperforming popular pitch trackers such as pYIN and SWIPE https://arxiv.org/abs/1802.06182
* pYIN: https://www.eecs.qmul.ac.uk/~simond/pub/2014/MauchDixon-PYIN-ICASSP2014.pdf
* SWIPE: https://pdfs.semanticscholar.org/0fd2/6e267cfa9b6d519967ea00db4ffeac272777.pdf

Other
* (397) https://github.com/AddictedCS/soundfingerprinting audio acoustic fingerprinting fingerprinting in C# (advertising etc. - not speech)

Books:
* Think DSP: Digital Signal Processing in Python: http://greenteapress.com/thinkdsp/thinkdsp.pdf
  * Include a Python lib and examples in Python: https://github.com/AllenDowney/ThinkDSP 
* Digital Signal Processing: https://dsp-nbsphinx.readthedocs.io/en/nbsphinx-experiment/index.html
* Sound Processing with Short Time Fourier Transform: https://www.numerical-tours.com/matlab/audio_1_processing/

### Text NLP

* (21k) https://github.com/facebookresearch/fastText - Library for fast text representation and classification
* (16.2k) https://github.com/explosion/spaCy (in Cython) -  Industrial-strength Natural Language Processing (NLP) with Python and Cython https://spacy.io
* (10.7k) https://github.com/RaRe-Technologies/gensim Topic Modelling for Humans, robust semantic analysis, topic modeling and vector-space modeling
* (25.3k) https://github.com/huggingface/pytorch-transformers A library of state-of-the-art pretrained models for Natural Language Processing (NLP)
* (8.8k) https://github.com/nltk/nltk - NLTK
* (8.4k) https://github.com/zalandoresearch/flair A very simple framework for state-of-the-art Natural Language Processing (NLP)
* (4.2k) https://github.com/wireservice/csvkit
* (4.2k) https://github.com/deepmipt/DeepPavlov - building end-to-end dialog systems and training chatbots

Lists:
* https://github.com/keon/awesome-nlp
* https://github.com/astorfi/Deep-Learning-NLP
* https://github.com/brianspiering/awesome-dl4nlp

### Video

* (6.4k) https://github.com/Zulko/moviepy - Video editing with Python
* (13.5k) https://github.com/iperov/DeepFaceLab DeepFaceLab is a tool that utilizes machine learning to replace faces in videos.

### Images

* (3.7k) https://github.com/scikit-image/scikit-image Image processing in Python
* (7.2k) https://github.com/python-pillow/Pillow PIL is the Python Imaging Library
* OpenCV: for images
* YOLO: https://pjreddie.com/darknet/yolo/
  * How to train YOLO to detect your own objects: https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/
* (1.2k) sod: https://github.com/symisc/sod/ An Embedded, Modern Computer Vision & Machine Learning Library
* DeepDetect https://github.com/beniz/deepdetect/tree/master/demo/objsearch http://www.deepdetect.com/
  * Object similarity search: https://github.com/beniz/deepdetect/tree/master/demo/objsearch
    * for suggesting labels (bounding box)

### Graphs, RDFs etc.

Graph stores:
* (1.5k) https://github.com/eBay/beam A distributed knowledge graph store 
* (1.6k) https://github.com/gchq/Gaffer A large-scale entity and relation database supporting aggregation of properties 

Databases:
* (10k) Dgraph Fast, Distributed Graph DB https://dgraph.io/
* Neo4j
* (3.2k) Janus Graph https://janusgraph.org/ https://github.com/janusgraph/janusgraph
  JanusGraph [2] has support for a lot of different backends, built by the old team behind TitanDB
* (9.5k) https://github.com/arangodb/arangodb ArangoDB is a native multi-model database with flexible data models for documents, graphs, and key-values. Best open source graph database is ArrangoDB they have master to master cluster

Visualizations and dashboards:
* graphviz


## Data and knowledge engineering

### Feature engineering

Lists:
* https://github.com/MaxHalford/xam - Personal data science and machine learning toolbox
* https://github.com/xiaoganghan/awesome-feature-engineering

Time series:
* (4.7k) tsfresh: https://github.com/blue-yonder/tsfresh
* (1k) https://github.com/bukosabino/ta Technical Analysis Library using Pandas (Python)
* (298) https://github.com/benfulcher/hctsa Highly comparative time-series analysis code repository
* https://github.com/chlubba/catch22 catch-22: CAnonical Time-series CHaracteristics

Feature extraction:
* (219) https://github.com/tyarkoni/pliers - Automated feature extraction in Python (audio/video)
* (4.8k) https://github.com/Featuretools/featuretools
* (1.6k) https://github.com/mapbox/robosat - feature extraction from aerial and satellite imagery. Semantic segmentation on aerial and satellite imagery. Extracts features such as: buildings, parking lots, roads, water, clouds

Feature selection:
* (1.6k) https://github.com/WillKoehrsen/feature-selector Feature selector is a tool for dimensionality reduction of machine learning datasets.
  Find unnecessary (redundant) features using simple methods: missing values, high correlation etc.
  See article: https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0
* (686) https://github.com/scikit-learn-contrib/boruta_py - Python implementations of the Boruta all-relevant feature selection method
* (251) https://github.com/EpistasisLab/scikit-rebate - A scikit-learn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms for Machine Learning
* (641) https://github.com/abhayspawar/featexp eature exploration for supervised learning

Hyper-parameter optimization:
* (7k) https://github.com/EpistasisLab/tpot Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming
* (2.2k) https://github.com/pfnet/optuna - A hyperparameter optimization framework https://optuna.org
* (1.5k) https://github.com/instacart/lore Lore makes machine learning approachable for Software Engineers and maintainable for Machine Learning Researchers
* (1.4k) https://github.com/ClimbsRocks/auto_ml [UNMAINTAINED] Automated machine learning for analytics & production
* (372) https://github.com/machinalis/featureforge - creating and testing machine learning features, with a scikit-learn compatible API

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

### Data integration, ETL, job management, web scrapping

* (14.8k) https://github.com/celery/celery Distributed Task Queue http://celeryproject.org/
* (16.4k) https://github.com/apache/incubator-airflow https://airflow.apache.org programmatically author, schedule, and monitor workflows as directed acyclic graphs (DAGs) of tasks
* (13.2k) https://github.com/spotify/luigi build complex pipelines of (long-running) batch jobs like Hadoop jobs, Spark jobs, dumping data to/from databases, running machine learning algorithms, Python snippet etc. The dependency graph is specified within Python (not XML or JSON).
* (3.1k) https://github.com/azkaban/azkaban Azkaban workflow manager
* Oozie 
* (842) https://github.com/d6t/d6tflow - Python library for building highly effective data science workflows (on top of luigi)

ETL:
* (1.4k) https://github.com/mara/data-integration A lightweight opinionated ETL framework, halfway between plain scripts and Apache Airflow
* (1.2k) https://github.com/python-bonobo/bonobo https://www.bonobo-project.org/ Transform Load Extract for Python 3.5+ 

Stream processing:
* (1.5k) https://github.com/nerevu/riko A Python stream processing engine modeled after Yahoo! Pipes

Web scrapping
* (36.8k) https://github.com/scrapy/scrapy - create spiders bots that scan website pages and collect structured data

Machine learning training, deployment etc.
* (8.8k) https://github.com/kubeflow/kubeflow Machine Learning Toolkit for Kubernetes
* (4.3k) https://github.com/cortexlabs/cortex Cloud native model serving infrastructure. Cortex is an open source platform for deploying machine learning models as production web services

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

## Asynchronous data processing

### What is asynchronous data processing

TBD

### Reactive programming

Approaches to asynchronous programming:

* Callback model:
  * A callback function is provided as part of an asynchronous call
  * The call is non-blocking and the source program continues execution
  * It is not possible to await for the return (it is essentially done by the callback function)
  * The callback function can be viewed as a one-time listener for a return event, that is, it represents code which consumes the result
  * The source code where the call is made and the consumer of the result are in different functions and cannot share (local) context
  * The callback function may make its own asynchronous calls which leads to a "callback hell"

* Future/promise:
  * An asynchronous call is made as usual but return a special wrapper object
  * A callback function is not specified and is not used
  * The returned result is consumed by the code which follows the call (as opposed to its use in a separate callback function)
  * The future/promise is supposed to be awaited. Awaiting denotes a point where we say that the next instruction needs the result
  * The awaiting point is like a one-time single-value listener for the result where the program execution is suspended until the return event is received

Links:
* Reactive programming in Python
* Reactive programming in Java
* Reactive programming in C#

### Reactive streaming

* Listeners:
  * A callback function is registered and then automatically called for each incoming event
  * Callback functions are (normally) called only sequentially, that is, next event can be processed only when the previous event has been processed by the previous callback invocation. Callbacks are not executed concurrently.
  * The result of a callback invocation is frequently needed because it is executed by the event producer

* Reactive streams:
  * It is a graph of producers and consumers
  * Consumers and producers are not supposed to communicate in a free manner by sending messages to each other where a sender knows the address(s) of the receivers
  * An element declares which messages it produces but it is unaware of who will subscribe to and consume its messages, and how they will be used (in contrast to the actor model)
  * An element must know what kind of messages it needs and explicitly subscribe to specific producers - messages will not come automatically just because somebody wants to send them to us
  * For data processing, reative streams provide a number of operators which can be applied to an input stream(s) and produce an output stream
  * Links:
    * https://www.reactive-streams.org/
    * (3.1k) https://github.com/reactor/reactor
    * http://reactivex.io/ An API for asynchronous programming with observable streams:
      * (42.4k) https://github.com/ReactiveX/RxJava Reactive Extensions for the JVM – a library for composing asynchronous and event-based programs using observable sequences for the Java VM
      * (3.4k) https://github.com/ReactiveX/RxPY Reactive Extensions for Python
      * etc. https://github.com/ReactiveX

* Actor model:
  * Each element has an identifier (address, reference) which is used by other actors to send messages
  * A sender (producer) must know what it wants to do when it sends messages and it must know its destination
  * Receiving elements can receive messages from any other element and their task is to respond according to their logic (as expected by the sender)
  * Actors receive messages without subscribing to any producer (as opposed to the actor model where you will not receive anything until you subscribe to some event producer)
  * Each actor has a handler (callback) which is invoked for processing incoming messages
  * Actors are supposed to have a state and frequently it is why we want to define different actors
  * Links:
    * (10.8k) https://github.com/akka/akka Build highly concurrent, distributed, and resilient message-driven applications on the JVM
    * (10.9k) https://github.com/eclipse-vertx/vert.x Vert.x is a tool-kit for building reactive applications on the JVM
    * (1.8k) https://github.com/quantmind/pulsar/ Event driven concurrent framework for Python https://quantmind.github.io/pulsar/index.html Pulsar implements two layers of components on top of python asyncio module: the actor layer and the application framework
    * (876) https://github.com/jodal/pykka Python implementation of the actor model, which makes it easier to build concurrent applications
    * (120) https://github.com/kquick/Thespian Python Actor concurrency library
    * https://gitlab.com/python-actorio/actorio

### Event loops vs. threads

* Both a thread task and an event loop task are executed until finished, that is, the code to execute is provided as a procedure
* Thread tasks are dispatched by the system (not application) while dispatching logic of event tasks is part of the application
* At each moment, there is a fixed number of threads concurrently executed by one process. The number of concurrently executed event loop tasks is not limited.
* Thread tasks are (automatically) switched at the instruction level and the dispatcher is unaware of the needs of this thread or the application. Event loop tasks are switched at the level of logical application units depending on what this application needs.
* In a multi-thread application, we need to manage the threads ourselves, e.g., by creating and deleting them. In an event loop application, the tasks (starting, suspending, finishing) is managed by the event loop manager.
* In an event loop application, tasks specify dependencies on other tasks, and these points are used while dispatching the execution of tasks. Threads cannot declare dependencies on the results provided by other tasks. If we need some external result, then the thread has to wait. This logic has to be implemented manually and the system dispatcher is unaware of these dependencies.

Event loops: 
* (14.9k) https://github.com/libuv/libuv Cross-platform asynchronous I/O
* (6.1k) https://github.com/libevent/libevent Event notification library
* (886) https://github.com/enki/libev Full-featured high-performance event loop loosely modelled after libevent
* Python asyncio: https://github.com/timofurrer/awesome-asyncio 

### Async networking libraries

* (5.1k) https://github.com/gevent/gevent coroutine - based Python networking library. "systems like gevent use lightweight threads to offer performance comparable to asynchronous systems, but they do not actually make things asynchronous"
  * greenlet to provide a high-level synchronous API 
    * on top of the libev or libuv event loop (like libevent)

* (909) https://github.com/eventlet/eventlet concurrent networking library for Python
  * epoll or kqueue or libevent for highly scalable non-blocking I/O

* (9.5k) https://github.com/aio-libs/aiohttp Asynchronous HTTP client/server framework for asyncio and Python

* (3.8k) https://github.com/twisted/twisted Event-driven networking engine written in Python. 
  * Twisted projects variously support TCP, UDP, SSL/TLS, IP multicast, Unix domain sockets, many protocols (including HTTP, XMPP, NNTP, IMAP, SSH, IRC, FTP, and others), and much more.
  * Twisted supports all major system event loops:
    * select (all platforms), 
    * poll (most POSIX platforms), 
    * epoll (Linux), 
    * kqueue (FreeBSD, macOS), 
    * IOCP (Windows), 
    * various GUI event loops (GTK+2/3, Qt, wxWidgets)

### Async web frameworks

* (19.0k) https://github.com/tornadoweb/tornado Python web framework and asynchronous networking library 
  * "Tornado is integrated with the standard library asyncio module and shares the same event loop (by default since Tornado 5.0). In general, libraries designed for use with asyncio can be mixed freely with Tornado." 
  * Some async client Libraries built on tornado.ioloop:
    * DynamoDB, CouchDB, Hbase, MongoDB, MySQL, PostgresQL, PrestoDB, RethinkDB
    * AMQP, NATS, RabbitMQ, SMTP
    * DNS, Memcached, Reis
    * etc. https://github.com/tornadoweb/tornado/wiki/Links
* (13.7k) https://github.com/huge-success/sanic Sanic
* (12.6k) https://github.com/tiangolo/fastapi FastAPI
* (5.6k) https://github.com/vibora-io/vibora Like Sanic but even faster
* https://gitlab.com/pgjones/quart API compatible with Flask 
* (3.4k) https://github.com/Pylons/pyramid Python web framework https://trypyramid.com/ (it seems to be a conventional web framework)

### Utilities

Retry libraries:
* (2.1k) https://github.com/jd/tenacity - originates from a fork of retrying
* (1.5k) https://github.com/rholder/retrying - not supported anymore
* (1.2k) https://github.com/litl/backoff
* (267) https://github.com/invl/retry
* (36) https://github.com/channable/opnieuw

## Libraries, utilities, tools

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

Identity and Access Management
  * (5.9k) https://github.com/keycloak/keycloak
  * (8.3k) https://github.com/ory/hydra OAuth2 Server and OpenID Certified™ OpenID Connect Provider written in Go - cloud native, security-first, open source API security for your infrastructure. SDKs for any language
  * (1.7k) https://github.com/lepture/authlib The ultimate Python library in building OAuth, OpenID Connect clients and servers. JWS,JWE,JWK,JWA,JWT included
  * (0.5k) https://github.com/OpenIDC/pyoidc/ A Python OpenID Connect implementation
  * (1.5k) https://github.com/RichardKnop/go-oauth2-server A standalone, specification-compliant, OAuth2 server written in Golang. 
  * https://www.gluu.org/

Secrets management, encryption as a service, and privileged access management. A secret is anything that you want to tightly control access to, such as API keys, passwords, certificates, and more.
  * (15.5k) https://github.com/hashicorp/vault A tool for secrets management, encryption as a service, and privileged access management

Policy Enforcement Point, Identity And Access Proxy (IAP), Zero-Trust Network Architecture, i.e. a reverse proxy in front of your upstream API or web server that rejects unauthorized requests and forwards authorized ones to your server. 
  * (1.7k) https://github.com/ory/oathkeeper ORY Oathkeeper 

Other links:
* (4.1k) https://github.com/dwyl/learn-json-web-tokens/ -  Learn how to use JSON Web Token (JWT) to secure your next Web App! (Tutorial/Example with Tests)

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
* https://github.com/uber-go/zap fast logging library for Go Zap
* http://opentracing.io/ - OpenTracing standard
* CNCF Jaeger, a Distributed Tracing System  https://github.com/jaegertracing/jaeger https://uber.github.io/jaeger/ https://jaegertracing.io/
* https://github.com/lightstep/lightstep-tracer-go Lightstep 
* https://github.com/sourcegraph/appdash Application tracing system for Go, based on Google's Dapper. (OpenTracing) https://sourcegraph.com

### Computing

* (6.5k) https://github.com/dask/dask Parallel computing with task scheduling
* (1k) https://github.com/dask/distributed distributed dask
* (11.2) https://github.com/ray-project/ray A system for parallel and distributed Python that unifies the ML ecosystem (similar to Dask)
* (3.0k) https://github.com/arrayfire/arrayfire a general purpose GPU library
* https://github.com/databricks/spark-sklearn [ARCHIVED] Scikit-learn integration package for Spark

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
