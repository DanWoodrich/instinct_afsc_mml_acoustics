#since it includes contrib dependences, put in contrib.
#however, should be built from the INSTINCT root. This dockerfile will assume relative paths to the INSTINCT root
#also, dockerfile will assume that build user has pulled in instinct_afsc_mml_acoustics into contrib

#other pre-build steps: define the following /etc/instinct.cfg:
#blah blah, make sure keypath is predictable to can provide

#at runtime: assume that on instance (like in batch), populate the cache locally if working off a previous step, and populate the local .nt file.

#ubuntu image with nvidia drivers for gcp server gpus. Look below to see specific libraries requested for GPU but ignore any that shouldn't affect reproducibility of tensorflow/keras inference
FROM ...

WORKDIR /app 

COPY . .

#install and set as default python 3.11.4, R ver 4.1.2.

#pip install requirements .txt and ./lib/user/requirements.txt
#Rscript run ./lib/user/Installe.R

#install conda: 

#init two conda environments: tf-gpu4 and mel-spec

#tf-gpu4 conda: init with whatever of these are essential to ensure reprocibility (leave floating to latest any that are not, and/or any that are covered by base image omit)
#python=3.10, cuda-nvcc, cudnn=8.1.0, cudatoolkit=11.2

#from with tf-gpu conda env, pip install tf_gpu4_requirements.txt

#mel-spec conda: init with mel-spec.yml

#install matlab runtime R2025b, ensure ENV variables set to be able to run matlab executible later

#define 

ENTRYPOINT = ["/bin/sh/","-c","python3 .//lib/user/pull_secrets_write.py expected_path.R && exec ./bin/instinct \"@\""]

#define command at run time, job and parameters .nt gcs path. 