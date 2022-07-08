FROM rocker/r-ver:latest

LABEL envs="[{\"name\": \"DATA_LOCATION\", \"type\": \"string\", \"required\": true}]"

WORKDIR /usr/local/src/myScripts

## Copy files
COPY  . .

# Install debian packages
RUN apt-get update -qq && apt-get install -y libxml2-dev libcurl4-openssl-dev libssl-dev

## Run the install packages script
RUN Rscript install_packages.R

## Run the script
RUN pwd && ls
CMD Rscript fetal_health_classfication.R
