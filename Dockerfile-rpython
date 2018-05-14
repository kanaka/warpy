FROM ubuntu:xenial
MAINTAINER Joel Martin <github@martintribe.org>

##########################################################
# General requirements for testing or common across many
# implementations
##########################################################

RUN apt-get -y update

# Required for running tests
RUN apt-get -y install make python

# Some typical implementation and test requirements
RUN apt-get -y install curl libreadline-dev libedit-dev

RUN mkdir -p /mal
WORKDIR /mal

##########################################################
# Specific implementation requirements
##########################################################

# Not enough because doesn't provide rpython
#RUN apt-get -y install bzip2
#RUN mkdir -p /opt/pypy && \
#    cd /opt/pypy && \
#    curl -L https://bitbucket.org/pypy/pypy/downloads/pypy2-v6.0.0-linux64.tar.bz2 | tar -xjf -

# For building rpython
RUN apt-get -y install g++

# pypy
RUN apt-get -y install software-properties-common
RUN add-apt-repository ppa:pypy
RUN apt-get -y update
RUN apt-get -y install pypy

# rpython deps
RUN apt-get -y install mercurial libffi-dev pkg-config libz-dev libbz2-dev \
    libsqlite3-dev libncurses-dev libexpat1-dev libssl-dev libgdbm-dev tcl-dev


RUN mkdir -p /opt/pypy && \
    curl -L https://bitbucket.org/pypy/pypy/downloads/pypy2-v6.0.0-src.tar.bz2 | tar -xjf - -C /opt/pypy/ --strip-components=1 && \
    cd /opt/pypy && make && \
    chmod -R ugo+rw /opt/pypy/rpython/_cache && \
    ln -sf /opt/pypy/rpython/bin/rpython /usr/local/bin/rpython && \
    ln -sf /opt/pypy/pypy-c /usr/local/bin/pypy && \
    rm -rf /tmp/usession*

    #curl https://bitbucket.org/pypy/pypy/get/tip.tar.gz | tar -xzf - -C /opt/pypy/ --strip-components=1

# TODO: this should be combined/updated in the above
RUN ln -sf /opt/pypy/pypy/goal/pypy-c /usr/local/bin/pypy

RUN apt-get -y autoremove pypy
