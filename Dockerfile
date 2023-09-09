####################
# Base stage
####################
# Please choose the appropriate base image for your project
FROM python:3.8-slim AS base

# Install base packages
COPY requirements/base.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

# Install basics
RUN apt-get update && \
    apt-get install -y make

# Install opencv dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-dev libglib2.0-0

####################
# Check stage
####################
FROM base AS check

# Install check packages
COPY requirements/check.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

####################
# Develop stage
####################
FROM check AS dev

ARG USER_ID
ARG USER_NAME
ARG USER_HOME
ARG GROUP_ID
ARG GROUP_NAME

# Set timezone
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install linux packages
RUN apt-get update && apt-get install -y gnupg
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get install -y --no-install-recommends \
        zsh \
        vim \
        tmux \
        sudo \
        git \
        wget \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Set locale
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Set user
RUN mkdir -p $(dirname "$USER_HOME") && \
    groupadd -g $GROUP_ID $GROUP_NAME && \
    useradd -g $GROUP_ID -u $USER_ID -m -d $USER_HOME $USER_NAME
RUN echo "$USER_NAME:$USER_NAME" | chpasswd
RUN echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

WORKDIR $USER_HOME
USER $USER_NAME:$GROUP_NAME
