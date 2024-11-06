FROM nvcr.io/nvidia/pytorch:23.12-py3
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
RUN mkdir -p /etc/sudoers.d
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

#RUN VER = $(lsb_release -c | awk '{print $2}')
#RUN echo 'deb http://deb.debian.org/debian jammy main contrib non-free' > /etc/apt/sources.list
#RUN  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 04EE7237B7D453EC
#RUN echo "deb http://archive.debian.org/debian stretch main contrib non-free" > /etc/apt/sources.list
#RUN add-apt-repository ppa:swi-prolog/stable
#RUN dpkg --configure -a
#RUN apt-get update
#RUN pip install poetry==1.3.2

# Install necessary packages
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends bash nano && \
#    rm -rf /var/lib/apt/lists/*

    # Customize bash prompt
#RUN echo "PS1='\[\e[1;34m\]\u@\h:\w\$ \[\e[0m\]'" >> /root/.bashrc

# Set the default command to bash
CMD ["bash"]

#RUN apt install -y sudo

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
#RUN pip install --upgrade pip


#COPY ../requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
RUN bash
